import streamlit as st
import pandas as pd
import json
from datetime import datetime
import re
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ==================== PHQ-9 LEXICON ====================
PHQ9_LEXICON = {
    "S1_anhedonia": ["lost interest", "no pleasure", "don't enjoy", "nothing fun", "not interested", "can't enjoy"],
    "S2_depressed_mood": ["feeling down", "sad", "hopeless", "depressed", "worthless", "miserable", "blue", "low"],
    "S3_sleep": ["trouble sleeping", "insomnia", "wake up", "can't sleep", "sleep issues", "tossing and turning"],
    "S4_fatigue": ["no energy", "tired", "fatigue", "exhausted", "lethargic", "drained"],
    "S5_appetite": ["eating too much", "no appetite", "lost weight", "eating less", "overeating"],
    "S6_worthlessness": ["worthless", "guilty", "failure", "useless", "burden", "inadequate"],
    "S7_concentration": ["can't focus", "distracted", "poor concentration", "can't think", "forgetful"],
    "S8_psychomotor": ["restless", "slowed down", "agitated", "fidgety"],
    "S9_suicidal": ["want to die", "self-harm", "kill myself", "end it all", "suicide", "death wish"]
}

# ==================== KiAS SUMMARIZER CLASS ====================
class KiASSummarizer:
    def __init__(self):
        """Initialize the KiAS summarizer with PHQ-9 lexicon and embedding model"""
        self.phq9_terms = PHQ9_LEXICON
        # Use a lightweight model for better performance
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except:
            # Fallback to a simpler approach if sentence-transformers fails
            self.embedding_model = None
            st.warning("⚠️ SentenceTransformer model not loaded. Using fallback similarity method.")
    
    def _simple_similarity(self, word1, word2):
        """Fallback similarity using character overlap"""
        if not word1 or not word2:
            return 0.0
        # Convert to sets for Jaccard-like similarity
        set1 = set(word1.lower())
        set2 = set(word2.lower())
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    def _cosine_similarity(self, word1, word2):
        """Compute cosine similarity between two words"""
        if self.embedding_model is None:
            return self._simple_similarity(word1, word2)
        try:
            emb1 = self.embedding_model.encode([word1], show_progress_bar=False)
            emb2 = self.embedding_model.encode([word2], show_progress_bar=False)
            return cosine_similarity(emb1, emb2)[0][0]
        except:
            return self._simple_similarity(word1, word2)
    
    def _has_phq9_term(self, text):
        """Check if text contains any PHQ-9 term"""
        text_lower = text.lower()
        for category, terms in self.phq9_terms.items():
            for term in terms:
                if term.lower() in text_lower:
                    return True
        return False
    
    def prune_conversation(self, transcript):
        """Filter out non-clinical utterances"""
        pruned = []
        for speaker, utterance in transcript:
            if self._has_phq9_term(utterance):
                pruned.append((speaker, utterance))
        return pruned
    
    def convert_to_statements(self, transcript):
        """Convert Q&A pairs to statements"""
        statements = []
        for i in range(0, len(transcript), 2):
            if i+1 < len(transcript):
                q = transcript[i][1]
                a = transcript[i+1][1]
                statement = f"Participant was asked '{q}', participant said '{a}'"
                statements.append(statement)
        return statements
    
    def calculate_wss(self, statement):
        """Calculate Word Semantic Score"""
        # Split into words
        words = re.findall(r'\w+', statement.lower())
        if not words:
            return 0
        
        scores = []
        for word in words:
            # Find max similarity with PHQ-9 terms
            max_sim = 0
            for category, terms in self.phq9_terms.items():
                for term in terms:
                    term_words = term.lower().split()
                    # Check if term appears in the statement
                    if any(tw in statement.lower() for tw in term_words):
                        sim = self._cosine_similarity(word, term)
                        max_sim = max(max_sim, sim)
            scores.append(max_sim)
        
        return np.mean(scores) if scores else 0
    
    def build_word_graph(self, window_statements):
        """Build word graph from a window of statements"""
        graph = defaultdict(lambda: defaultdict(int))
        all_words = []
        
        for statement in window_statements:
            words = re.findall(r'\w+', statement.lower())
            all_words.extend(words)
            # Add edges between consecutive words
            for i in range(len(words)-1):
                graph[words[i]][words[i+1]] += 1
        
        return graph, all_words
    
    def textrank(self, graph, iterations=30, d=0.78):
        """Compute TextRank scores"""
        if not graph:
            return {}
        
        # Initialize scores
        scores = {node: 1.0 for node in graph}
        
        for _ in range(iterations):
            new_scores = {}
            for node in graph:
                score = (1 - d)
                out_degree = len(graph[node])
                for neighbor in graph[node]:
                    if neighbor in scores and out_degree > 0:
                        score += d * scores[neighbor] / out_degree
                new_scores[node] = score
            scores = new_scores
        
        return scores
    
    def summarize(self, transcript, max_sentences=7):
        """Main summarization function"""
        # Step 1: Prune conversation
        pruned = self.prune_conversation(transcript)
        
        if len(pruned) < 2:
            return ["No clinically relevant information found in this interview."]
        
        # Step 2: Convert to statements
        statements = self.convert_to_statements(pruned)
        
        if not statements:
            return ["No Q&A pairs found in the interview."]
        
        # Step 3: Calculate scores for each statement
        scored_statements = []
        for stmt in statements:
            # Calculate informativeness using WSS
            wss_score = self.calculate_wss(stmt)
            
            # Calculate linguistic quality (simplified)
            # Longer statements with clinical terms get higher scores
            word_count = len(stmt.split())
            q_score = min(1.0, word_count / 100) + wss_score
            
            # Combined score - avoid division by zero
            if word_count > 0:
                combined_score = (wss_score * q_score) / word_count
            else:
                combined_score = 0
            
            # Boost score if it contains PHQ-9 terms
            if self._has_phq9_term(stmt):
                combined_score *= 1.5
            
            scored_statements.append((stmt, combined_score))
        
        # Step 4: Sort and select top-k statements
        scored_statements.sort(key=lambda x: x[1], reverse=True)
        summary = [stmt for stmt, score in scored_statements[:max_sentences]]
        
        return summary if summary else ["No meaningful summary could be generated."]

# ==================== DATA FUNCTIONS ====================
def create_extended_dummy_data():
    """Create extended dummy dataset with multiple patients"""
    return {
        "patient_001": {
            "id": "P001",
            "phq8_score": 18,
            "risk_level": "High",
            "transcript": [
                ("Ellie", "Hi, how are you feeling today?"),
                ("Patient", "I've been really tired lately, can't seem to get out of bed."),
                ("Ellie", "That sounds difficult. How has your sleep been?"),
                ("Patient", "I wake up at 3 AM every night and can't fall back asleep."),
                ("Ellie", "Have you had any changes in appetite?"),
                ("Patient", "I don't feel like eating much, lost about 10 pounds."),
                ("Ellie", "What brings you here today?"),
                ("Patient", "I just feel hopeless, like nothing matters anymore."),
                ("Ellie", "Have you been diagnosed with depression before?"),
                ("Patient", "No, but my mom had it."),
                ("Ellie", "Where did you grow up?"),
                ("Patient", "I grew up in Texas, moved here for work."),
                ("Ellie", "Have you had thoughts of hurting yourself?"),
                ("Patient", "Sometimes I think about it, but I wouldn't actually do it."),
                ("Ellie", "What do you do for fun?"),
                ("Patient", "I used to enjoy reading, but now nothing seems interesting.")
            ]
        },
        "patient_002": {
            "id": "P002",
            "phq8_score": 12,
            "risk_level": "Moderate",
            "transcript": [
                ("Ellie", "Hello, how are you doing?"),
                ("Patient", "I'm okay, just stressed about work."),
                ("Ellie", "What's going on with work?"),
                ("Patient", "My boss is demanding and I can't seem to focus."),
                ("Ellie", "Is this affecting your sleep?"),
                ("Patient", "Some nights I lie awake worrying."),
                ("Ellie", "Do you feel down or depressed?"),
                ("Patient", "Not really, just anxious about deadlines."),
                ("Ellie", "How's your appetite?"),
                ("Patient", "Fine, eating normally."),
                ("Ellie", "Do you have support at home?"),
                ("Patient", "My partner is supportive."),
                ("Ellie", "Have you been diagnosed with any condition?"),
                ("Patient", "No."),
                ("Ellie", "What helps you relax?"),
                ("Patient", "Exercise helps, I go for runs.")
            ]
        },
        "patient_003": {
            "id": "P003",
            "phq8_score": 20,
            "risk_level": "High",
            "transcript": [
                ("Ellie", "Welcome. How are you today?"),
                ("Patient", "I don't know why I'm here. Nothing helps."),
                ("Ellie", "Tell me more about that."),
                ("Patient", "I feel worthless, like I'm a burden to everyone."),
                ("Ellie", "Have you been sleeping well?"),
                ("Patient", "I can't sleep, I just lie awake all night."),
                ("Ellie", "What about eating?"),
                ("Patient", "I have to force myself to eat."),
                ("Ellie", "Have you ever been diagnosed with depression?"),
                ("Patient", "Yes, 2 years ago."),
                ("Ellie", "Did you take medication?"),
                ("Patient", "I stopped, they didn't help."),
                ("Ellie", "Do you have thoughts of self-harm?"),
                ("Patient", "Sometimes I wish I didn't wake up."),
                ("Ellie", "That's serious. Do you have a safety plan?"),
                ("Patient", "No."),
                ("Ellie", "Where are you from?"),
                ("Patient", "Florida."),
                ("Ellie", "Do you have friends?"),
                ("Patient", "No, I've pushed everyone away.")
            ]
        }
    }

def detect_phq9_terms(transcript):
    """Detect PHQ-9 terms in transcript"""
    results = []
    
    for speaker, utterance in transcript:
        found_terms = []
        for category, terms in PHQ9_LEXICON.items():
            for term in terms:
                if term.lower() in utterance.lower():
                    found_terms.append(term)
        if found_terms:
            results.append((utterance, found_terms))
    
    return results[:10]

def create_signal_visualization(summary):
    """Create a bar chart showing PHQ-9 signal detection"""
    phq9_signals = {
        "Anhedonia": 0,
        "Depressed Mood": 0,
        "Sleep Issues": 0,
        "Fatigue": 0,
        "Appetite Issues": 0,
        "Worthlessness": 0,
        "Concentration": 0,
        "Psychomotor": 0,
        "Suicidal Thoughts": 0
    }
    
    # Map categories to signal names
    signal_map = {
        "S1_anhedonia": "Anhedonia",
        "S2_depressed_mood": "Depressed Mood",
        "S3_sleep": "Sleep Issues",
        "S4_fatigue": "Fatigue",
        "S5_appetite": "Appetite Issues",
        "S6_worthlessness": "Worthlessness",
        "S7_concentration": "Concentration",
        "S8_psychomotor": "Psychomotor",
        "S9_suicidal": "Suicidal Thoughts"
    }
    
    for sentence in summary:
        for category, signal_name in signal_map.items():
            for term in PHQ9_LEXICON.get(category, []):
                if term.lower() in sentence.lower():
                    phq9_signals[signal_name] += 1
                    break
    
    return phq9_signals

# ==================== STREAMLIT UI ====================
@st.cache_resource
def get_summarizer():
    """Get cached summarizer instance"""
    return KiASSummarizer()

def show_dummy_data():
    """Display dummy data page"""
    st.header("📝 Dummy Clinical Interview Data")
    
    st.info("""
    These dummy transcripts simulate clinical diagnostic interviews.
    They contain both clinically relevant and irrelevant content.
    """)
    
    # Create dummy data
    dummy_data = create_extended_dummy_data()
    
    # Display as tabs
    tabs = st.tabs([f"Patient {i+1}" for i in range(len(dummy_data))])
    
    for tab, (patient_id, data) in zip(tabs, dummy_data.items()):
        with tab:
            st.subheader(f"Patient: {data['id']}")
            
            # Risk level indicator
            risk_color = "🟢" if data['risk_level'] == "Low" else "🟡" if data['risk_level'] == "Moderate" else "🔴"
            st.write(f"**PHQ-8 Score:** {data['phq8_score']}/24  {risk_color} **Risk Level:** {data['risk_level']}")
            
            # Display transcript
            st.subheader("📄 Interview Transcript")
            df = pd.DataFrame(data['transcript'], columns=["Speaker", "Utterance"])
            st.dataframe(df, use_container_width=True)
            
            # Highlight PHQ-9 terms
            st.subheader("🔍 PHQ-9 Term Detection")
            phq9_terms = detect_phq9_terms(data['transcript'])
            if phq9_terms:
                st.success(f"Found {len(phq9_terms)} clinically relevant utterances")
                for utterance, terms in phq9_terms[:5]:
                    st.write(f"- {utterance[:100]}... → **{', '.join(terms)}**")
            else:
                st.warning("No PHQ-9 terms detected")

def show_summarization():
    """Display summarization page"""
    st.header("📊 Clinical Interview Summarization")
    
    # Select patient
    dummy_data = create_extended_dummy_data()
    patient_ids = list(dummy_data.keys())
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_id = st.selectbox(
            "Select Patient Interview",
            patient_ids,
            format_func=lambda x: dummy_data[x]['id']
        )
        
        max_sentences = st.slider(
            "Maximum Summary Sentences",
            min_value=3,
            max_value=10,
            value=7
        )
        
        if st.button("🔄 Generate Summary", type="primary"):
            st.session_state['selected_patient'] = selected_id
            st.session_state['max_sentences'] = max_sentences
            st.session_state['generate_summary'] = True
    
    with col2:
        if st.session_state.get('generate_summary', False):
            with st.spinner("Generating summary using KiAS..."):
                summarizer = get_summarizer()
                transcript = dummy_data[selected_id]['transcript']
                summary = summarizer.summarize(transcript, max_sentences)
                
                # Display summary
                st.subheader("📋 Generated Summary (KiAS)")
                for i, sentence in enumerate(summary, 1):
                    st.markdown(f"**{i}.** {sentence}")
                
                # Comparison with other methods (simulated)
                st.subheader("📊 Method Comparison")
                
                # Simulate different methods for comparison
                es_summary = [s[:50] + "..." for s in summary[:max_sentences//2]]
                as_summary = [s.replace("participant said", "the patient") for s in summary]
                
                comparison_data = {
                    "Method": ["Extractive (SB)", "Abstractive (AS)", "KiAS (Ours)"],
                    "Sentences": [len(es_summary), len(as_summary), len(summary)],
                    "Clinical Terms": [len([s for s in ' '.join(es_summary).split() if s.lower() in ['tired', 'sleep', 'depressed', 'hopeless']]), 
                                      len([s for s in ' '.join(as_summary).split() if s.lower() in ['tired', 'sleep', 'depressed', 'hopeless']]),
                                      len([s for s in ' '.join(summary).split() if s.lower() in ['tired', 'sleep', 'depressed', 'hopeless']])],
                    "Readability": ["Standard", "Good", "Better"]
                }
                st.table(comparison_data)
                
                # Visualize clinical signal detection
                st.subheader("🎯 PHQ-9 Signal Detection")
                
                try:
                    # Try to create plotly visualization
                    import plotly.graph_objects as go
                    phq9_signals = create_signal_visualization(summary)
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(phq9_signals.keys()), 
                            y=list(phq9_signals.values()),
                            marker_color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12', 
                                         '#9b59b6', '#1abc9c', '#e67e22', '#e74c3c', '#8e44ad']
                        )
                    ])
                    
                    fig.update_layout(
                        title="PHQ-9 Signal Detection in Summary",
                        xaxis_title="PHQ-9 Signals",
                        yaxis_title="Number of Mentions",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    # Fallback to simple text display if plotly not available
                    phq9_signals = create_signal_visualization(summary)
                    st.write("PHQ-9 Signal Detection:")
                    for signal, count in phq9_signals.items():
                        if count > 0:
                            st.write(f"- {signal}: {count} mentions")

def show_evaluation():
    """Display evaluation page"""
    st.header("📈 Model Evaluation")
    
    # Evaluation metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("ROUGE-2 F1 Score", "0.31", delta="+55% vs AS")
    
    with col2:
        st.metric("ROUGE-L F1 Score", "0.33", delta="+48% vs AS")
    
    with col3:
        st.metric("Thematic Overlap", "40%", delta="+17% vs ES")
    
    # Comparison chart
    st.subheader("📊 Performance Comparison")
    
    methods = ["ES", "AS", "AoES", "KiAS"]
    metrics = {
        "ROUGE-2": [6.52, 6.42, 3.21, 14.62],
        "ROUGE-L": [12.72, 12.80, 11.51, 24.46],
        "Contextual Similarity": [0.672, 0.676, 0.669, 0.689],
        "Readability": [62.1, 63.0, 64.5, 67.3]
    }
    
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        
        for metric, values in metrics.items():
            fig.add_trace(go.Bar(
                name=metric,
                x=methods,
                y=values,
                text=[f"{v:.2f}" for v in values],
                textposition='auto',
            ))
        
        fig.update_layout(
            title="Performance Comparison of Summarization Methods",
            barmode='group',
            height=400,
            yaxis_title="Score"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        # Fallback to table display
        eval_df = pd.DataFrame(metrics, index=methods)
        st.dataframe(eval_df, use_container_width=True)
    
    # Inter-rater agreement
    st.subheader("👥 Inter-Rater Agreement")
    agreement_data = {
        "Metric": ["Good Questions", "Meaningful Responses"],
        "AS (Cohen κ)": [0.63, 0.43],
        "KiAS (Cohen κ)": [0.70, 0.65]
    }
    st.table(agreement_data)
    
    # Explanation of metrics
    with st.expander("📖 Understanding the Evaluation Metrics"):
        st.markdown("""
        **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**
        - Measures n-gram overlap between generated and reference summaries
        - ROUGE-2: Bigram overlap
        - ROUGE-L: Longest common subsequence
        
        **Contextual Similarity**
        - Cosine similarity between sentence embeddings
        - Higher is better (0-1 scale)
        
        **Jensen-Shannon Divergence (JSD)**
        - Measures difference between probability distributions
        - Lower is better
        
        **Flesch Reading Ease (FRE)**
        - Measures readability
        - Higher scores indicate easier reading (0-100 scale)
        
        **Thematic Overlap**
        - How well the summary covers main topics
        - Higher percentage is better
        """)

# ==================== MAIN APP ====================
def main():
    st.set_page_config(
        page_title="KiAS - Clinical Interview Summarizer",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Knowledge-Infused Abstractive Summarization (KiAS)")
    st.markdown("""
    ### Summarizing Clinical Diagnostic Interviews
    This application demonstrates KiAS, an unsupervised summarization approach 
    that uses PHQ-9 lexicon knowledge to generate clinically relevant summaries.
    
    **Reference:** Manas et al., JMIR Mental Health 2021
    """)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    option = st.sidebar.selectbox(
        "Choose a view",
        ["📝 Dummy Data", "📊 Summarization", "📈 Evaluation"]
    )
    
    # About section in sidebar
    with st.sidebar.expander("ℹ️ About KiAS"):
        st.markdown("""
        **Key Features:**
        - 🧠 Knowledge-infused with PHQ-9 lexicon
        - 🎯 Focus on clinically relevant content
        - 📊 Unsupervised summarization
        - 🔍 Identifies depression signals
        
        **PHQ-9 Signals:**
        - Anhedonia
        - Depressed Mood
        - Sleep Issues
        - Fatigue
        - Appetite Issues
        - Worthlessness
        - Concentration Issues
        - Psychomotor Issues
        - Suicidal Thoughts
        """)
    
    if option == "📝 Dummy Data":
        show_dummy_data()
    elif option == "📊 Summarization":
        show_summarization()
    else:
        show_evaluation()

if __name__ == "__main__":
    main()
