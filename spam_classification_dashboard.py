import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
import re
import pickle
import io
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="📧 Spam Classification Dashboard",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #ff6b6b 0%, #ee5a24 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .section-header {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        border-bottom: 3px solid #ff6b6b;
        padding-bottom: 0.5rem;
    }
    
    .info-box {
        background-color: #f8f9fa;
        border-left: 4px solid #ff6b6b;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .stSelectbox, .stTextInput, .stNumberInput {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Load and preprocess data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("spam_ham_dataset.csv")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Text preprocessing function
def preprocess_text(text):
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove special characters and extra whitespace
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Data preprocessing function
def preprocess_data(df):
    if df is None:
        return None
    
    # Create a copy to avoid modifying original data
    df_processed = df.copy()
    
    # Clean column names
    df_processed.columns = df_processed.columns.str.strip()
    
    # Remove unnamed index column if exists
    if 'Unnamed: 0' in df_processed.columns:
        df_processed = df_processed.drop('Unnamed: 0', axis=1)
    
    # Preprocess text
    df_processed['text_cleaned'] = df_processed['text'].apply(preprocess_text)
    
    # Add text features
    df_processed['text_length'] = df_processed['text'].str.len()
    df_processed['word_count'] = df_processed['text'].str.split().str.len()
    df_processed['avg_word_length'] = df_processed['text'].str.split().apply(
        lambda x: np.mean([len(word) for word in x]) if x else 0
    )
    
    # Add spam indicators
    spam_words = ['free', 'money', 'click', 'offer', 'limited', 'act now', 'urgent', 'winner', 'prize']
    df_processed['spam_word_count'] = df_processed['text_cleaned'].apply(
        lambda x: sum(1 for word in spam_words if word in x.lower())
    )
    
    # Add exclamation and question marks count
    df_processed['exclamation_count'] = df_processed['text'].str.count(r'\!')
    df_processed['question_count'] = df_processed['text'].str.count(r'\?')
    df_processed['uppercase_count'] = df_processed['text'].str.count(r'[A-Z]')
    
    # Add URL count
    df_processed['url_count'] = df_processed['text'].str.count(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    
    return df_processed

# Main function
def main():
    # Header
    st.markdown('<h1 class="main-header">📧 Spam Classification Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎛️ Dashboard Controls")
    
    # Load data
    with st.spinner("Loading spam/ham data..."):
        df = load_data()
    
    if df is None:
        st.error("Failed to load data. Please check if the spam_ham_dataset.csv file is in the correct location.")
        return
    
    # Preprocess data
    with st.spinner("Preprocessing email data..."):
        df_processed = preprocess_data(df)
    
    if df_processed is None:
        st.error("Failed to preprocess data.")
        return
    
    # Sidebar filters
    st.sidebar.subheader("📋 Filters")
    
    # Label filter
    labels = ['All'] + list(df_processed['label'].unique())
    selected_label = st.sidebar.selectbox("Select Label", labels)
    
    # Text length filter
    text_length_min, text_length_max = st.sidebar.slider(
        "Text Length Range",
        min_value=int(df_processed['text_length'].min()),
        max_value=int(df_processed['text_length'].max()),
        value=(int(df_processed['text_length'].min()), int(df_processed['text_length'].max()))
    )
    
    # Word count filter
    word_count_min, word_count_max = st.sidebar.slider(
        "Word Count Range",
        min_value=int(df_processed['word_count'].min()),
        max_value=int(df_processed['word_count'].max()),
        value=(int(df_processed['word_count'].min()), int(df_processed['word_count'].max()))
    )
    
    # Apply filters
    filtered_df = df_processed.copy()
    if selected_label != 'All':
        filtered_df = filtered_df[filtered_df['label'] == selected_label]
    filtered_df = filtered_df[
        (filtered_df['text_length'] >= text_length_min) & (filtered_df['text_length'] <= text_length_max) &
        (filtered_df['word_count'] >= word_count_min) & (filtered_df['word_count'] <= word_count_max)
    ]
    
    # Main content with dropdowns
    st.markdown('<h2 class="section-header">📊 Dashboard Sections</h2>', unsafe_allow_html=True)
    
    # Data Exploration Section
    with st.expander("📊 Email Data Exploration", expanded=True):
        st.markdown('<h2 class="section-header">📊 Email Data Exploration</h2>', unsafe_allow_html=True)
        
        # Dataset overview
        st.subheader("📋 Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"**Total Emails:** {len(filtered_df):,}")
            st.info(f"**Total Features:** {len(filtered_df.columns)}")
            st.info(f"**Label Distribution:** {len(filtered_df['label'].unique())} classes")
        
        with col2:
            st.info(f"**Ham Emails:** {len(filtered_df[filtered_df['label'] == 'ham'])}")
            st.info(f"**Spam Emails:** {len(filtered_df[filtered_df['label'] == 'spam'])}")
            st.info(f"**Avg Text Length:** {filtered_df['text_length'].mean():.0f} chars")
        
        with col3:
            st.info(f"**Missing Values:** {filtered_df.isnull().sum().sum()}")
            st.info(f"**Memory Usage:** {filtered_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            st.info(f"**Avg Word Count:** {filtered_df['word_count'].mean():.1f}")
        
        # Data structure
        st.subheader("🏗️ Data Structure")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Data Types:**")
            dtype_df = pd.DataFrame({
                'Column': filtered_df.dtypes.index,
                'Data Type': filtered_df.dtypes.values.astype(str)
            })
            st.dataframe(dtype_df, use_container_width=True)
        
        with col2:
            st.write("**Sample Data:**")
            st.dataframe(filtered_df[['label', 'text_length', 'word_count', 'spam_word_count']].head(10), use_container_width=True)
        
        # Column descriptions
        st.subheader("📝 Feature Descriptions")
        
        column_descriptions = {
            'label': 'Email classification (ham/spam)',
            'text': 'Original email content',
            'label_num': 'Numeric label (0=ham, 1=spam)',
            'text_cleaned': 'Preprocessed text content',
            'text_length': 'Number of characters in email',
            'word_count': 'Number of words in email',
            'avg_word_length': 'Average length of words',
            'spam_word_count': 'Count of common spam words',
            'exclamation_count': 'Number of exclamation marks',
            'question_count': 'Number of question marks',
            'uppercase_count': 'Number of uppercase letters',
            'url_count': 'Number of URLs in email'
        }
        
        desc_df = pd.DataFrame(list(column_descriptions.items()), columns=['Feature', 'Description'])
        st.dataframe(desc_df, use_container_width=True)
    
    # Overview Dashboard Section
    with st.expander("📈 Email Overview", expanded=False):
        st.markdown('<h2 class="section-header">📈 Email Overview</h2>', unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Emails</h3>
                <h2>{len(filtered_df):,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            ham_count = len(filtered_df[filtered_df['label'] == 'ham'])
            st.markdown(f"""
            <div class="metric-card">
                <h3>Ham Emails</h3>
                <h2>{ham_count:,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            spam_count = len(filtered_df[filtered_df['label'] == 'spam'])
            st.markdown(f"""
            <div class="metric-card">
                <h3>Spam Emails</h3>
                <h2>{spam_count:,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_length = filtered_df['text_length'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3>Avg Length</h3>
                <h2>{avg_length:.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Additional metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_words = filtered_df['word_count'].mean()
            st.metric("Avg Word Count", f"{avg_words:.1f}")
        
        with col2:
            avg_spam_words = filtered_df['spam_word_count'].mean()
            st.metric("Avg Spam Words", f"{avg_spam_words:.2f}")
        
        with col3:
            avg_exclamations = filtered_df['exclamation_count'].mean()
            st.metric("Avg Exclamations", f"{avg_exclamations:.2f}")
        
        with col4:
            avg_urls = filtered_df['url_count'].mean()
            st.metric("Avg URLs", f"{avg_urls:.2f}")
        
        # Email distribution
        st.markdown('<h3 class="section-header">📋 Email Distribution</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Label Distribution")
            label_counts = filtered_df['label'].value_counts()
            fig_label_pie = px.pie(
                values=label_counts.values,
                names=label_counts.index,
                title="Email Label Distribution",
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            st.plotly_chart(fig_label_pie, use_container_width=True)
        
        with col2:
            st.subheader("Text Length Distribution")
            fig_length_hist = px.histogram(
                filtered_df,
                x='text_length',
                color='label',
                title="Text Length Distribution by Label",
                nbins=30,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_length_hist, use_container_width=True)
    
    # EDA Section
    with st.expander("🔍 Exploratory Data Analysis", expanded=False):
        st.markdown('<h2 class="section-header">🔍 Exploratory Data Analysis</h2>', unsafe_allow_html=True)
        
        # Distribution analysis
        st.subheader("📊 Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Word count distribution
            fig_word_count = px.histogram(
                filtered_df, 
                x='word_count', 
                nbins=30,
                title="Word Count Distribution",
                color_discrete_sequence=['#ff6b6b'],
                marginal='box'
            )
            fig_word_count.update_layout(showlegend=False)
            st.plotly_chart(fig_word_count, use_container_width=True)
        
        with col2:
            # Spam word count distribution
            fig_spam_words = px.histogram(
                filtered_df, 
                x='spam_word_count', 
                nbins=20,
                title="Spam Word Count Distribution",
                color_discrete_sequence=['#ee5a24'],
                marginal='box'
            )
            fig_spam_words.update_layout(showlegend=False)
            st.plotly_chart(fig_spam_words, use_container_width=True)
        
        # Correlation analysis
        st.subheader("🔗 Correlation Analysis")
        
        # Select numeric columns for correlation
        numeric_cols = ['text_length', 'word_count', 'avg_word_length', 'spam_word_count', 
                       'exclamation_count', 'question_count', 'uppercase_count', 'url_count']
        
        # Filter columns that exist in the dataset
        available_cols = [col for col in numeric_cols if col in filtered_df.columns]
        correlation_matrix = filtered_df[available_cols].corr()
        
        # Create correlation heatmap
        fig_corr = px.imshow(
            correlation_matrix,
            title="Email Features Correlation Matrix",
            color_continuous_scale='RdBu_r',
            aspect='auto',
            text_auto=True,
            width=800,
            height=600
        )
        
        fig_corr.update_layout(
            title_x=0.5,
            title_font_size=20,
            xaxis_title="Features",
            yaxis_title="Features",
            font=dict(size=10)
        )
        
        fig_corr.update_coloraxes(
            colorbar_title="Correlation Coefficient",
            colorbar_len=0.8
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Feature comparison by label
        st.subheader("📊 Feature Comparison by Label")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Text length by label
            fig_length_box = px.box(
                filtered_df,
                x='label',
                y='text_length',
                title="Text Length by Label",
                color='label',
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            st.plotly_chart(fig_length_box, use_container_width=True)
        
        with col2:
            # Word count by label
            fig_word_box = px.box(
                filtered_df,
                x='label',
                y='word_count',
                title="Word Count by Label",
                color='label',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig_word_box, use_container_width=True)
    
    # Visualizations Section
    with st.expander("📊 Email Visualizations", expanded=False):
        st.markdown('<h2 class="section-header">📊 Email Visualizations</h2>', unsafe_allow_html=True)
        
        # Text analysis
        st.subheader("📝 Text Analysis")
        
        # Sample of data for visualization
        sample_size = min(100, len(filtered_df))
        sample_df = filtered_df.sample(n=sample_size)
        
        fig_text = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Text Length vs Word Count', 'Spam Words vs Exclamations', 
                          'Text Length Distribution', 'Word Count vs Avg Word Length'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Text Length vs Word Count
        fig_text.add_trace(
            go.Scatter(x=sample_df['text_length'], y=sample_df['word_count'], 
                      mode='markers', name='Length vs Words', 
                      marker=dict(color='#ff6b6b', size=6)),
            row=1, col=1
        )
        
        # Spam Words vs Exclamations
        fig_text.add_trace(
            go.Scatter(x=sample_df['spam_word_count'], y=sample_df['exclamation_count'], 
                      mode='markers', name='Spam vs Exclamations', 
                      marker=dict(color='#ee5a24', size=6)),
            row=1, col=2
        )
        
        # Text Length Distribution
        fig_text.add_trace(
            go.Histogram(x=sample_df['text_length'], name='Text Length',
                        marker_color='#ff9ff3'),
            row=2, col=1
        )
        
        # Word Count vs Avg Word Length
        fig_text.add_trace(
            go.Scatter(x=sample_df['word_count'], y=sample_df['avg_word_length'], 
                      mode='markers', name='Words vs Avg Length',
                      marker=dict(color='#54a0ff', size=6)),
            row=2, col=2
        )
        
        fig_text.update_layout(height=600, showlegend=False, title_text="Text Analysis")
        st.plotly_chart(fig_text, use_container_width=True)
        
        # Interactive analysis
        st.subheader("🎯 Interactive Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_axis = st.selectbox("Select X-axis", ['text_length', 'word_count', 'spam_word_count', 'exclamation_count', 'question_count', 'uppercase_count', 'url_count'])
        
        with col2:
            y_axis = st.selectbox("Select Y-axis", ['avg_word_length', 'spam_word_count', 'exclamation_count', 'question_count', 'uppercase_count', 'url_count'])
        
        with col3:
            color_by = st.selectbox("Color by", ['label'])
        
        fig_scatter = px.scatter(
            filtered_df,
            x=x_axis,
            y=y_axis,
            color=color_by,
            hover_data=['text_length', 'word_count'],
            title=f"{x_axis} vs {y_axis}",
            color_discrete_sequence=px.colors.qualitative.Set3,
            size='text_length',
            size_max=20
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # ML Models Section
    with st.expander("🤖 Spam Classification Models", expanded=False):
        st.markdown('<h2 class="section-header">🤖 Spam Classification Models</h2>', unsafe_allow_html=True)
        
        # Model selection
        st.subheader("🎯 Model Configuration")
        
        # Add helpful information
        st.info("💡 **Tip**: For best results, use 'All' filters or select filters that include diverse data. Very specific filters may result in insufficient data for training.")
        
        # Show current data info
        st.write(f"**Current filtered dataset size:** {len(filtered_df)} records")
        
        col1, col2 = st.columns(2)
        
        with col1:
            vectorizer_type = st.selectbox(
                "Select Text Vectorizer",
                ["TF-IDF", "Count Vectorizer"],
                help="TF-IDF: Better for most cases, Count Vectorizer: Simpler approach"
            )
        
        with col2:
            max_features = st.slider(
                "Max Features",
                min_value=100,
                max_value=5000,
                value=1000,
                help="Maximum number of features to extract from text"
            )
        
        # Show target variable information
        target_col = 'label_num'
        target_counts = filtered_df[target_col].value_counts()
        st.write(f"**Target variable '{target_col}' distribution:**")
        st.dataframe(target_counts, use_container_width=True)
        
        # Feature engineering
        st.subheader("🔧 Feature Engineering")
        
        # Prepare text features
        X_text = filtered_df['text_cleaned']
        y = filtered_df[target_col]
        
        # Prepare numeric features
        numeric_features = ['text_length', 'word_count', 'avg_word_length', 'spam_word_count', 
                           'exclamation_count', 'question_count', 'uppercase_count', 'url_count']
        available_numeric = [col for col in numeric_features if col in filtered_df.columns]
        X_numeric = filtered_df[available_numeric]
        
        # Remove rows with missing values
        mask = ~(X_numeric.isnull().any(axis=1) | pd.isnull(y))
        X_text = X_text[mask]
        X_numeric = X_numeric[mask]
        y = y[mask]
        
        if len(X_text) == 0:
            st.error("No valid data for modeling after removing missing values.")
            return
        
        # Check for classification issues
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            st.error(f"❌ Classification requires at least 2 classes. Found only {len(unique_classes)} class(es) in {target_col}.")
            st.info("💡 Try selecting different filters.")
            return
        
        # Check if any class has too few samples
        class_counts = np.bincount(y)
        min_samples = min(class_counts)
        if min_samples < 10:
            st.warning(f"⚠️ Some classes have very few samples (minimum: {min_samples}). This may affect model performance.")
        
        # Split data
        X_text_train, X_text_test, X_numeric_train, X_numeric_test, y_train, y_test = train_test_split(
            X_text, X_numeric, y, test_size=0.2, random_state=42
        )
        
        # Vectorize text
        if vectorizer_type == "TF-IDF":
            vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        else:
            vectorizer = CountVectorizer(max_features=max_features, stop_words='english')
        
        X_text_train_vec = vectorizer.fit_transform(X_text_train)
        X_text_test_vec = vectorizer.transform(X_text_test)
        
        # Combine text and numeric features
        from scipy.sparse import hstack
        X_train_combined = hstack([X_text_train_vec, X_numeric_train])
        X_test_combined = hstack([X_text_test_vec, X_numeric_test])
        
        # Model training
        st.subheader("🚀 Model Training")
        
        # Classification models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'AdaBoost': AdaBoostClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Multinomial Naive Bayes': MultinomialNB(),
            'Decision Tree': DecisionTreeClassifier(random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            with st.spinner(f"Training {name}..."):
                try:
                    model.fit(X_train_combined, y_train)
                    y_pred = model.predict(X_test_combined)
                    y_pred_proba = model.predict_proba(X_test_combined)[:, 1] if hasattr(model, 'predict_proba') else None
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
                    
                    results[name] = {
                        'model': model,
                        'vectorizer': vectorizer,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'auc': auc,
                        'predictions': y_pred,
                        'predictions_proba': y_pred_proba
                    }
                    
                    # Save to session state for export
                    if 'trained_models' not in st.session_state:
                        st.session_state.trained_models = {}
                    st.session_state.trained_models[name] = results[name]
                    
                except Exception as e:
                    st.error(f"❌ Error training {name}: {str(e)}")
                    continue
        
        # Display results
        if not results:
            st.error("❌ No models were successfully trained. Please check your data and filters.")
            return
            
        st.subheader("📊 Classification Results")
        
        # Model comparison table
        st.subheader("🏆 Model Comparison")
        comparison_data = []
        for name, result in results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': f"{result['accuracy']:.4f}",
                'Precision': f"{result['precision']:.4f}",
                'Recall': f"{result['recall']:.4f}",
                'F1-Score': f"{result['f1']:.4f}",
                'AUC': f"{result['auc']:.4f}" if result['auc'] is not None else "N/A",
                'Status': '✅ Trained Successfully'
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Accuracy comparison
            accuracies = {name: result['accuracy'] for name, result in results.items()}
            fig_acc = px.bar(
                x=list(accuracies.keys()),
                y=list(accuracies.values()),
                title="Model Accuracy Comparison",
                color=list(accuracies.values()),
                color_continuous_scale='Viridis',
                text=list(accuracies.values())
            )
            fig_acc.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig_acc.update_layout(yaxis_title="Accuracy")
            st.plotly_chart(fig_acc, use_container_width=True)
        
        with col2:
            # Best model details
            best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
            best_model = results[best_model_name]
            
            st.markdown(f"**Best Model:** {best_model_name}")
            st.markdown(f"**Accuracy:** {best_model['accuracy']:.4f}")
            st.markdown(f"**Precision:** {best_model['precision']:.4f}")
            st.markdown(f"**Recall:** {best_model['recall']:.4f}")
            st.markdown(f"**F1-Score:** {best_model['f1']:.4f}")
            if best_model['auc'] is not None:
                st.markdown(f"**AUC:** {best_model['auc']:.4f}")
            
            # Feature importance for tree-based models
            if 'Forest' in best_model_name or 'Tree' in best_model_name or 'Boosting' in best_model_name:
                feature_names = list(vectorizer.get_feature_names_out()) + available_numeric
                feature_importance = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': best_model['model'].feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig_importance = px.bar(
                    feature_importance.head(15),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Top 15 Feature Importance",
                    color='Importance',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_importance, use_container_width=True)
        
        # Confusion matrix for best model
        st.subheader("📈 Confusion Matrix")
        
        cm = confusion_matrix(y_test, best_model['predictions'])
        
        # Create confusion matrix heatmap
        fig_cm = ff.create_annotated_heatmap(
            cm,
            x=['Predicted Ham', 'Predicted Spam'],
            y=['Actual Ham', 'Actual Spam'],
            colorscale='Blues',
            showscale=True
        )
        fig_cm.update_layout(title="Confusion Matrix")
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Classification report
        st.subheader("📋 Classification Report")
        report = classification_report(y_test, best_model['predictions'], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)
    
    # Export Section
    with st.expander("📤 Export Data & Models", expanded=False):
        st.markdown('<h2 class="section-header">📤 Export Data & Models</h2>', unsafe_allow_html=True)
        
        # Export cleaned data
        st.subheader("📊 Export Cleaned Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Current Filtered Dataset:**")
            st.write(f"• **Rows:** {len(filtered_df):,}")
            st.write(f"• **Columns:** {len(filtered_df.columns)}")
            st.write(f"• **File Size:** {filtered_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Export CSV
            csv_buffer = io.StringIO()
            filtered_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="📥 Download Cleaned CSV",
                data=csv_data,
                file_name=f"spam_data_cleaned_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Download the cleaned and filtered spam dataset as CSV"
            )
        
        with col2:
            st.write("**Data Summary:**")
            st.write("• Preprocessed and cleaned email data")
            st.write("• Applied text preprocessing")
            st.write("• Handled missing values")
            st.write("• Added derived features")
            
            # Show sample of data to be exported
            st.write("**Sample of data to be exported:**")
            st.dataframe(filtered_df[['label', 'text_length', 'word_count', 'spam_word_count']].head(5), use_container_width=True)
        
        # Export trained models
        st.subheader("🤖 Export Trained Models")
        
        # Check if models were trained in this session
        if 'trained_models' not in st.session_state:
            st.session_state.trained_models = {}
        
        if st.session_state.trained_models:
            st.success("✅ Models available for export!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Available Models:**")
                for model_name, model_info in st.session_state.trained_models.items():
                    st.write(f"• **{model_name}**")
                    st.write(f"  - Accuracy: {model_info['accuracy']:.4f}")
                    st.write(f"  - F1-Score: {model_info['f1']:.4f}")
                    if model_info['auc'] is not None:
                        st.write(f"  - AUC: {model_info['auc']:.4f}")
            
            with col2:
                # Export all models as pickle
                model_data = pickle.dumps(st.session_state.trained_models)
                
                st.download_button(
                    label="📥 Download All Models (PKL)",
                    data=model_data,
                    file_name=f"spam_models_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                    mime="application/octet-stream",
                    help="Download all trained spam classification models as pickle file"
                )
                
                # Export best model separately
                if st.session_state.trained_models:
                    best_model_name = max(st.session_state.trained_models.keys(), 
                                        key=lambda k: st.session_state.trained_models[k]['accuracy'])
                    best_model = st.session_state.trained_models[best_model_name]
                    
                    st.write(f"**Best Model:** {best_model_name}")
                    
                    best_model_data = pickle.dumps(best_model)
                    
                    st.download_button(
                        label=f"📥 Download Best Model ({best_model_name})",
                        data=best_model_data,
                        file_name=f"best_spam_model_{best_model_name.replace(' ', '_')}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                        mime="application/octet-stream",
                        help=f"Download the best performing spam model: {best_model_name}"
                    )
        else:
            st.info("ℹ️ **No models trained yet.** Train models in the 'Spam Classification Models' section to export them.")
            st.write("**To export models:**")
            st.write("1. Go to '🤖 Spam Classification Models' section")
            st.write("2. Select vectorizer type and max features")
            st.write("3. Train the models")
            st.write("4. Return here to export the trained models")
        
        # Export configuration and metadata
        st.subheader("⚙️ Export Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export configuration as JSON
            config_data = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'filters_applied': {
                    'label': selected_label,
                    'text_length_range': [int(text_length_min), int(text_length_max)],
                    'word_count_range': [int(word_count_min), int(word_count_max)]
                },
                'dataset_info': {
                    'total_rows': int(len(filtered_df)),
                    'total_columns': int(len(filtered_df.columns)),
                    'missing_values': int(filtered_df.isnull().sum().sum()),
                    'memory_usage_mb': float(filtered_df.memory_usage(deep=True).sum() / 1024**2)
                },
                'preprocessing_info': {
                    'text_preprocessing_applied': True,
                    'missing_values_handled': True,
                    'derived_features_created': True
                }
            }
            
            import json
            config_json = json.dumps(config_data, indent=2)
            
            st.download_button(
                label="📥 Download Configuration (JSON)",
                data=config_json,
                file_name=f"spam_dashboard_config_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                help="Download the current spam dashboard configuration and metadata"
            )
        
        with col2:
            st.write("**Configuration includes:**")
            st.write("• Applied filters")
            st.write("• Dataset statistics")
            st.write("• Text preprocessing")
            st.write("• Feature engineering")
        
        # Usage instructions
        st.subheader("📖 Usage Instructions")
        
        st.markdown("""
        **How to use exported files:**
        
        **📊 CSV File:**
        ```python
        import pandas as pd
        df = pd.read_csv('spam_data_cleaned.csv')
        ```
        
        **🤖 Model Files:**
        ```python
        import pickle
        
        # Load all models
        with open('spam_models.pkl', 'rb') as f:
            models = pickle.load(f)
        
        # Load best model
        with open('best_spam_model.pkl', 'rb') as f:
            best_model = pickle.load(f)
        
        # Use the model for spam classification
        vectorizer = best_model['vectorizer']
        model = best_model['model']
        
        # Preprocess new text
        new_text = "Your email text here"
        new_text_processed = preprocess_text(new_text)
        
        # Vectorize and predict
        new_text_vec = vectorizer.transform([new_text_processed])
        prediction = model.predict(new_text_vec)
        probability = model.predict_proba(new_text_vec)
        ```
        
        **⚙️ Configuration:**
        ```python
        import json
        with open('spam_dashboard_config.json', 'r') as f:
            config = json.load(f)
        ```
        """)

if __name__ == "__main__":
    main() 