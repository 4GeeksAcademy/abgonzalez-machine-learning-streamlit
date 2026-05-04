import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pathlib import Path
from pickle import load
import warnings

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# Disable SSL certificate validation
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# ---------- Page Config ----------
st.set_page_config(
    page_title="URL Spam Detection - NLP",
    page_icon="🔗",
    layout="wide"
)

st.title("🔗 URL Spam Detection")
st.markdown("""
This app uses a pre-trained SVM model to predict whether URLs are **spam** or **legitimate**.  
Upload a CSV dataset or enter a single URL to get predictions.
""")

# ---------- Load Saved Model ----------
model_path = Path('./models/svm_url_spam_model.pkl')
vectorizer_path = Path('./models/tfidf_url_spam_vectorizer.pkl')

if not model_path.exists() or not vectorizer_path.exists():
    st.error("❌ Model files not found. Please ensure `models/svm_url_spam_model.pkl` and `models/tfidf_url_spam_vectorizer.pkl` exist.")
    st.stop()

with open(model_path, 'rb') as f:
    model = load(f)
with open(vectorizer_path, 'rb') as f:
    tfidf = load(f)

# Sidebar - Model info
st.sidebar.success("✓ Model loaded successfully!")
st.sidebar.markdown("### Model Info")
st.sidebar.markdown(f"**Model:** `{model_path.name}`")
st.sidebar.markdown(f"**Vectorizer:** `{vectorizer_path.name}`")
params = model.get_params()
st.sidebar.markdown(f"**Kernel:** `{params.get('kernel')}`")
st.sidebar.markdown(f"**C:** `{params.get('C')}`")
st.sidebar.markdown(f"**Gamma:** `{params.get('gamma')}`")

# ---------- Helper Functions ----------
def tokenize_url(url):
    """Tokenize a URL by splitting on punctuation marks."""
    url = re.sub(r'https?://', '', url)
    tokens = re.split(r'[/.\-_=?&#:@%+~]', url)
    tokens = [t.lower() for t in tokens if t and not t.isdigit()]
    return tokens

def remove_stopwords(tokens):
    """Remove English stopwords from token list."""
    stop_words = set(stopwords.words('english'))
    return [t for t in tokens if t not in stop_words]

def lemmatize_tokens(tokens):
    """Lemmatize tokens using WordNet lemmatizer."""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(t) for t in tokens]

def preprocess_url(url):
    """Full preprocessing pipeline for a single URL."""
    tokens = tokenize_url(url)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    return ' '.join(tokens)


# ---------- Single URL Prediction ----------
st.header("🔮 Predict a Single URL")

user_url = st.text_input("Enter a URL:", placeholder="https://example.com/some-page")

if user_url:
    processed = preprocess_url(user_url)
    vectorized = tfidf.transform([processed])
    prediction = model.predict(vectorized)[0]
    
    if prediction == 1:
        st.error("🚨 **SPAM** — This URL is likely spam!")
    else:
        st.success("✅ **NOT SPAM** — This URL appears to be legitimate.")
    
    with st.expander("See preprocessing details"):
        st.write(f"**Original URL:** {user_url}")
        st.write(f"**Tokens:** {tokenize_url(user_url)}")
        st.write(f"**Processed:** {processed}")


# ---------- Batch Prediction with Upload ----------
st.header("📁 Predict from a Dataset")
st.markdown("""
Upload a CSV file with a column named `url` to get batch predictions.  
If the file also has an `is_spam` column (0/1), evaluation metrics will be shown.
""")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.dataframe(test_df.head(10), use_container_width=True)
    
    if 'url' not in test_df.columns:
        st.error("The CSV must have a column named `url`.")
    else:
        # Preprocess and predict
        test_df['processed_url'] = test_df['url'].apply(preprocess_url)
        X_custom = tfidf.transform(test_df['processed_url'])
        test_df['prediction'] = model.predict(X_custom)
        test_df['prediction_label'] = test_df['prediction'].map({0: 'Not Spam', 1: 'Spam'})
        
        # Show results
        st.subheader("Prediction Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total URLs", len(test_df))
            st.metric("Predicted Spam", int(test_df['prediction'].sum()))
        with col2:
            st.metric("Predicted Not Spam", int((test_df['prediction'] == 0).sum()))
            spam_pct = test_df['prediction'].mean() * 100
            st.metric("Spam %", f"{spam_pct:.1f}%")
        
        # Results table
        st.subheader("All Predictions")
        st.dataframe(
            test_df[['url', 'prediction_label']],
            use_container_width=True
        )
        
        # If ground truth labels exist, show evaluation metrics
        if 'is_spam' in test_df.columns:
            st.subheader("📊 Evaluation against Ground Truth")
            y_true = test_df['is_spam'].astype(int)
            y_pred_custom = test_df['prediction']
            
            acc = accuracy_score(y_true, y_pred_custom)
            st.metric("Accuracy", f"{acc:.4f}")
            
            st.markdown("**Classification Report:**")
            report_custom = classification_report(y_true, y_pred_custom, 
                                                   target_names=['Not Spam', 'Spam'], output_dict=True)
            st.dataframe(pd.DataFrame(report_custom).transpose(), use_container_width=True)
            
            # Confusion matrix
            cm_custom = confusion_matrix(y_true, y_pred_custom)
            fig_custom, ax_custom = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm_custom, annot=True, fmt='d', cmap='Purples', ax=ax_custom,
                        xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam'])
            ax_custom.set_xlabel('Predicted')
            ax_custom.set_ylabel('Actual')
            ax_custom.set_title('Confusion Matrix')
            st.pyplot(fig_custom)
        
        # Download predictions
        csv_output = test_df[['url', 'prediction', 'prediction_label']].to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv_output,
            file_name="url_spam_predictions.csv",
            mime="text/csv"
        )

