import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pathlib import Path
from pickle import dump, load
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

st.title("🔗 Sistema de Detección de SPAM en URLs")
st.markdown("""
We want to implement a system that is able to automatically detect whether a web page contains spam 
or not based on its URL. This app walks through the entire NLP pipeline using an SVM classifier.
""")

# ---------- Load Saved Model ----------
model_path = Path('./models/svm_url_spam_model.pkl')
vectorizer_path = Path('./models/tfidf_url_spam_vectorizer.pkl')

USE_SAVED_MODEL = model_path.exists() and vectorizer_path.exists()

if USE_SAVED_MODEL:
    with open(model_path, 'rb') as f:
        saved_model = load(f)
    with open(vectorizer_path, 'rb') as f:
        saved_tfidf = load(f)
    st.sidebar.success("✓ Saved model loaded successfully!")
    st.sidebar.markdown(f"**Model:** `{model_path.name}`")
    st.sidebar.markdown(f"**Vectorizer:** `{vectorizer_path.name}`")
    st.sidebar.markdown(f"**Model params:** {saved_model.get_params()}")
else:
    st.sidebar.warning("No saved model found. Training from scratch.")

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


# ---------- Step 1: Load Data ----------
st.header("📊 Step 1: Loading the Dataset")
st.markdown("""
The dataset contains two columns: `url` and `is_spam`. The `is_spam` column indicates whether 
the URL is spam (1) or not (0).
""")

data_path = Path('data/raw/url_spam.csv')

if not data_path.exists():
    st.error(f"Dataset not found at `{data_path}`. Please ensure the file exists.")
    st.stop()

df = pd.read_csv(data_path)

col1, col2 = st.columns(2)
with col1:
    st.metric("Total URLs", df.shape[0])
    st.metric("Features", df.shape[1])
with col2:
    st.metric("Spam URLs", int(df['is_spam'].sum()))
    st.metric("Not Spam URLs", int((~df['is_spam'].astype(bool)).sum()))

st.subheader("Sample Data")
st.dataframe(df.head(10), use_container_width=True)

st.subheader("Label Distribution")
fig_dist, ax_dist = plt.subplots(figsize=(6, 4))
df['is_spam'].value_counts().plot(kind='bar', ax=ax_dist, color=['steelblue', 'salmon'])
ax_dist.set_xticklabels(['Not Spam (0)', 'Spam (1)'], rotation=0)
ax_dist.set_ylabel('Count')
ax_dist.set_title('Label Distribution')
st.pyplot(fig_dist)


# ---------- Step 2: Preprocess ----------
st.header("🔧 Step 2: Preprocess the URLs")
st.markdown("""
The preprocessing pipeline:
1. Remove protocol (`http://` or `https://`)
2. Tokenize by splitting on URL punctuation marks (`/ . - _ = ? & # : @`)
3. Remove stopwords
4. Lemmatize tokens
5. Vectorize with TF-IDF
""")

# Show preprocessing example
with st.expander("🔍 See preprocessing examples"):
    sample_urls = df['url'].head(5).tolist()
    for url in sample_urls:
        tokens = tokenize_url(url)
        clean = remove_stopwords(tokens)
        lemmatized = lemmatize_tokens(clean)
        st.markdown(f"**URL:** `{url[:80]}...`")
        st.markdown(f"  - Tokens: `{tokens[:10]}`")
        st.markdown(f"  - After stopwords: `{clean[:10]}`")
        st.markdown(f"  - After lemmatization: `{lemmatized[:10]}`")
        st.markdown("---")

# Full preprocessing
df['is_spam'] = df['is_spam'].astype(int)
df['processed_url'] = df['url'].apply(preprocess_url)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['processed_url'])
y = df['is_spam']

st.success(f"✓ TF-IDF matrix shape: {X.shape}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

col1, col2 = st.columns(2)
with col1:
    st.metric("Training Samples", X_train.shape[0])
with col2:
    st.metric("Test Samples", X_test.shape[0])


# ---------- Step 3: Default SVM ----------
st.header("🤖 Step 3: SVM with Default Parameters")
st.markdown("Training a Support Vector Machine with default hyperparameters (RBF kernel, C=1.0).")

@st.cache_resource
def train_default_svm(_X_train, _y_train):
    model = SVC(random_state=42)
    model.fit(_X_train, _y_train)
    return model

svm_model = train_default_svm(X_train, y_train)

y_pred_train = svm_model.predict(X_train)
y_pred_test = svm_model.predict(X_test)

col1, col2 = st.columns(2)
with col1:
    st.metric("Training Accuracy", f"{accuracy_score(y_train, y_pred_train):.4f}")
with col2:
    st.metric("Test Accuracy", f"{accuracy_score(y_test, y_pred_test):.4f}")

st.subheader("Classification Report (Test Set)")
report = classification_report(y_test, y_pred_test, target_names=['Not Spam', 'Spam'], output_dict=True)
st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred_test)
fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
            xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam'])
ax_cm.set_xlabel('Predicted')
ax_cm.set_ylabel('Actual')
ax_cm.set_title('Confusion Matrix - SVM (Default Parameters)')
st.pyplot(fig_cm)


# ---------- Step 4: Optimize SVM ----------
st.header("⚡ Step 4: Hyperparameter Optimization")
st.markdown("Optimizing the SVM using Grid Search with 5-fold cross-validation.")

param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

st.json(param_grid)

@st.cache_resource
def train_optimized_svm(_X_train, _y_train):
    grid_search = GridSearchCV(
        SVC(random_state=42),
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=0,
        return_train_score=True
    )
    grid_search.fit(_X_train, _y_train)
    return grid_search

with st.spinner("Running Grid Search... This may take a moment."):
    grid_search = train_optimized_svm(X_train, y_train)

st.success(f"✓ Best parameters: {grid_search.best_params_}")
st.info(f"Best CV F1-score: {grid_search.best_score_:.4f}")

# Results table
results_df = pd.DataFrame(grid_search.cv_results_)
cols = ['param_C', 'param_kernel', 'param_gamma', 'mean_train_score', 'mean_test_score', 'rank_test_score']
st.subheader("All Combinations (ranked by F1-score)")
st.dataframe(
    results_df[cols].sort_values('rank_test_score').reset_index(drop=True),
    use_container_width=True
)

# Optimized model evaluation
best_model = grid_search.best_estimator_
y_pred_opt = best_model.predict(X_test)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Default Accuracy", f"{accuracy_score(y_test, y_pred_test):.4f}")
with col2:
    st.metric("Optimized Accuracy", f"{accuracy_score(y_test, y_pred_opt):.4f}")
with col3:
    improvement = (accuracy_score(y_test, y_pred_opt) - accuracy_score(y_test, y_pred_test)) * 100
    st.metric("Improvement", f"{improvement:+.2f}%")

# Confusion matrix comparison
st.subheader("Confusion Matrix Comparison")
fig_comp, axes = plt.subplots(1, 2, figsize=(14, 5))

cm_opt = confusion_matrix(y_test, y_pred_opt)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam'])
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_title('Default SVM (RBF, C=1.0)')

sns.heatmap(cm_opt, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam'])
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_title(f"Optimized SVM ({grid_search.best_params_})")

plt.tight_layout()
st.pyplot(fig_comp)

# Optimized classification report
st.subheader("Optimized Model - Classification Report")
report_opt = classification_report(y_test, y_pred_opt, target_names=['Not Spam', 'Spam'], output_dict=True)
st.dataframe(pd.DataFrame(report_opt).transpose(), use_container_width=True)


# ---------- Step 5: Save Model ----------
st.header("💾 Step 5: Save the Model")

model_dir = Path('../models')
model_dir.mkdir(parents=True, exist_ok=True)

if st.button("Save Model & Vectorizer"):
    with open(model_dir / 'svm_url_spam_model.pkl', 'wb') as f:
        dump(best_model, f)
    with open(model_dir / 'tfidf_url_spam_vectorizer.pkl', 'wb') as f:
        dump(tfidf, f)
    st.success(f"✓ Model saved to `{model_dir / 'svm_url_spam_model.pkl'}`")
    st.success(f"✓ Vectorizer saved to `{model_dir / 'tfidf_url_spam_vectorizer.pkl'}`")


# ---------- Interactive Prediction ----------
st.header("🔮 Try it yourself!")
st.markdown("Enter a URL below to predict whether it's spam or not.")

# Use saved model if available, otherwise use the one just trained
if USE_SAVED_MODEL:
    predict_model = saved_model
    predict_tfidf = saved_tfidf
    st.info("Using the **saved model** for predictions.")
else:
    predict_model = best_model
    predict_tfidf = tfidf
    st.info("Using the **newly trained model** for predictions.")

user_url = st.text_input("Enter a URL:", placeholder="https://example.com/some-page")

if user_url:
    processed = preprocess_url(user_url)
    vectorized = predict_tfidf.transform([processed])
    prediction = predict_model.predict(vectorized)[0]
    
    if prediction == 1:
        st.error("🚨 **SPAM** — This URL is likely spam!")
    else:
        st.success("✅ **NOT SPAM** — This URL appears to be legitimate.")
    
    with st.expander("See preprocessing details"):
        st.write(f"**Original URL:** {user_url}")
        st.write(f"**Tokens:** {tokenize_url(user_url)}")
        st.write(f"**Processed:** {processed}")


# ---------- Test with Custom Dataset ----------
st.header("📁 Test with a Custom Dataset")
st.markdown("""
Upload a CSV file to batch-test the model. The file should have at least a column named `url`.  
If it also has an `is_spam` column (with 0/1 or True/False labels), the app will show evaluation metrics.
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
        X_custom = predict_tfidf.transform(test_df['processed_url'])
        test_df['prediction'] = predict_model.predict(X_custom)
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
        
        # Color-coded results table
        st.dataframe(
            test_df[['url', 'prediction_label']].style.apply(
                lambda row: ['background-color: #ffcccc' if row['prediction_label'] == 'Spam' 
                            else 'background-color: #ccffcc'] * len(row), axis=1
            ),
            use_container_width=True
        )
        
        # If ground truth labels exist, show evaluation metrics
        if 'is_spam' in test_df.columns:
            st.subheader("📊 Evaluation against Ground Truth")
            y_true = test_df['is_spam'].astype(int)
            y_pred_custom = test_df['prediction']
            
            acc = accuracy_score(y_true, y_pred_custom)
            st.metric("Accuracy on Custom Dataset", f"{acc:.4f}")
            
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
            ax_custom.set_title('Confusion Matrix - Custom Dataset')
            st.pyplot(fig_custom)
        
        # Download predictions
        csv_output = test_df[['url', 'prediction', 'prediction_label']].to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv_output,
            file_name="url_spam_predictions.csv",
            mime="text/csv"
        )
