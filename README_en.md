# 🔗 URL Spam Detection - Streamlit Web App

A machine learning web application that detects whether a URL is spam or not, using Natural Language Processing (NLP) and a Support Vector Machine (SVM) classifier.

**Live App:** [https://url-spam-detector.onrender.com](https://url-spam-detector.onrender.com)

---

## Features

- Full NLP pipeline visualization (tokenization, stopwords removal, lemmatization, TF-IDF)
- SVM model training with default and optimized hyperparameters
- Interactive single URL prediction
- Batch prediction by uploading a CSV file
- Model saving and loading from disk

---

## Project Structure

```
├── app.py                 # Streamlit application
├── requirements.txt       # Python dependencies
├── render.yaml            # Render deployment configuration
├── setup.sh              # NLTK data download script
├── .gitignore            # Git ignore rules
├── 12-nlp-url-spam.ipynb # Jupyter notebook with the full analysis
├── data/
│   └── raw/
│       └── url_spam.csv  # Dataset
└── models/
    ├── svm_url_spam_model.pkl        # Trained SVM model
    └── tfidf_url_spam_vectorizer.pkl # Fitted TF-IDF vectorizer
```

---

## Run Locally

```bash
# Clone the repository
git clone https://github.com/4GeeksAcademy/abgonzalez-machine-learning-streamlit.git
cd abgonzalez-machine-learning-streamlit

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -m nltk.downloader stopwords wordnet

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Deploy on Render

### Step 1: Push your code to GitHub

Make sure these files are committed and pushed:
- `app.py`
- `requirements.txt`
- `data/raw/url_spam.csv`
- `models/svm_url_spam_model.pkl`
- `models/tfidf_url_spam_vectorizer.pkl`

### Step 2: Create a Web Service on Render

1. Go to [https://render.com](https://render.com) and sign up or log in
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub account and select the repository
4. Configure the service:

| Setting | Value |
|---------|-------|
| **Name** | `url-spam-detector` |
| **Runtime** | Python |
| **Branch** | `main` |
| **Build Command** | `pip install -r requirements.txt && python -m nltk.downloader stopwords wordnet` |
| **Start Command** | `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true` |

5. Select the **Free** plan
6. Click **"Create Web Service"**

### Step 3: Wait for deployment

Render will install dependencies, download NLTK data, and start the Streamlit app. After a few minutes, you'll get a public URL like:

```
https://url-spam-detector.onrender.com
```

---

## Important Notes

- **Plain text commands:** When pasting the Build/Start commands into Render's dashboard, use `Cmd+Shift+V` (Mac) or `Ctrl+Shift+V` (Windows) to paste as plain text. Do NOT paste formatted text from VS Code.
- **Free tier:** The free Render instance spins down after inactivity. The first request after inactivity may take 30-60 seconds.
- **NLTK data:** The build command downloads the required NLTK packages (`stopwords` and `wordnet`). This runs once during the build phase.

---

## Technologies

- Python 3.12
- Streamlit
- scikit-learn (SVM, TF-IDF, GridSearchCV)
- NLTK (stopwords, WordNet lemmatizer)
- pandas, numpy, matplotlib, seaborn
