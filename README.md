# 🕵️ Fake News Detection using LSTM & NLP

> An end-to-end deep learning system that classifies news articles as **Real** or **Fake** with **99.8% accuracy**, built to combat the growing threat of misinformation using Natural Language Processing and LSTM neural networks.

---

## 🎯 Business Problem

Misinformation spreads **6x faster** than factual news on social media, causing reputational and financial damage to media organizations and the public. Manual fact-checking is slow, expensive, and unscalable. This system automates the detection pipeline — classifying 44,000+ articles in seconds with near-perfect accuracy, reducing manual review effort by over **90%**.

---

## 📊 Results at a Glance

| Metric | Score |
|---|---|
| Validation Accuracy | **99.8%** |
| False Negative Rate | **0.02%** (9 misclassified out of 44K) |
| Model Type | LSTM (Long Short-Term Memory) |
| Dataset Size | ~44,000 news articles |
| Training Convergence | Both Train & Val Acc > 95% within 5 epochs |

---

## 🚀 Key Features

- **Interactive Streamlit UI** — paste any news article and get an instant Real/Fake classification
- **Confidence Score** — Sigmoid probability score tells you *how certain* the model is (e.g., 100.00%)
- **Visual Probability Analysis** — color-coded pie chart showing Real vs Fake probability breakdown
- **Model Performance Dashboard** — live accuracy/loss graphs from training, accessible via sidebar toggle
- **Confusion Matrix Analysis** — detailed breakdown of True Positives, False Positives, and misclassification rates
- **Real-Time Inference** — single-click analysis triggers the full NLP pipeline: cleaning → tokenization → padding → LSTM inference

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.x |
| Deep Learning | TensorFlow / Keras |
| NLP | NLTK (tokenization, stopword removal) |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Streamlit |
| Model Architecture | Embedding → LSTM → Dense (Sigmoid) |

---

## 🧠 How It Works

```
Raw News Text
     │
     ▼
Text Preprocessing
(Lowercase → Remove punctuation → Remove stopwords)
     │
     ▼
Tokenization & Sequence Padding
(Keras Tokenizer → Fixed-length sequences)
     │
     ▼
LSTM Neural Network
(Embedding Layer → LSTM Layer → Dense + Sigmoid)
     │
     ▼
Classification Output
REAL ✅ / FAKE 🚨 + Confidence Score
```

---

## 🔬 NLP Pipeline Details

### Text Preprocessing
- Lowercasing and punctuation removal
- Stopword removal using NLTK corpus
- Tokenization and vocabulary building (~vocabulary size based on training corpus)
- Sequence padding to fixed length for uniform LSTM input

### Feature Engineering
- **Sequence Length Analysis** — distribution of article word counts to set optimal padding length
- **Word Frequency Distribution** — top-N most frequent tokens to build embedding vocabulary
- **Token Index Mapping** — each unique word mapped to a numeric index for embedding layer input

### Model Architecture
```
Embedding Layer    →  Word vector representations (dense, trainable)
LSTM Layer         →  Captures long-range contextual dependencies in text
Dropout Layer      →  Regularization to prevent overfitting
Dense Layer        →  Fully connected output
Sigmoid Activation →  Binary probability output (0 = Fake, 1 = Real)
```

---

## 📁 Project Structure

```
Fake-News-Detection/
│
├── app.py                  # Streamlit frontend application
├── model/
│   ├── lstm_model.h5       # Trained LSTM model weights
│   └── tokenizer.pkl       # Fitted Keras tokenizer
├── notebooks/
│   └── Fake_News_ML.ipynb  # Full training & evaluation notebook
├── data/
│   ├── True.csv            # Real news dataset
│   └── Fake.csv            # Fake news dataset
├── utils/
│   └── preprocess.py       # Text cleaning & preprocessing functions
└── README.md
```

---

## ⚙️ Setup & Installation

```bash
# Clone the repository
git clone https://github.com/Niya3-Navya/Fake-News-Detection_LSTM.git
cd Fake-News-Detection_LSTM

# Download NLTK data
python -c "import nltk; nltk.download('stopwords')"

# Run the Streamlit app
streamlit run app.py
```

> **Note:** Download `True.csv` and `Fake.csv` from Kaggle before running the notebook.  
> Dataset: [Fake and Real News Dataset — Kaggle](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)

---

## 📈 Model Performance

### Training Curves
Both **Train Accuracy** and **Validation Accuracy** converge above 95%, confirming the model generalizes well to unseen news articles — not just memorizing training data.

### Confusion Matrix Interpretation (Business Context)
- **True Positives (Real → Real):** Correctly verified authentic articles — no unnecessary flags
- **True Negatives (Fake → Fake):** Successfully caught misinformation before it spreads
- **False Negatives:** Only ~9 fake articles out of 44,000 slipped through — a **0.02% miss rate**
- **False Positives:** Minimal real articles flagged as fake — preserving credibility of legitimate journalism

---

## 💡 Business Impact & Applications

| Use Case | Impact |
|---|---|
| News platforms | Auto-flag suspicious articles before publishing |
| Social media moderation | Scale fact-checking without proportional headcount |
| Research & journalism | Rapid screening of large article volumes |
| Public awareness tools | Browser extensions for real-time news verification |

---

## 🔮 Future Enhancements

- [ ] Integrate **BERT / RoBERTa** transformer models for improved contextual understanding
- [ ] Add **multi-language support** for non-English news detection
- [ ] Build **REST API** using FastAPI for integration with third-party platforms
- [ ] Implement **explainability layer** (LIME/SHAP) to highlight words that triggered the classification
- [ ] Connect to **live news RSS feeds** for real-time monitoring dashboard

---



## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

*Built with ❤️ to fight misinformation using the power of Deep Learning.*
