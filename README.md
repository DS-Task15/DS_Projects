# Customer Review Intelligence System

An integrated educational NLP project demonstrating preprocessing, tokenization, stopword removal, TF-IDF, text classification, sentiment analysis, NER, Word2Vec and sentence embeddings.

## Folder contents
- `data/customer_reviews.csv`: 300 synthetic labelled reviews.
- `preprocessing.py`: reusable cleaning, tokenization and stopword helpers.
- `train.py`: trains aspect and sentiment classifiers using TF-IDF + Logistic Regression.
- `analyze_review.py`: analyzes a new review and returns JSON.
- `train_word2vec.py`: trains a Word2Vec Skip-gram demo model.
- `models/`: saved models appear after training.
- `reports/`: metrics and confusion matrices appear after training.
- `requirements.txt`: dependencies.

## Windows setup
Open Command Prompt in this folder:
```bat
py -m venv .venv
.venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python train.py
python analyze_review.py "The camera is excellent but the battery drains quickly"
```
Optional Word2Vec:
```bat
python train_word2vec.py
```
Optional NER:
```bat
python -m spacy download en_core_web_sm
```
Optional Word2Vec + sentence embeddings:
```bat
python analyze_review.py "Rahul bought a Samsung phone from Amazon. The camera is excellent." --optional
```
Sentence Transformers downloads `all-MiniLM-L6-v2` on first use, so internet is required then.

## Detailed explanation

### 1. Dataset and labels
`review_text` is input X. `aspect` and `sentiment` are two different target labels. This is supervised learning because the examples have labels.

### 2. Preprocessing and tokenization
`clean_text()` decodes HTML entities, removes HTML tags and URLs, lowercases and normalizes whitespace. `simple_tokens()` extracts word-like tokens. `remove_stopwords()` removes a small illustrative stopword list but deliberately preserves negations such as `not`, `no`, `never`, and `without`. Real projects should tune preprocessing to their domain and language.

### 3. TF-IDF
`TfidfVectorizer` learns vocabulary and inverse-document-frequency weights from training data and converts each review to a sparse numeric vector. `ngram_range=(1,2)` includes single words and two-word phrases. The scikit-learn Pipeline ensures the vectorizer is fit only on training data.

### 4. Text classification and sentiment
Two separate Logistic Regression pipelines are trained:
- Aspect classifier predicts Battery, Camera, Shipping, Price or Build Quality.
- Sentiment classifier predicts Positive, Negative or Neutral.
They are separate because aspect and sentiment are distinct targets.

### 5. Evaluation
A stratified 75/25 train/test split is used. Accuracy, precision, recall, F1 and macro F1 are printed. Confusion matrices and metrics are saved in `reports/`. A small synthetic test set is only a teaching exercise and cannot establish production performance.

### 6. NER
If spaCy and `en_core_web_sm` are installed, `analyze_review.py` extracts entities such as people, organizations, locations, dates and money. General-purpose NER may miss product-specific entities.

### 7. Word2Vec
`train_word2vec.py` trains Skip-gram (`sg=1`). `vector_size=100` means each word vector has 100 dimensions; `window=5` controls context width. This tiny corpus only demonstrates implementation and does not produce high-quality general embeddings.

### 8. Sentence embeddings
With `--optional`, Sentence Transformers encodes the complete review using `all-MiniLM-L6-v2`. This supports semantic search, clustering and RAG. The first use downloads the pretrained model.

## Production guidance
Replace synthetic data with real, representative and properly labelled reviews. Collect enough examples for every class, keep a final untouched test set, use cross-validation during tuning, inspect errors, check class imbalance and drift, and do not treat predictions as guaranteed facts. Only load joblib model files you trust.
