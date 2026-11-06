"""Train a small Word2Vec Skip-gram demonstration model."""
from pathlib import Path
import pandas as pd
from gensim.models import Word2Vec
from preprocessing import simple_tokens
ROOT=Path(__file__).resolve().parent
df=pd.read_csv(ROOT/"data/customer_reviews.csv")
sentences=[simple_tokens(t) for t in df.review_text.dropna()]
model=Word2Vec(sentences,vector_size=100,window=5,min_count=1,workers=1,sg=1,seed=42,epochs=30)
(ROOT/"models").mkdir(exist_ok=True); model.save(str(ROOT/"models/word2vec.model"))
print("Saved Word2Vec model; vocabulary size:",len(model.wv))
