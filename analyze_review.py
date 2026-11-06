"""Run aspect, sentiment, preprocessing and optional NER/embedding analysis."""
from pathlib import Path
import argparse, json, joblib
from preprocessing import clean_text,simple_tokens,remove_stopwords
ROOT=Path(__file__).resolve().parent
def analyze(text,optional=False):
    ap=ROOT/"models/aspect_classifier.joblib"; sp=ROOT/"models/sentiment_classifier.joblib"
    if not ap.exists() or not sp.exists(): raise FileNotFoundError("Run python train.py first.")
    cleaned=clean_text(text); tokens=simple_tokens(cleaned)
    result={"original_review":text,"cleaned_text":cleaned,"tokens":tokens,
      "tokens_after_stopword_removal":remove_stopwords(tokens),
      "predicted_aspect":str(joblib.load(ap).predict([text])[0]),
      "predicted_sentiment":str(joblib.load(sp).predict([text])[0])}
    try:
        import spacy
        nlp=spacy.load("en_core_web_sm")
        result["named_entities"]=[{"text":e.text,"label":e.label_} for e in nlp(text).ents]
    except Exception:
        result["named_entities"]="Optional: pip install spacy && python -m spacy download en_core_web_sm"
    if optional:
        try:
            from gensim.models import Word2Vec
            m=Word2Vec.load(str(ROOT/"models/word2vec.model"))
            words=[w for w in tokens if w in m.wv]
            result["word2vec_similar_words"]={w:m.wv.most_similar(w,topn=3) for w in words[:3]}
        except Exception: result["word2vec_similar_words"]="Run python train_word2vec.py first."
        try:
            from sentence_transformers import SentenceTransformer
            v=SentenceTransformer("all-MiniLM-L6-v2").encode([text])[0]
            result["sentence_embedding"]={"dimensions":len(v),"first_8_values":v[:8].round(4).tolist()}
        except Exception: result["sentence_embedding"]="Optional: install sentence-transformers; first use downloads model."
    return result
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("review",nargs="?"); p.add_argument("--optional",action="store_true")
    a=p.parse_args(); text=a.review or input("Enter a customer review: ")
    print(json.dumps(analyze(text,a.optional),indent=2,ensure_ascii=False))
