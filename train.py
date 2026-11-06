"""Train aspect and sentiment text classifiers; save metrics and models."""
from pathlib import Path
import json, joblib, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
ROOT=Path(__file__).resolve().parent
df=pd.read_csv(ROOT/"data/customer_reviews.csv").dropna(subset=["review_text","aspect","sentiment"])
df["review_text"]=df["review_text"].astype(str).str.strip()
df=df[df.review_text!=""].drop_duplicates(subset=["review_text"])
(ROOT/"models").mkdir(exist_ok=True); (ROOT/"reports").mkdir(exist_ok=True)
def train_target(target,filename):
    X=df.review_text; y=df[target]
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
    pipe=Pipeline([("tfidf",TfidfVectorizer(lowercase=True,ngram_range=(1,2),min_df=1,sublinear_tf=True)),
                   ("classifier",LogisticRegression(max_iter=2000,class_weight="balanced",random_state=42))])
    pipe.fit(Xtr,ytr); pred=pipe.predict(Xte)
    report=classification_report(yte,pred,output_dict=True,zero_division=0)
    result={"accuracy":float(accuracy_score(yte,pred)),"macro_f1":float(f1_score(yte,pred,average="macro")),"report":report}
    joblib.dump(pipe,ROOT/"models"/filename)
    labels=sorted(y.unique()); cm=confusion_matrix(yte,pred,labels=labels)
    disp=ConfusionMatrixDisplay(cm,display_labels=labels)
    fig,ax=plt.subplots(figsize=(8,6)); disp.plot(ax=ax,xticks_rotation=35,colorbar=False)
    ax.set_title(f"Confusion Matrix — {target}"); plt.tight_layout()
    fig.savefig(ROOT/"reports"/f"{target}_confusion_matrix.png",dpi=150); plt.close(fig)
    print(f"\\n{target}: accuracy={result['accuracy']:.3f}, macro-F1={result['macro_f1']:.3f}")
    print(classification_report(yte,pred,zero_division=0))
    return result
results={"aspect":train_target("aspect","aspect_classifier.joblib"),
         "sentiment":train_target("sentiment","sentiment_classifier.joblib")}
with open(ROOT/"reports/training_metrics.json","w") as f: json.dump(results,f,indent=2)
print("Saved models and reports.")
