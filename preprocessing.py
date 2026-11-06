"""Conservative reusable text preprocessing helpers."""
import re, html
def clean_text(text):
    text=html.unescape(str(text))
    text=re.sub(r"<[^>]+>"," ",text)
    text=re.sub(r"https?://\S+|www\.\S+"," ",text)
    return re.sub(r"\s+"," ",text.lower()).strip()
def simple_tokens(text):
    return re.findall(r"\b[\w']+\b",clean_text(text))
def remove_stopwords(tokens):
    stop={"the","a","an","is","am","are","was","were","and","or","of","to","in","for","this","that","it","i","my","with","on","at","as","by"}
    protected={"not","no","never","without"}
    return [t for t in tokens if t not in stop or t in protected]
