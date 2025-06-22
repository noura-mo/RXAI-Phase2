from fastapi import FastAPI, Request
from pydantic import BaseModel
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import re

warnings.filterwarnings("ignore")

# إعدادات
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# تحميل البيانات
df = pd.read_csv("Final dataset.csv")

# تحميل الموديل
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# نموذج الطلب
class InputData(BaseModel):
    name: str

# إنشاء FastAPI app
app = FastAPI()

@app.get("/")
def root():
    return {"message": "RXAI API is up and running"}

@app.post("/match")
def match(data: InputData):
    input_text = data.name
    input_emb = model.encode([input_text])
    corpus_embeddings = model.encode(df['Trade Name'].astype(str).tolist())

    similarities = cosine_similarity(input_emb, corpus_embeddings)[0]
    df["score"] = similarities
    top5 = df.sort_values(by="score", ascending=False).head(5)[["Trade Name", "score"]]
    return top5.to_dict(orient="records")

