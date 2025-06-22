from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import torch
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings("ignore")

# إعداد الجهاز
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# تحميل البيانات
df = pd.read_csv("Final dataset.csv")

# تحميل النموذج
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# تعريف FastAPI
app = FastAPI()

# تعريف شكل البيانات المطلوبة من المستخدم
class InputData(BaseModel):
    name: str

# الراوت الأساسي
@app.get("/")
def root():
    return {"message": "API working"}

# الراوت الخاص بالماتشنج
@app.post("/match")
def match(data: InputData):
    input_text = data.name
    input_emb = model.encode([input_text])
    corpus_embeddings = model.encode(df['Trade Name'].astype(str).tolist())

    similarities = cosine_similarity(input_emb, corpus_embeddings)[0]
    df["score"] = similarities
    top5 = df.sort_values(by="score", ascending=False).head(5)[["Trade Name", "score"]]
    return top5.to_dict(orient="records")

# تشغيل السيرفر في حالة local فقط
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
