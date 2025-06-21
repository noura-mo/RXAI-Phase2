from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from utils import clean_text, parse_strength, find_medicines

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# تحميل البيانات
try:
    df = pd.read_csv("Final dataset.csv")

    # تنظيف الأعمدة
    df['strength_value'], _ = zip(*df['strength'].map(parse_strength))
    df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
    df['clean_composition'] = df['composition'].map(clean_text)
    df['clean_uses'] = df['Uses'].map(clean_text)
    df['clean_name'] = df['Medicine Name'].map(clean_text)

    # دمج النصوص
    df['combined_text'] = (
        df['clean_name'] + " " + df['clean_composition'] + " " + df['clean_uses']
    ).map(lambda x: x.strip())

except Exception as e:
    print("Failed to load or clean data:", e)
    df = pd.DataFrame()

class MedicineRequest(BaseModel):
    active_ingredient: str
    strength: float
    form: str

@app.post("/get_best_medicine")
async def get_best_medicine(request: MedicineRequest):
    try:
        if df.empty:
            raise HTTPException(status_code=500, detail="Dataset not loaded.")
        results = find_medicines(df, request.active_ingredient, request.strength, request.form)
        if not results:
            raise HTTPException(status_code=404, detail="No matching medicines found")
        return {"Available_medicines": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
