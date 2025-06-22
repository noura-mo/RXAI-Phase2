import re
import numpy as np
from fuzzywuzzy import fuzz

def parse_strength(strength):
    if not isinstance(strength, str):
        strength = str(strength)
    s = strength.strip().lower()

    m = re.match(r'(\d+\.?\d*)\s*%', s)
    if m:
        return float(m.group(1)), "ambiguous"
    m = re.match(r'(\d+\.?\d*)\s*(w/v|w/w|v/v)', s)
    if m:
        return float(m.group(1)), "ambiguous"
    m = re.match(r'(\d+\.?\d*)\s*mg\s*/\s*(\d+\.?\d*)\s*ml', s)
    if m:
        mg, ml = float(m.group(1)), float(m.group(2))
        return mg / ml, "standard"
    m = re.match(r'(\d+\.?\d*)\s*mg\s*/\s*ml', s)
    if m:
        return float(m.group(1)), "standard"
    m = re.match(r'(\d+\.?\d*)\s*(mg|g|mcg|μg|iu|au|tab|caps|drop|puff|unit|live)?', s)
    if m:
        val, unit = float(m.group(1)), m.group(2)
        if unit == 'g': return val * 1000, "standard"
        if unit in ['mcg', 'μg']: return val / 1000, "standard"
        if unit == 'mg': return val, "standard"
        return val, "ambiguous"
    return np.nan, "invalid"

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)

    synonym_map = {
        'paracetamol': 'acetaminophen',
        'fever': 'pyrexia',
        'high blood pressure': 'hypertension',
        'bp': 'blood pressure',
        'painkiller': 'analgesic',
        'diarrhoea': 'diarrhea',
        'anti-inflammatory': 'antiinflammatory',
        'sugar': 'glucose',
    }

    for k, v in synonym_map.items():
        text = re.sub(rf"\b{k}\b", v, text)

    tokens = text.split()
    return " ".join(sorted(set(tokens)))

def calculate_priority(active_match, strength_diff, form_match, med_price, strength_input):
    if active_match == 1 and form_match == 1 and strength_diff == 0:
        return 100.0
    strength_percentage = (1 - strength_diff / strength_input) * 100 if strength_diff != float('inf') else 0
    price_factor = (1 / (1 + np.log1p(med_price))) * 100 if med_price > 0 and not np.isnan(med_price) else 0
    priority_score = (0.5 * (active_match * 100)) + (0.2 * strength_percentage) + (0.2 * form_match * 100) + (0.1 * price_factor)
    return round(priority_score, 2)

def find_medicines(df, active_ingredient, strength, form):
    priority_list = []
    for _, row in df.iterrows():
        med_strength = row['strength_value']
        med_price = row['price'] if not np.isnan(row['price']) else float('inf')
        usage = row['Uses'] if 'Uses' in df.columns and isinstance(row['Uses'], str) else "N/A"

        strength_diff = abs(med_strength - strength)
        active_match = 1 if isinstance(row['composition'], str) and active_ingredient.lower() in row['composition'].lower() else 0

        form_match = 0
        if isinstance(row['form'], str):
            form_values = [f.strip().lower() for f in row['form'].split(',')]
            if form.lower() in form_values:
                form_match = 1

        priority = calculate_priority(active_match, strength_diff, form_match, med_price, strength)

        if active_match:
            priority_list.append({
                "Trade Name": row['Medicine Name'],
                "Active Ingredient": row['composition'],
                "Strength": f"{med_strength}mg",
                "Pharmaceutical Form": row['form'],
                "Priority": priority,
                "Price (EGP)": round(med_price, 2),
                "Indication": usage
            })

    return sorted(priority_list, key=lambda x: (-x["Priority"], x["Price (EGP)"]))
