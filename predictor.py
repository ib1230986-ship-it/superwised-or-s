"""Prediction logic (model.joblib + upc_lookup.csv use karta hai)."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
model = joblib.load(BASE / "model.joblib")
FEATURES = json.load(open(BASE / "features.json"))
LOOKUP = pd.read_csv(BASE / "upc_lookup.csv")


def predict_units(upc, unit_price, promo, holiday, comp_price_index=1.0):
    """Ek product ke liye predicted units sold (plain units mein)."""
    row = LOOKUP[LOOKUP.upc == upc].iloc[0]
    x = pd.DataFrame(0.0, index=[0], columns=FEATURES)
    x["ln_unit_price"] = np.log(unit_price)
    x["ln_rel_price"] = np.log(unit_price / row.typical_price)
    x["promo_flag"] = int(promo)
    x["holiday_week"] = int(holiday)
    x["comp_price_index"] = comp_price_index
    x["upc_effect"] = row.upc_effect
    cat_col = f"cat_{row.category}"
    if cat_col in x.columns:
        x[cat_col] = 1.0
    return float(np.exp(model.predict(x)[0]))
