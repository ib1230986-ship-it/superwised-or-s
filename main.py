"""FastAPI backend -- model ko HTTP API ke zariye available karta hai
aur static/index.html page bhi serve karta hai.

Chalane ka tareeqa:   uvicorn main:app --reload
Phir browser mein:     http://localhost:8000
API docs (auto):       http://localhost:8000/docs
"""
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from predictor import LOOKUP, predict_units

BASE = Path(__file__).parent
app = FastAPI(title="Demand Predictor API")

# CORS: agar HTML page kisi aur jagah se khol rahe hain to API call chal sake
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _get_product(upc: int):
    rows = LOOKUP[LOOKUP.upc == upc]
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"UPC {upc} model ke data mein nahi hai")
    return rows.iloc[0]


@app.get("/")
def home():
    return FileResponse(BASE / "static" / "index.html")


@app.get("/api/categories")
def categories():
    return sorted(LOOKUP.category.unique().tolist())


@app.get("/api/products")
def products(category: str):
    df = LOOKUP[LOOKUP.category == category].sort_values("upc")
    if df.empty:
        raise HTTPException(status_code=404, detail=f"Category '{category}' nahi mili")
    return [{"upc": int(r.upc), "brand": r.brand, "typical_price": round(float(r.typical_price), 2)}
            for r in df.itertuples()]


@app.get("/api/predict")
def predict(upc: int, price: float = Query(..., gt=0), promo: bool = False,
            holiday: bool = False, comp: float = Query(1.0, gt=0)):
    p = _get_product(upc)
    typical = float(p.typical_price)
    units = predict_units(upc, price, promo, holiday, comp)
    base = predict_units(upc, typical, False, False, 1.0)
    ratio = price / typical
    return {
        "upc": upc, "category": p.category, "brand": p.brand,
        "typical_price": round(typical, 2),
        "predicted_units": round(units, 2),
        "expected_revenue": round(units * price, 2),
        "change_vs_typical_pct": round((units - base) / base * 100, 1),
        "warning": None if 0.7 <= ratio <= 1.15 else
                   "Price typical price ke 70%-115% se bahar hai, prediction bharosemand nahi.",
    }


@app.get("/api/curve")
def curve(upc: int, promo: bool = False, holiday: bool = False, comp: float = Query(1.0, gt=0)):
    p = _get_product(upc)
    typical = float(p.typical_price)
    prices = np.linspace(typical * 0.7, typical * 1.15, 30)
    units = [predict_units(upc, float(x), promo, holiday, comp) for x in prices]
    revenue = [u * float(x) for u, x in zip(units, prices)]
    best = float(prices[int(np.argmax(revenue))])
    return {"prices": [round(float(x), 3) for x in prices],
            "units": [round(u, 3) for u in units],
            "revenue": [round(r, 3) for r in revenue],
            "best_revenue_price": round(best, 2)}
