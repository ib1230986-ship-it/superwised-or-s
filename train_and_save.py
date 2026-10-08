"""
Model train karke save karta hai (sirf EK baar chalana hai).
Output: model.joblib, features.json, upc_lookup.csv
Usage:  python train_and_save.py path/to/dominicks_style_merged.csv
"""
import sys, json, joblib
import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

PATH = sys.argv[1] if len(sys.argv) > 1 else "dominicks_style_merged.csv"
COLS = ["ln_unit_price", "ln_move", "promo_flag", "holiday_week", "comp_price_index",
        "category", "upc", "brand", "unit_price", "regular_price", "ok"]
df = pd.read_csv(PATH, usecols=COLS)
df = df[df.ok == 1].copy()
if len(df) > 300_000:
    df = df.sample(300_000, random_state=42).reset_index(drop=True)

# product-level fixed effect (upc_effect) lookup -- app mein bhi yahi use hoga
lookup = (df.groupby("upc")
            .agg(category=("category", "first"), brand=("brand", "first"),
                 upc_effect=("ln_move", "mean"), typical_price=("regular_price", "median"))
            .reset_index())
df = df.merge(lookup[["upc", "upc_effect", "typical_price"]], on="upc", how="left")
# product ki apni normal price ke muqable mein price (isi se model ko price ka asal asar samajh aata hai)
df["ln_rel_price"] = np.log(df["unit_price"] / df["typical_price"])

cat_dummies = pd.get_dummies(df["category"], prefix="cat", drop_first=True)
X = pd.concat([df[["ln_unit_price", "ln_rel_price", "promo_flag", "holiday_week",
                   "comp_price_index", "upc_effect"]], cat_dummies], axis=1)
y = df["ln_move"]

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=2)
model.fit(Xtr, ytr)
print("Test R2:", round(r2_score(yte, model.predict(Xte)), 4))

joblib.dump(model, "model.joblib")
json.dump(list(X.columns), open("features.json", "w"))
lookup.round(4).to_csv("upc_lookup.csv", index=False)
print("Saved: model.joblib, features.json, upc_lookup.csv")
