# Price Elasticity - Demand Predictor

Ek chhota end-to-end project: retail data par XGBoost model train kiya gaya hai jo batata hai
ke **price, promo aur holiday badalne par kisi product ki demand (units sold) kitni badalti hai**.
Model ko FastAPI ke zariye serve kiya gaya hai aur ek simple HTML page us se baat karta hai.

> **Note:** Ye model **simulated (banaye hue) data** par train hua hai, asli Dominick's Finer Foods
> data par nahi. Natayej asli market ke liye nahi hain, sirf seekhne/demo ke liye hain.

## Features
- 14 product categories, 1,390 products
- Predicted units / week, expected revenue, aur typical price ke muqable mein farq
- Price vs Demand curve (model ke hisab se sab se zyada revenue wali price ke sath)
- REST API (`/docs` par auto documentation)

## Project structure
```
main.py            FastAPI server (API + HTML page)
predictor.py       Prediction logic
static/index.html  Frontend (plain HTML/CSS/JS, koi external library nahi)
model.joblib       Trained XGBoost model
features.json      Model ke input columns
upc_lookup.csv     Product info (category, brand, typical price, product baseline demand)
train_and_save.py  Model dobara train karne ke liye
```

## Run locally
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
python -m uvicorn main:app --port 8000
```
Phir browser mein kholein: http://localhost:8000

## API
| Endpoint | Description |
|---|---|
| `GET /api/categories` | Saari categories |
| `GET /api/products?category=Cereals` | Category ke products |
| `GET /api/predict?upc=1056423&price=2.34&promo=false&holiday=false&comp=1.0` | Predicted units aur revenue |
| `GET /api/curve?upc=1056423` | Price vs demand curve ka data |

## Model details
- **Model:** XGBoost Regressor (300 trees, max_depth 6, learning rate 0.05)
- **Target:** `ln(units sold)`
- **Inputs (19):** `ln_unit_price`, `ln_rel_price` (product ki typical price ke muqable mein price),
  `promo_flag`, `holiday_week`, `comp_price_index`, `upc_effect` (product baseline demand),
  aur category dummies
- **Test R²:** ~0.65

## Limitations
- Simulated data par train hua hai.
- Price sirf product ki typical price ke 70%-115% tak bharosemand hai; is se bahar page warning dikhata hai.
- Jo product `upc_lookup.csv` mein nahi, uski prediction nahi chalegi.
- Price curve thori step-jaisi ho sakti hai, kyun ke model trees par based hai.

## Retrain
```bash
python train_and_save.py path/to/dominicks_style_merged.csv
```
(Training CSV bari hai, is liye repo mein shamil nahi.)
