import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sqlalchemy import create_engine
from dotenv import load_dotenv
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ============================================
# DATABASE CONNECTION
# ============================================
load_dotenv()

DATABASE_URL = (
    f"postgresql://{os.getenv('DB_USER', 'postgres')}:"
    f"{os.getenv('DB_PASSWORD', '')}@"
    f"{os.getenv('DB_HOST', 'localhost')}:"
    f"{os.getenv('DB_PORT', '5432')}/"
    f"{os.getenv('DB_NAME', 'ai_analytics')}"
)

engine = create_engine(DATABASE_URL)

# ============================================
# LOAD DATA
# ============================================
query = """
    SELECT
        f.price,
        f.freight_value,
        f.payment_value,
        f.payment_installments,
        f.payment_type,
        f.days_to_deliver,
        f.days_estimated,
        f.delivery_delay_days,
        f.review_score,
        f.order_status,
        p.category_name_english     AS product_category,
        c.customer_state,
        s.seller_state
    FROM fact_orders f
    JOIN dim_products p  ON f.product_key  = p.product_key
    JOIN dim_customers c ON f.customer_key = c.customer_key
    JOIN dim_sellers s   ON f.seller_key   = s.seller_key
    WHERE f.review_score  IS NOT NULL
      AND f.days_to_deliver IS NOT NULL
      AND f.order_status = 'delivered'
"""

df = pd.read_sql(query, engine)

# ============================================
# FEATURE ENGINEERING
# ============================================

# Delivered faster than estimated = positive experience
df["delivered_early"]     = (df["delivery_delay_days"] < 0).astype(int)

# Price relative to freight — high freight on cheap item = bad experience
df["freight_ratio"]       = df["freight_value"] / (df["price"] + 1)

# Total cost paid vs item price
df["price_to_payment"]    = df["price"] / (df["payment_value"] + 1)

# Binary target — good review (4-5) vs bad review (1-3)
df["review_good"]         = (df["review_score"] >= 4).astype(int)

# ============================================
# ENCODE CATEGORICALS
# ============================================
cat_cols = ["payment_type", "product_category", "customer_state", "seller_state"]

le = LabelEncoder()
for col in cat_cols:
    df[col] = df[col].fillna("unknown")
    df[col] = le.fit_transform(df[col].astype(str))

# ============================================
# DEFINE FEATURES
# ============================================
FEATURES = [
    "price",
    "freight_value",
    "payment_value",
    "payment_installments",
    "payment_type",
    "days_to_deliver",
    "days_estimated",
    "delivery_delay_days",
    "delivered_early",
    "freight_ratio",
    "price_to_payment",
    "product_category",
    "customer_state",
    "seller_state"
]

TARGET = "review_good"

df = df.dropna(subset=FEATURES + [TARGET])

X = df[FEATURES]
y = df[TARGET]

# ============================================
# TRAIN / TEST SPLIT — 80/20 stratified
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("-" * 50)
print("DATASET SUMMARY")
print("-" * 50)
print(f"  Total samples     : {len(df):,}")
print(f"  Training samples  : {len(X_train):,}")
print(f"  Test samples      : {len(X_test):,}")
print(f"  Good reviews (1)  : {y.sum():,} ({y.mean()*100:.1f}%)")
print(f"  Bad reviews  (0)  : {(1-y).sum():,} ({(1-y.mean())*100:.1f}%)")
print("-" * 50)

# ============================================
# TRAIN XGBOOST MODEL
# ============================================
scale_pos_weight = (len(y) - y.sum()) / y.sum()
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# ============================================
# EVALUATE ON TEST SET
# ============================================

# predicted probabilities
y_pred_prob = model.predict_proba(X_test)[:, 1]

threshold = 0.25
y_pred = (y_pred_prob >= threshold).astype(int)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
print("\n" + "-" * 50)
print("XGBOOST MODEL - EVALUATION RESULTS")
print("-" * 50)
print(f"  Accuracy   : {accuracy*100:.2f}%")
print(f"  Precision  : {precision*100:.2f}%")
print(f"  Recall     : {recall*100:.2f}%")
print(f"  F1 Score   : {f1*100:.2f}%")
print("-" * 50)

print("\nCLASSIFICATION REPORT")
print("-" * 50)
print(classification_report(y_test, y_pred, target_names=["Bad Review", "Good Review"]))

# ============================================
# CROSS VALIDATION
# ============================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=-1)

print("-" * 50)
print("CROSS VALIDATION - 5 FOLD")
print("-" * 50)
print(f"  F1 per fold  : {[f'{s*100:.2f}%' for s in cv_scores]}")
print(f"  Mean F1      : {cv_scores.mean()*100:.2f}%")
print(f"  Std Dev      : {cv_scores.std()*100:.2f}%")
print("-" * 50)

# ============================================
# FEATURE IMPORTANCE
# ============================================
importance_df = pd.DataFrame({
    "feature":    FEATURES,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\nFEATURE IMPORTANCE - TOP 10")
print("-" * 50)
for _, row in importance_df.head(10).iterrows():
    bar = "#" * int(row["importance"] * 200)
    print(f"  {row['feature']:<25} {row['importance']:.4f}  {bar}")
print("-" * 50)

# ============================================
# PLOTS
# ============================================
output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
os.makedirs(output_dir, exist_ok=True)

# Plot 1 - Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(
    importance_df["feature"].head(10)[::-1],
    importance_df["importance"].head(10)[::-1],
    color="#1f77b4"
)
plt.title("XGBoost - Feature Importance (Top 10)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "xgboost_feature_importance.png"), dpi=150)
plt.show()

# Plot 2 - Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
plt.colorbar(im)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Bad Review", "Good Review"])
ax.set_yticklabels(["Bad Review", "Good Review"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=14)
plt.title("XGBoost - Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "xgboost_confusion_matrix.png"), dpi=150)
plt.show()

# Plot 3 - Cross Validation Scores
plt.figure(figsize=(8, 4))
plt.bar(
    [f"Fold {i+1}" for i in range(len(cv_scores))],
    cv_scores * 100,
    color="#1f77b4",
    edgecolor="white"
)
plt.axhline(y=cv_scores.mean() * 100, color="orange", linestyle="--", linewidth=2, label=f"Mean: {cv_scores.mean()*100:.2f}%")
plt.title("XGBoost - Cross Validation F1 Score (5 Fold)")
plt.ylabel("F1 Score (%)")
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "xgboost_cross_validation.png"), dpi=150)
plt.show()

# ============================================
# SAVE MODEL
# ============================================
models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(models_dir, exist_ok=True)

with open(os.path.join(models_dir, "xgboost_review_model.pkl"), "wb") as f:
    pickle.dump(model, f)

print("\nModel saved: ml/models/xgboost_review_model.pkl")