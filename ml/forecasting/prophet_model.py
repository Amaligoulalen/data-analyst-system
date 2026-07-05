import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings("ignore")

from sqlalchemy import create_engine
from dotenv import load_dotenv
from xgboost import XGBClassifier
from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    cross_val_score, RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score
)
load_dotenv()

DATABASE_URL = (
    f"postgresql://{os.getenv('DB_USER', 'postgres')}:"
    f"{os.getenv('DB_PASSWORD', '')}@"
    f"{os.getenv('DB_HOST', 'localhost')}:"
    f"{os.getenv('DB_PORT', '5432')}/"
    f"{os.getenv('DB_NAME', 'ai_analytics')}"
)

engine = create_engine(DATABASE_URL)

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
    WHERE f.review_score    IS NOT NULL
      AND f.days_to_deliver IS NOT NULL
      AND f.order_status    = 'delivered'
"""

df = pd.read_sql(query, engine)

df["delivered_early"]   = (df["delivery_delay_days"] < 0).astype(int)
df["delivery_gap"]      = df["days_to_deliver"] - df["days_estimated"]
df["freight_ratio"]     = df["freight_value"] / (df["price"] + 1)
df["price_to_payment"]  = df["price"] / (df["payment_value"] + 1)
df["expensive_item"]    = (df["price"] > df["price"].median()).astype(int)
df["high_installments"] = (df["payment_installments"] > 3).astype(int)


df["review_good"] = (df["review_score"] >= 4).astype(int)


cat_cols = ["payment_type", "product_category", "customer_state", "seller_state"]

df[cat_cols] = df[cat_cols].fillna("unknown")
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

exclude  = ["review_score", "review_good", "order_status"]
FEATURES = [col for col in df.columns if col not in exclude]
TARGET   = "review_good"

df = df.dropna(subset=[TARGET])
X  = df[FEATURES]
y  = df[TARGET]

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


scale_pos_weight = (len(y) - y.sum()) / y.sum()
print(f"\n  scale_pos_weight  : {scale_pos_weight:.4f}")

print("\nRunning hyperparameter search (this may take a few minutes)...")

param_grid = {
    "max_depth":        [4, 5, 6, 7],
    "learning_rate":    [0.01, 0.03, 0.05],
    "n_estimators":     [200, 300, 400],
    "subsample":        [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9]
}

search = RandomizedSearchCV(
    XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    ),
    param_grid,
    n_iter=20,
    scoring="f1",
    cv=5,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

search.fit(X_train, y_train)
print(f"\n  Best params : {search.best_params_}")

best_params = search.best_params_

model = XGBClassifier(
    **best_params,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    early_stopping_rounds=30,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)


y_pred_prob = model.predict_proba(X_test)[:, 1]

threshold = 0.40
y_pred = (y_pred_prob >= threshold).astype(int)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_pred_prob)

print("\n" + "-" * 50)
print("XGBOOST MODEL - EVALUATION RESULTS")
print("-" * 50)
print(f"  Accuracy   : {accuracy*100:.2f}%")
print(f"  Precision  : {precision*100:.2f}%")
print(f"  Recall     : {recall*100:.2f}%")
print(f"  F1 Score   : {f1*100:.2f}%")
print(f"  ROC-AUC    : {roc_auc:.4f}")
print("-" * 50)

print("\nCLASSIFICATION REPORT")
print("-" * 50)
print(classification_report(y_test, y_pred, target_names=["Bad Review", "Good Review"]))

cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_model = XGBClassifier(
    **best_params,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

cv_scores = cross_val_score(cv_model, X, y, cv=cv, scoring="f1", n_jobs=-1)

print("-" * 50)
print("CROSS VALIDATION - 5 FOLD")
print("-" * 50)
print(f"  F1 per fold : {[f'{s*100:.2f}%' for s in cv_scores]}")
print(f"  Mean F1     : {cv_scores.mean()*100:.2f}%")
print(f"  Std Dev     : {cv_scores.std()*100:.2f}%")
print("-" * 50)

importance_df = pd.DataFrame({
    "feature":    X.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False).head(15)

print("\nFEATURE IMPORTANCE - TOP 15")
print("-" * 50)
for _, row in importance_df.iterrows():
    bar = "#" * int(row["importance"] * 300)
    print(f"  {row['feature']:<35} {row['importance']:.4f}  {bar}")
print("-" * 50)


output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
os.makedirs(output_dir, exist_ok=True)


plt.figure(figsize=(10, 7))
plt.barh(
    importance_df["feature"][::-1],
    importance_df["importance"][::-1],
    color="#1f77b4"
)
plt.title("XGBoost - Feature Importance (Top 15)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "xgboost_feature_importance.png"), dpi=150)
plt.show()

# Plot 2 - Confusion Matrix
cm  = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
im  = ax.imshow(cm, interpolation="nearest", cmap="Blues")
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


plt.figure(figsize=(8, 4))
plt.bar(
    [f"Fold {i+1}" for i in range(len(cv_scores))],
    cv_scores * 100,
    color="#1f77b4",
    edgecolor="white"
)
plt.axhline(
    y=cv_scores.mean() * 100,
    color="orange", linestyle="--", linewidth=2,
    label=f"Mean F1: {cv_scores.mean()*100:.2f}%"
)
plt.title("XGBoost - Cross Validation F1 Score (5 Fold)")
plt.ylabel("F1 Score (%)")
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "xgboost_cross_validation.png"), dpi=150)
plt.show()


models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(models_dir, exist_ok=True)

with open(os.path.join(models_dir, "xgboost_review_model.pkl"), "wb") as f:
    pickle.dump(model, f)

print("\nModel saved: ml/models/xgboost_review_model.pkl")