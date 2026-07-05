import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings("ignore")

from sqlalchemy import create_engine
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

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




query = """
    SELECT
        c.customer_id,

        COUNT(DISTINCT f.order_id) AS total_orders,
        SUM(f.price)               AS total_spent,
        AVG(f.price)               AS avg_order_value,
        AVG(f.freight_value)       AS avg_freight,
        AVG(f.days_to_deliver)     AS avg_delivery_days,
        AVG(f.review_score)        AS avg_review_score,
        AVG(f.delivery_delay_days) AS avg_delay,
        SUM(f.payment_installments) AS total_installments,
        COUNT(DISTINCT p.category_name_english) AS unique_categories,

        -- RFM features
        MAX(f.order_purchase_timestamp) AS last_purchase,
        MIN(f.order_purchase_timestamp) AS first_purchase

    FROM fact_orders f
    JOIN dim_customers c ON f.customer_key = c.customer_key
    JOIN dim_products  p ON f.product_key  = p.product_key

    WHERE f.order_status = 'delivered'
      AND f.days_to_deliver IS NOT NULL
      AND f.review_score    IS NOT NULL

    GROUP BY c.customer_id

    -- REMOVE single-order customers
    HAVING COUNT(DISTINCT f.order_id) >= 1
"""

df = pd.read_sql(query, engine)
# Price per day
df["price_per_day"] = (
    df["total_spent"] /
    (df["avg_delivery_days"] + 1)
)

# Freight ratio
df["freight_ratio"] = (
    df["avg_freight"] /
    (df["avg_order_value"] + 1)
)
customer_ids = df["customer_id"]
# Convert dates
df["last_purchase"]  = pd.to_datetime(df["last_purchase"])
df["first_purchase"] = pd.to_datetime(df["first_purchase"])

# Reference date
today = df["last_purchase"].max()

# RFM Features
df["recency"] = (today - df["last_purchase"]).dt.days
df["customer_lifetime_days"] = (
    df["last_purchase"] - df["first_purchase"]
).dt.days

# Drop original timestamps
df = df.drop(columns=["last_purchase", "first_purchase"])

print("-" * 50)
print("DATASET SUMMARY")
print("-" * 50)
print(f"  Total customers   : {len(df):,}")

print("-" * 50)

# ============================================
# FEATURES FOR CLUSTERING
# ============================================
FEATURES = [
    "total_spent",
    "avg_order_value",
    "avg_freight",
    "avg_delivery_days",
    "avg_review_score",
    "avg_delay",
    "total_installments",
    "unique_categories",
    "price_per_day",
    "freight_ratio"
]

X = df[FEATURES].fillna(0)

# ============================================
# NORMALIZATION — mandatory for K-Means
# ============================================
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================
# ELBOW METHOD — find optimal K
# ============================================
print("\nRunning Elbow Method...")

inertia         = []
silhouette      = []
k_range         = range(2, 11)

for k in k_range:
    km  = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)
    sil = silhouette_score(X_scaled, km.labels_, sample_size=10000, random_state=42)
    silhouette.append(sil)
    print(f"  K={k}  Inertia={km.inertia_:,.0f}  Silhouette={sil:.4f}")

# Best K by silhouette
best_k = 4
print(f"\n  Best K (Silhouette) : {best_k}")

# ============================================
# TRAIN FINAL MODEL
# ============================================
model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
model.fit(X_scaled)

df["cluster"]    = model.labels_
df["customer_id"] = customer_ids.values


print("\nCLUSTER SIZES")

sizes = df["cluster"].value_counts().sort_index()

for cluster, count in sizes.items():
    pct = count / len(df) * 100
    print(
        f"Cluster {cluster}: "
        f"{count:,} customers "
        f"({pct:.1f}%)"
    )
# ============================================
# CLUSTER PROFILES
# ============================================
profile = df.groupby("cluster")[FEATURES].mean().round(2)

print("\n" + "-" * 50)
print("CLUSTER PROFILES")
print("-" * 50)
print(profile.to_string())
print("-" * 50)

# Cluster sizes
sizes = df["cluster"].value_counts().sort_index()
print("\nCLUSTER SIZES")
print("-" * 50)
for cluster, count in sizes.items():
    pct = count / len(df) * 100
    print(f"  Cluster {cluster} : {count:,} customers ({pct:.1f}%)")
print("-" * 50)

# ============================================
# SILHOUETTE SCORE — final model
# ============================================
final_sil = silhouette_score(X_scaled, model.labels_, sample_size=10000, random_state=42)
print(f"\n  Final Silhouette Score : {final_sil:.4f}")
print(f"  Number of Clusters     : {best_k}")
print("-" * 50)


output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
os.makedirs(output_dir, exist_ok=True)


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(list(k_range), inertia, marker="o", color="#1f77b4", linewidth=2)
axes[0].axvline(x=best_k, color="orange", linestyle="--", linewidth=1.5, label=f"Best K={best_k}")
axes[0].set_title("Elbow Method - Inertia vs K")
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("Inertia")
axes[0].legend()

axes[1].plot(list(k_range), silhouette, marker="o", color="#2ca02c", linewidth=2)
axes[1].axvline(x=best_k, color="orange", linestyle="--", linewidth=1.5, label=f"Best K={best_k}")
axes[1].set_title("Silhouette Score vs K")
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Silhouette Score")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "kmeans_elbow.png"), dpi=150)
plt.show()


pca        = PCA(n_components=2, random_state=42)
X_pca      = pca.fit_transform(X_scaled)
explained  = pca.explained_variance_ratio_

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

plt.figure(figsize=(10, 7))
for i in range(best_k):
    mask = model.labels_ == i
    plt.scatter(
        X_pca[mask, 0], X_pca[mask, 1],
        label=f"Cluster {i} ({sizes[i]:,})",
        alpha=0.4, s=10, color=colors[i % len(colors)]
    )
plt.title(f"Customer Segments - PCA Projection (K={best_k})")
plt.xlabel(f"PC1 ({explained[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({explained[1]*100:.1f}% variance)")
plt.legend(markerscale=3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "kmeans_clusters_pca.png"), dpi=150)
plt.show()


fig, ax = plt.subplots(figsize=(12, 5))
profile_normalized = (profile - profile.min()) / (profile.max() - profile.min())
im = ax.imshow(profile_normalized.values, cmap="YlOrRd", aspect="auto")
plt.colorbar(im, ax=ax, label="Normalized Value")
ax.set_xticks(range(len(FEATURES)))
ax.set_yticks(range(best_k))
ax.set_xticklabels(FEATURES, rotation=45, ha="right")
ax.set_yticklabels([f"Cluster {i}" for i in range(best_k)])
for i in range(best_k):
    for j in range(len(FEATURES)):
        ax.text(j, i, f"{profile.values[i, j]:.1f}",
                ha="center", va="center", fontsize=7, color="black")
plt.title("Cluster Profiles Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "kmeans_profile_heatmap.png"), dpi=150)
plt.show()


plt.figure(figsize=(8, 4))
plt.bar(
    [f"Cluster {i}" for i in sizes.index],
    sizes.values,
    color=[colors[i % len(colors)] for i in sizes.index],
    edgecolor="white"
)
plt.title("Customer Count per Cluster")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "kmeans_cluster_sizes.png"), dpi=150)
plt.show()



models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(models_dir, exist_ok=True)

with open(os.path.join(models_dir, "kmeans_model.pkl"), "wb") as f:
    pickle.dump(model, f)

with open(os.path.join(models_dir, "kmeans_scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

print("\nModel saved : ml/models/kmeans_model.pkl")
print("Scaler saved: ml/models/kmeans_scaler.pkl")