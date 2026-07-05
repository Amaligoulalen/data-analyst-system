import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv


load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "ai_analytics")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Path to your raw CSVs
RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

def get_engine():
    engine = create_engine(DATABASE_URL)
    
    return engine


# ============================================
# 1. LOAD dim_customers
# ============================================
def load_dim_customers(engine):
    
    df = pd.read_csv(os.path.join(RAW_DATA_PATH, "olist_customers_dataset.csv"))

    df = df.rename(columns={
        "customer_id":                  "customer_id",
        "customer_city":                "customer_city",
        "customer_state":               "customer_state",
        "customer_zip_code_prefix":     "customer_zip_code"
    })

    df = df[["customer_id", "customer_city", "customer_state", "customer_zip_code"]]
    df = df.drop_duplicates(subset="customer_id")

    df.to_sql("dim_customers", engine, if_exists="append", index=False,
              method="multi", chunksize=1000)
    


# ============================================
# 2. LOAD dim_sellers
# ============================================
def load_dim_sellers(engine):
    
    df = pd.read_csv(os.path.join(RAW_DATA_PATH, "olist_sellers_dataset.csv"))

    df = df.rename(columns={
        "seller_id":                "seller_id",
        "seller_city":              "seller_city",
        "seller_state":             "seller_state",
        "seller_zip_code_prefix":   "seller_zip_code"
    })

    df = df[["seller_id", "seller_city", "seller_state", "seller_zip_code"]]
    df = df.drop_duplicates(subset="seller_id")

    df.to_sql("dim_sellers", engine, if_exists="append", index=False,
              method="multi", chunksize=1000)
    


# ============================================
# 3. LOAD dim_products
# ============================================
def load_dim_products(engine):
    
    products = pd.read_csv(os.path.join(RAW_DATA_PATH, "olist_products_dataset.csv"))
    translation = pd.read_csv(os.path.join(RAW_DATA_PATH, "product_category_name_translation.csv"))

    # Merge translation into products
    df = products.merge(translation, on="product_category_name", how="left")

    df = df.rename(columns={
        "product_category_name":            "category_name",
        "product_category_name_english":    "category_name_english",
        "product_weight_g":                 "product_weight_g",
        "product_length_cm":                "product_length_cm",
        "product_height_cm":                "product_height_cm",
        "product_width_cm":                 "product_width_cm"
    })

    df = df[[
        "product_id", "category_name", "category_name_english",
        "product_weight_g", "product_length_cm",
        "product_height_cm", "product_width_cm"
    ]]
    df = df.drop_duplicates(subset="product_id")

    df.to_sql("dim_products", engine, if_exists="append", index=False,
              method="multi", chunksize=1000)
    


# ============================================
# 4. LOAD dim_date
# ============================================
def load_dim_date(engine):
    
    orders = pd.read_csv(os.path.join(RAW_DATA_PATH, "olist_orders_dataset.csv"))

    # Collect all unique dates from order timestamps
    date_cols = [
        "order_purchase_timestamp",
        "order_delivered_customer_date"
    ]

    all_dates = []
    for col in date_cols:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")
        all_dates.extend(orders[col].dropna().dt.date.tolist())

    unique_dates = sorted(set(all_dates))

    date_df = pd.DataFrame({"full_date": pd.to_datetime(unique_dates)})
    date_df["day"]          = date_df["full_date"].dt.day
    date_df["month"]        = date_df["full_date"].dt.month
    date_df["month_name"]   = date_df["full_date"].dt.strftime("%B")
    date_df["quarter"]      = date_df["full_date"].dt.quarter
    date_df["year"]         = date_df["full_date"].dt.year
    date_df["week_of_year"] = date_df["full_date"].dt.isocalendar().week.astype(int)
    date_df["day_of_week"]  = date_df["full_date"].dt.strftime("%A")
    date_df["is_weekend"]   = date_df["full_date"].dt.dayofweek >= 5
    date_df["full_date"]    = date_df["full_date"].dt.date

    date_df.to_sql("dim_date", engine, if_exists="append", index=False,
                   method="multi", chunksize=1000)
    


# ============================================
# 5. LOAD dim_geolocation
# ============================================
def load_dim_geolocation(engine):
    
    df = pd.read_csv(os.path.join(RAW_DATA_PATH, "olist_geolocation_dataset.csv"))

    df = df.rename(columns={
        "geolocation_zip_code_prefix":  "zip_code",
        "geolocation_city":             "city",
        "geolocation_state":            "state",
        "geolocation_lat":              "latitude",
        "geolocation_lng":              "longitude"
    })

    df = df[["zip_code", "city", "state", "latitude", "longitude"]]
    # Keep one row per zip code (average lat/lng)
    df = df.groupby(["zip_code", "city", "state"], as_index=False).agg({
        "latitude": "mean",
        "longitude": "mean"
    })

    df.to_sql("dim_geolocation", engine, if_exists="append", index=False,
              method="multi", chunksize=1000)
    print(f"    {len(df)} geolocation records loaded")


# ============================================
# 6. LOAD fact_orders
# ============================================
def load_fact_orders(engine):
    

    # Load all source files
    orders   = pd.read_csv(os.path.join(RAW_DATA_PATH, "olist_orders_dataset.csv"))
    items    = pd.read_csv(os.path.join(RAW_DATA_PATH, "olist_order_items_dataset.csv"))
    payments = pd.read_csv(os.path.join(RAW_DATA_PATH, "olist_order_payments_dataset.csv"))
    reviews  = pd.read_csv(os.path.join(RAW_DATA_PATH, "olist_order_reviews_dataset.csv"))

    # Parse timestamps
    orders["order_purchase_timestamp"]      = pd.to_datetime(orders["order_purchase_timestamp"], errors="coerce")
    orders["order_delivered_customer_date"] = pd.to_datetime(orders["order_delivered_customer_date"], errors="coerce")
    orders["order_estimated_delivery_date"] = pd.to_datetime(orders["order_estimated_delivery_date"], errors="coerce")

    # Aggregate payments per order
    payments_agg = payments.groupby("order_id", as_index=False).agg(
        payment_value=("payment_value", "sum"),
        payment_installments=("payment_installments", "max"),
        payment_type=("payment_type", "first")
    )

    # Aggregate reviews per order (keep highest score if multiple)
    reviews_agg = reviews.groupby("order_id", as_index=False).agg(
        review_score=("review_score", "max")
    )

    # Merge everything into orders
    df = orders.merge(items,        on="order_id", how="left")
    df = df.merge(payments_agg,     on="order_id", how="left")
    df = df.merge(reviews_agg,      on="order_id", how="left")

    # Calculate delivery metrics
    df["days_to_deliver"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.days

    df["days_estimated"] = (
        df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]
    ).dt.days

    df["delivery_delay_days"] = df["days_to_deliver"] - df["days_estimated"]

    # Load dimension keys from DB
    with engine.connect() as conn:
        customers  = pd.read_sql("SELECT customer_key, customer_id FROM dim_customers", conn)
        sellers    = pd.read_sql("SELECT seller_key, seller_id FROM dim_sellers", conn)
        products   = pd.read_sql("SELECT product_key, product_id FROM dim_products", conn)
        dates      = pd.read_sql("SELECT date_key, full_date FROM dim_date", conn)

    dates["full_date"] = pd.to_datetime(dates["full_date"]).dt.date

    # Map foreign keys
    df = df.merge(customers, on="customer_id", how="left")
    df = df.merge(sellers,   on="seller_id",   how="left")
    df = df.merge(products,  on="product_id",  how="left")

    # Map order date key
    df["order_date"] = df["order_purchase_timestamp"].dt.date
    df = df.merge(dates.rename(columns={"date_key": "order_date_key", "full_date": "order_date"}),
                  on="order_date", how="left")

    # Map delivered date key
    df["delivered_date"] = df["order_delivered_customer_date"].dt.date
    df = df.merge(dates.rename(columns={"date_key": "delivered_date_key", "full_date": "delivered_date"}),
                  on="delivered_date", how="left")

    # Select final columns
    fact = df[[
        "order_id", "customer_key", "seller_key", "product_key",
        "order_date_key", "delivered_date_key", "order_status",
        "price", "freight_value", "payment_value",
        "payment_installments", "payment_type",
        "days_to_deliver", "days_estimated", "delivery_delay_days",
        "review_score", "order_purchase_timestamp",
        "order_delivered_customer_date"
    ]].rename(columns={
        "order_delivered_customer_date": "order_delivered_timestamp"
    })

    fact.to_sql("fact_orders", engine, if_exists="append", index=False,
                method="multi", chunksize=500)
    


# ============================================
# MAIN — Run all loaders in order
# ============================================
if __name__ == "__main__":
    
    engine = get_engine()

    try:
        load_dim_customers(engine)
        load_dim_sellers(engine)
        load_dim_products(engine)
        load_dim_date(engine)
        load_dim_geolocation(engine)
        load_fact_orders(engine)
        
    except Exception as e:
        print(f" ETL Failed  {e}")
        raise