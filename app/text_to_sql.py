import os
import ollama
from dotenv import load_dotenv

load_dotenv()

SQL_MODEL = os.getenv("OLLAMA_SQL_MODEL", "llama3.2:3b")

# Schema context given to the LLM so it knows your tables
SCHEMA_CONTEXT = """
You are an expert SQL generator for a PostgreSQL Data Warehouse.

TABLES AVAILABLE:
- fact_orders(order_id, customer_key, seller_key, product_key, order_date_key,
  delivered_date_key, order_status, price, freight_value, payment_value,
  payment_installments, payment_type, days_to_deliver, days_estimated,
  delivery_delay_days, review_score, order_purchase_timestamp,
  order_delivered_timestamp)

- dim_customers(customer_key, customer_id, customer_city, customer_state, customer_zip_code)

- dim_sellers(seller_key, seller_id, seller_city, seller_state, seller_zip_code)

- dim_products(product_key, product_id, category_name, category_name_english,
  product_weight_g, product_length_cm, product_height_cm, product_width_cm)

- dim_date(date_key, full_date, day, month, month_name, quarter, year,
  week_of_year, day_of_week, is_weekend)

- dim_geolocation(geo_key, zip_code, city, state, latitude, longitude)

VIEWS AVAILABLE (prefer these for performance):
- vw_monthly_sales     → columns: year, month, month_name, total_orders, total_revenue, total_freight, avg_review_score, avg_delivery_days
- vw_sales_by_category → columns: category, total_orders, total_revenue, avg_review_score  — use column 'category' NOT 'category_name'
- vw_sales_by_state    → columns: state, total_orders, total_revenue, avg_delivery_days

IMPORTANT: In vw_sales_by_category the product category column is named 'category', not 'category_name'.
RULES:
- Return ONLY the SQL query, nothing else
- No explanations, no markdown, no code blocks
- Always use LIMIT 100 unless the user asks for totals or aggregations
- Use views when the question matches their scope
- Always use lowercase table and column names
- For revenue questions use SUM(price)
- For date filtering use dim_date joined to fact_orders on order_date_key
"""

def generate_sql(question: str) -> str:
    prompt = f"{SCHEMA_CONTEXT}\n\nUser question: {question}\n\nSQL query:"

    response = ollama.chat(
        model=SQL_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    sql = response["message"]["content"].strip()

    # Clean any markdown the model may add
    sql = sql.replace("```sql", "").replace("```", "").strip()

    return sql