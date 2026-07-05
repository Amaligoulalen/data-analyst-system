
CREATE TABLE IF NOT EXISTS dim_customers (
    customer_key        SERIAL PRIMARY KEY,
    customer_id         VARCHAR(50) UNIQUE NOT NULL,
    customer_city       VARCHAR(100),
    customer_state      VARCHAR(10),
    customer_zip_code   VARCHAR(20)
);


CREATE TABLE IF NOT EXISTS dim_sellers (
    seller_key          SERIAL PRIMARY KEY,
    seller_id           VARCHAR(50) UNIQUE NOT NULL,
    seller_city         VARCHAR(100),
    seller_state        VARCHAR(10),
    seller_zip_code     VARCHAR(20)
);

CREATE TABLE IF NOT EXISTS dim_products (
    product_key             SERIAL PRIMARY KEY,
    product_id              VARCHAR(50) UNIQUE NOT NULL,
    category_name           VARCHAR(100),
    category_name_english   VARCHAR(100),
    product_weight_g        FLOAT,
    product_length_cm       FLOAT,
    product_height_cm       FLOAT,
    product_width_cm        FLOAT
);


CREATE TABLE IF NOT EXISTS dim_date (
    date_key        SERIAL PRIMARY KEY,
    full_date       DATE UNIQUE NOT NULL,
    day             INT,
    month           INT,
    month_name      VARCHAR(20),
    quarter         INT,
    year            INT,
    week_of_year    INT,
    day_of_week     VARCHAR(20),
    is_weekend      BOOLEAN
);

CREATE TABLE IF NOT EXISTS dim_geolocation (
    geo_key         SERIAL PRIMARY KEY,
    zip_code        VARCHAR(20),
    city            VARCHAR(100),
    state           VARCHAR(10),
    latitude        FLOAT,
    longitude       FLOAT
);


CREATE TABLE IF NOT EXISTS fact_orders (
    order_key               SERIAL PRIMARY KEY,
    order_id                VARCHAR(50) NOT NULL,

    
    customer_key            INT REFERENCES dim_customers(customer_key),
    seller_key              INT REFERENCES dim_sellers(seller_key),
    product_key             INT REFERENCES dim_products(product_key),
    order_date_key          INT REFERENCES dim_date(date_key),
    delivered_date_key      INT REFERENCES dim_date(date_key),

  
    order_status            VARCHAR(50),

   
    price                   NUMERIC(10, 2),     
    freight_value           NUMERIC(10, 2),     
    payment_value           NUMERIC(10, 2),     
    payment_installments    INT,
    payment_type            VARCHAR(50),

   
    days_to_deliver         INT,             
    days_estimated          INT,               
    delivery_delay_days     INT,                

   
    review_score            INT,                -- 1 to 5

    order_purchase_timestamp    TIMESTAMP,
    order_delivered_timestamp   TIMESTAMP
);




CREATE INDEX IF NOT EXISTS idx_fact_orders_customer    ON fact_orders(customer_key);
CREATE INDEX IF NOT EXISTS idx_fact_orders_product     ON fact_orders(product_key);
CREATE INDEX IF NOT EXISTS idx_fact_orders_seller      ON fact_orders(seller_key);
CREATE INDEX IF NOT EXISTS idx_fact_orders_date        ON fact_orders(order_date_key);
CREATE INDEX IF NOT EXISTS idx_fact_orders_status      ON fact_orders(order_status);



CREATE OR REPLACE VIEW vw_monthly_sales AS
SELECT
    d.year,
    d.month,
    d.month_name,
    COUNT(DISTINCT f.order_id)      AS total_orders,
    SUM(f.price)                    AS total_revenue,
    SUM(f.freight_value)            AS total_freight,
    AVG(f.review_score)             AS avg_review_score,
    AVG(f.days_to_deliver)          AS avg_delivery_days
FROM fact_orders f
JOIN dim_date d ON f.order_date_key = d.date_key
WHERE f.order_status = 'delivered'
GROUP BY d.year, d.month, d.month_name
ORDER BY d.year, d.month;



CREATE OR REPLACE VIEW vw_sales_by_category AS
SELECT
    p.category_name_english         AS category,
    COUNT(DISTINCT f.order_id)      AS total_orders,
    SUM(f.price)                    AS total_revenue,
    AVG(f.review_score)             AS avg_review_score
FROM fact_orders f
JOIN dim_products p ON f.product_key = p.product_key
GROUP BY p.category_name_english
ORDER BY total_revenue DESC;



CREATE OR REPLACE VIEW vw_sales_by_state AS
SELECT
    c.customer_state                AS state,
    COUNT(DISTINCT f.order_id)      AS total_orders,
    SUM(f.price)                    AS total_revenue,
    AVG(f.days_to_deliver)          AS avg_delivery_days
FROM fact_orders f
JOIN dim_customers c ON f.customer_key = c.customer_key
GROUP BY c.customer_state
ORDER BY total_revenue DESC;