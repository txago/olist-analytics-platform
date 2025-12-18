-- KPIs Overview: High-level business metrics
-- Shows: Total Revenue, Orders, Customers, AOV, Avg Review Score

SELECT 
  -- Revenue Metrics
  CONCAT('R$ ', FORMAT_NUMBER(SUM(monetary_value), 2)) as total_revenue,
  CONCAT('R$ ', FORMAT_NUMBER(AVG(avg_order_value), 2)) as avg_order_value,
  
  -- Volume Metrics
  FORMAT_NUMBER(SUM(frequency), 0) as total_orders,
  FORMAT_NUMBER(COUNT(DISTINCT customer_unique_id), 0) as total_customers,
  
  -- Quality Metrics
  ROUND(AVG(avg_review_score), 2) as avg_review_score,
  
  -- Delivery Performance
  ROUND(AVG(avg_delivery_days), 1) as avg_delivery_days,
  ROUND(AVG(avg_delivery_delay), 1) as avg_delivery_delay_days
  
FROM gold_customer_metrics
WHERE monetary_value > 0;