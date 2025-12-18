-- Top Customers by CLV
-- High-value customer identification

SELECT 
  customer_unique_id,
  customer_city,
  customer_state,
  customer_segment,
  CONCAT('R$ ', FORMAT_NUMBER(monetary_value, 2)) as total_spent,
  frequency as total_orders,
  CONCAT('R$ ', FORMAT_NUMBER(avg_order_value, 2)) as avg_order_value,
  CONCAT('R$ ', FORMAT_NUMBER(clv_12_months, 2)) as projected_clv_12m,
  recency_days as days_since_last_order,
  ROUND(avg_review_score, 2) as avg_review_score,
  rfm_score
  
FROM gold_customer_metrics
WHERE monetary_value > 0
ORDER BY clv_12_months DESC
LIMIT 100;