-- RFM Customer Segmentation Analysis
-- Shows customer distribution across segments with key metrics

SELECT 
  customer_segment,
  COUNT(*) as customer_count,
  CONCAT('R$ ', FORMAT_NUMBER(SUM(monetary_value), 2)) as total_revenue,
  CONCAT('R$ ', FORMAT_NUMBER(AVG(monetary_value), 2)) as avg_customer_value,
  ROUND(AVG(frequency), 1) as avg_orders_per_customer,
  ROUND(AVG(recency_days), 0) as avg_recency_days,
  ROUND(AVG(avg_review_score), 2) as avg_review_score,
  CONCAT('R$ ', FORMAT_NUMBER(AVG(clv_12_months), 2)) as avg_clv_12m
  
FROM gold_customer_metrics
GROUP BY customer_segment
ORDER BY SUM(monetary_value) DESC;