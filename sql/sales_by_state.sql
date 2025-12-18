-- Sales Performance by State
-- Geographic distribution of revenue and customers

SELECT 
  customer_state as state,
  SUM(total_orders) as total_orders_sum,
  SUM(total_revenue) as total_revenue_sum,
  AVG(avg_order_value) as avg_order_value,
  SUM(unique_customers) as total_customers_sum,
  ROUND(AVG(avg_review_score), 2) as avg_review_score,
  ROUND(AVG(avg_delivery_days), 1) as avg_delivery_days
  
FROM workspace.default.gold_geographic_metrics
GROUP BY customer_state
ORDER BY total_revenue_sum DESC
LIMIT 15;