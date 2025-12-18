-- Daily Revenue Trend
-- Shows revenue and order patterns over time

SELECT 
  order_date,
  revenue,
  orders_count,
  unique_customers,
  ROUND(avg_order_value, 2) as avg_order_value,
  ROUND(avg_review_score, 2) as avg_review_score,
  
  -- 7-day moving average
  ROUND(AVG(revenue) OVER (
    ORDER BY order_date 
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
  ), 2) as revenue_7day_avg
  
FROM gold_daily_metrics
ORDER BY order_date;