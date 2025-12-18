-- Churn Risk Analysis
-- Identifies at-risk customers and potential revenue impact

SELECT 
  c.churn_risk_category,
  COUNT(*) as customer_count,
  CONCAT('R$ ', FORMAT_NUMBER(SUM(m.monetary_value), 2)) as total_revenue_at_risk,
  CONCAT('R$ ', FORMAT_NUMBER(AVG(m.monetary_value), 2)) as avg_customer_value,
  ROUND(AVG(c.churn_probability) * 100, 1) as avg_churn_probability_pct,
  ROUND(AVG(m.recency_days), 0) as avg_days_since_last_order,
  ROUND(AVG(m.avg_review_score), 2) as avg_review_score
  
FROM gold_customer_churn_predictions c
JOIN gold_customer_metrics m ON c.customer_unique_id = m.customer_unique_id
GROUP BY c.churn_risk_category
ORDER BY 
  CASE c.churn_risk_category
    WHEN 'High Risk' THEN 1
    WHEN 'Medium Risk' THEN 2
    WHEN 'Low Risk' THEN 3
  END;