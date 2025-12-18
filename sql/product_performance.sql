-- Top Categories by Revenue
-- Shows best performing categories with proper numeric values for visualization

SELECT 
  category_english as category,
  total_orders,
  total_revenue,
  avg_item_price,
  unique_products,
  revenue_rank
  
FROM workspace.default.gold_category_metrics
WHERE category_english IS NOT NULL
ORDER BY total_revenue DESC
LIMIT 20;