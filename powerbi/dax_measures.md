# Core KPI Measures
## Customer Metrics (`powerbi_customer_metrics` table)
### Total Revenue
```dax
Total Revenue = 
SUM(powerbi_customer_metrics[monetary_value])
```
### Total Revenue (Formatted for card)
```dax
Total Revenue (Formatted) = 
"R$ " & FORMAT([Total Revenue], "#,##0.00")
```
### Total Orders
```dax
Total Orders = 
SUM(powerbi_customer_metrics[frequency])
```
### Total Orders (Formatted for card)
```dax
Total Orders (Formatted) = 
FORMAT([Total Orders], "#,##0")
```
### Total Customers
```dax
Total Customers = 
DISTINCTCOUNT(powerbi_customer_metrics[customer_unique_id])
```
### Average Order Value
```dax
Average Order Value = 
DIVIDE([Total Revenue], [Total Orders], 0)
```
### Average Order Value (Formatted for card)
```dax
AOV (Formatted) = 
"R$ " & FORMAT([Average Order Value], "#,##0.00")
```
### Average Review Score
```dax
Avg Review Score = 
AVERAGE(powerbi_customer_metrics[avg_review_score])
```
### Average Delivery Days
```dax
Avg Delivery Days = 
AVERAGE(powerbi_customer_metrics[avg_delivery_days])
```
### Average Delivery Delay
```dax
Avg Delivery Delay = 
AVERAGE(powerbi_customer_metrics[avg_delivery_delay])
```
### Total CLV
```dax
Total CLV = 
SUM('Customer Metrics'[clv_12_months])
```
### Average CLV per Customer
```dax
Avg CLV per Customer = 
DIVIDE([Total CLV], [Total Customers], 0)
```
## Churn Analysis (`powerbi_churn_analysis` table)
### Customers at Risk
```dax
Customers at Risk = 
CALCULATE(
    DISTINCTCOUNT(powerbi_churn_analysis[customer_unique_id]),
    powerbi_churn_analysis[churn_risk_category] = "High Risk"
)
```
### Revenue at Risk
```dax
Revenue at Risk = 
CALCULATE(
    SUM(powerbi_churn_analysis[potential_revenue_loss]),
    powerbi_churn_analysis[churn_risk_category] = "High Risk"
)
```
### Churn Rate
```dax
Churn Rate = 
DIVIDE(
    COUNTROWS(FILTER(powerbi_churn_analysis, powerbi_churn_analysis[predicted_churn] = 1)),
    COUNTROWS(powerbi_churn_analysis),
    0
) * 100
```
# Time Intelligence Measures
## Daily Metrics (`powerbi_daily_metrics` table)
### Revenue Yesterday
```dax
Revenue Yesterday = 
CALCULATE(
    SUM(powerbi_daily_metrics[revenue]),
    DATEADD(powerbi_daily_metrics[order_date], -1, DAY)
)
```
### Revenue vs Yesterday
```dax
Revenue vs Yesterday = 
[Total Revenue] - [Revenue Yesterday]
```
### Revenue vs Yesterday %
```dax
Revenue vs Yesterday % = 
DIVIDE([Revenue vs Yesterday], [Revenue Yesterday], 0) * 100
```
# Category Performance Measures
## Category Metrics (`powerbi_category_metrics` table)
### Total Categories
```dax
Total Categories = 
DISTINCTCOUNT(powerbi_category_metrics[category])
```
### Average Revenue per Category
```dax
Avg Revenue per Category = 
DIVIDE(
    SUM(powerbi_category_metrics[total_revenue]),
    [Total Categories],
    0
)
```