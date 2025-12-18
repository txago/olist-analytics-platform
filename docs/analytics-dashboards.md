# 📊 Analytics Dashboards Documentation

## Overview

This project includes **two complementary dashboard solutions**:
1. **Databricks SQL Dashboards** - Real-time operational monitoring
2. **Power BI Dashboards** - Executive reporting and deep-dive analysis

---

## 1. Databricks SQL Dashboards

### Dashboard: Executive Overview

**Purpose:** High-level business performance monitoring

**Key Visualizations:**

#### KPI Cards (Top Row)
- **Total Revenue:** R$ 15,422,605.23
- **Total Orders:** 96,478
- **Total Customers:** 93,358
- **Average Order Value:** R$ 159.86
- **Avg Review Score:** 4.08 / 5.0
- **Avg Delivery Days:** 12.5 days

#### Revenue Trend (Line Chart)
- **X-axis:** Order date (daily)
- **Y-axis:** Revenue (R$)
- **Additional line:** 7-day moving average
- **Insight:** Identifies seasonal patterns and growth trends

#### Sales by State (Map/Bar Chart)
- **Top state:** SP (São Paulo) - R$ 5.8M
- **Fastest growing:** MG (Minas Gerais) - +23% MoM
- **Delivery performance:** Correlated with distance from distribution centers

---

### Dashboard: Customer Analytics

**Purpose:** Deep-dive into customer behavior and segmentation

**Key Visualizations:**

#### Customer Segmentation (Donut Chart)
| Segment | Customers (%) | Revenue | Avg Value | Status |
|---------|-----------|---------|-----------|--------|
| **Recent Customers** | 23,716 (24.64%) | R$ 3.1M | R$ 133 | Bought recently, watching |
| **Promising** | 23,735 (24.47%) | R$ 1.2M | R$ 54 | Recent activity emerging |
| **Loyal Customers** | 19,295 (20%) | R$ 8.2M | R$ 425 | Consistent buyers |
| **Customers Needing Attention** | 14,857 (15.46%) | R$ 0.8M | R$ 54 | Used to purchase, warning signs detected |
| **Potential Loyalists** | 14,875 (15.43%) | R$ 2M | R$ 132 | Recent promising buyers |

#### RFM Distribution (Scatter Plot)
- **X-axis:** Recency (days since last purchase)
- **Y-axis:** Frequency (number of orders)
- **Bubble size:** Monetary value
- **Color:** Customer segment
- **Insight:** Visual identification of high-value customers needing attention

#### Top 20 Customers (Table)
- Columns: Customer ID, City, State, Segment, Total Spent, Orders, CLV
- Sorted by: CLV (12-month projection) descending
- **Action:** Export for personalized marketing campaigns

---

### Dashboard: Churn & Retention

**Purpose:** Identify at-risk customers and retention opportunities

**Key Visualizations:**

#### Churn Risk Funnel
| Risk Level  | Customers | Potential Revenue Loss |
|-------------|-----------|------------------------|
| High Risk   | 20,847    | R$ 3,135,424.72        |
| Medium Risk | 32,088    | R$ 5,046,218.18        |
| Low Risk    | 50,135    | -                      |

#### Churn Probability Distribution (Histogram)
- **X-axis:** Churn probability (0-100%)
- **Y-axis:** Customer count
- **Peak:** Most customers at 10-20% probability (healthy)
- **Long tail:** High-risk customers (70-100%) requiring intervention

#### At-Risk Customer Details (Table)
- Filters: High Risk only
- Columns: Customer ID, Segment, Recency, Value, Churn %, Last Order Date
- **Sorted by:** Churn probability descending
- **Action items:** 
  - Send re-engagement email
  - Offer loyalty discount
  - Personalized product recommendations

---

### Dashboard: Product Performance

**Purpose:** Optimize product portfolio and pricing

**Key Visualizations:**

#### Revenue by Category (Treemap)
- **Size:** Revenue contribution
- **Color:** Growth rate (green = growing, red = declining)
- **Top 5 categories:**
  1. Health & Beauty - R$ 1,258,681.34
  2. Watches & Gifts - R$ 1,205,005.68
  3. Bed, Bath & Table - R$ 1,036,988.68
  4. Sports & Leisure - R$ 988,048.98
  5. Computers Accessories - R$ 911,954.32

#### Category Performance Matrix (Table)
| Category              | Orders | Revenue  | Avg Price | Unique Products |
|-----------------------|--------|----------|-----------|-----------------|
| Health & Beauty       | 9,670  | R$ 1.25M | R$ 130    | 2,444           |
| Watches & Gifts       | 5,991  | R$ 1.20M | R$ 201    | 1,329           |
| Bed, Bath & Table     | 11,115 | R$ 1.04M | R$ 93     | 3,029           |
| Sports & Leisure      | 8,641  | R$ 988K  | R$ 114    | 2,867           |
| Computers Accessories | 7,827  | R$ 912K  | R$ 117    | 1,639           |

**Insights:**
- High AOV categories → Focus on premium marketing
- High volume/low price → Bundle offers
- Low performers → Consider discontinuation

---

## 2. Power BI Dashboards

### Why Power BI in Addition to Databricks?

**Complementary Strengths:**
- **Databricks SQL:** Real-time monitoring, operational queries
- **Power BI:** Advanced visuals, offline access, executive presentations

**Use Cases:**
- Quarterly board meetings (PDF export)
- Offline analysis during travel
- Integration with Excel/PowerPoint
- Mobile app for executives

---

### Power BI: Dashboard 1 - Executive Overview

**Design:** 
- Clean, minimalist design
- Focus on key metrics
- One-page snapshot for busy executives
- Line chart layout

**Key Features:**
- **Date slicer:** Filter entire page by date range
- **State filter:** Focus on specific regions
- **Drill-through:** Click state → See detailed city breakdown
- **Export:** PDF for email distribution

---

### Power BI: Dashboard 2 - Customer Analytics

**Purpose:** Actionable insights for marketing teams

**Key Visuals:**

#### 1. Customer Segment Distribution (Donut + Bar)
- **Donut:** Count of customers per segment
- **Bar:** Revenue contribution per segment
- **Insight:** Champions = 8.6% of customers but 23% of revenue

#### 2. RFM Scatter Plot
- **Interactive:** Hover to see customer details
- **Filters:** Filter by segment, state, or value range
- **Action:** Select high-value, high-recency customers for VIP program

#### 3. Customer Lifetime Value (CLV) Analysis
- **Histogram:** Distribution of projected CLV
- **Conditional formatting:** Green = high CLV, Red = low CLV
- **Top 10 table:** Highest CLV customers with contact info

#### 4. Cohort Analysis (Matrix)
- **Rows:** Order month (cohort)
- **Columns:** Months since first order
- **Values:** Retention rate (%)
- **Insight:** Month 1 retention = 28%, Month 6 = 12%

---

### Power BI: Dashboard 3 - Churn Management

**Purpose:** Proactive customer retention

**Key Visuals:**

#### 1. Churn Risk Summary Cards
| High Risk | Medium Risk | Churn Rate | 
|-----------|-------------|------------|
| 20,847 customers | 32,088 customers | 59.6% |
| R$ 3.13M at risk | R$ 5.04M at risk | x |

#### 2. Churn Probability Funnel
- **Stage 1:** Low Risk (50,135 customers)
- **Stage 2:** Medium Risk (32,088 customers)
- **Stage 3:** High Risk (20,847 customers)
- **Color:** Gradient from green → yellow → red

#### 3. Feature Importance (Bar Chart)
Shows which factors most predict churn:
1. Recency days (38% importance)
2. Average review score (22%)
3. Delivery delay (18%)
4. Frequency (12%)
5. Monetary value (10%)

**Action:** Address delivery issues to reduce churn

#### 4. At-Risk Customer List (Table with Drill-through)
- Click customer → See full order history
- Includes: Last order date, total spent, churn probability
- **Conditional formatting:** Red if churn probability > 70%

---

### Power BI: Dashboard 4 - Product Performance

**Purpose:** Optimize inventory and pricing

**Key Visuals:**

#### 1. Revenue by Category (Treemap)
- Interactive: Click category → Filter entire page
- Size = Revenue, Color = Profit margin

#### 2. Category Performance Matrix
| Metric    | Health & Beauty | Computers | Furniture & Decor |
|-----------|-----------------|-----------|-----------|
| Revenue | R$ 1.25M | R$ 988K | R$ 876K |
| Revenue % | 9.4% | 1.7% | 5.4% |
| Orders | 9,670 | 203 | 8,334 |
| Avg Price | R$ 130 | R$ 1,098 | R$ 88 |

#### 3. Price vs. Sales Volume (Scatter)
- **X-axis:** Average price
- **Y-axis:** Total units sold
- **Bubble size:** Revenue
- **Quadrants:**
  - Top-left: High volume, low price (mass market)
  - Top-right: High volume, high price (premium best-sellers)
  - Bottom-left: Low volume, low price (consider discontinuing)
  - Bottom-right: Low volume, high price (luxury niche)

#### 4. Product Recommendations
- **AI-powered:** Uses association rules mining
- "Customers who bought X also bought Y"
- **Action:** Create product bundles

---

## DAX Measures Reference

### Core Metrics

```dax
Total Revenue = 
SUM(powerbi_customer_metrics[monetary_value])

Total Orders = 
SUM(powerbi_customer_metrics[frequency])

Total Customers = 
DISTINCTCOUNT(powerbi_customer_metrics[customer_unique_id])

Average Order Value = 
DIVIDE([Total Revenue], [Total Orders], 0)
```

### Time Intelligence

```dax
Revenue MTD = 
TOTALMTD([Total Revenue], powerbi_daily_metrics[order_date])

Revenue YTD = 
TOTALYTD([Total Revenue], powerbi_daily_metrics[order_date])

Revenue vs Last Month = 
[Total Revenue] - CALCULATE(
    [Total Revenue],
    DATEADD(powerbi_daily_metrics[order_date], -1, MONTH)
)

Revenue Growth % = 
DIVIDE([Revenue vs Last Month], 
    CALCULATE([Total Revenue], 
        DATEADD(powerbi_daily_metrics[order_date], -1, MONTH)
    ), 0) * 100
```

### Advanced Analytics

```dax
Customer Lifetime Value (CLV) = 
AVERAGEX(
    powerbi_customer_metrics,
    powerbi_customer_metrics[monetary_value] / 
    powerbi_customer_metrics[customer_lifetime_days] * 365
)

Churn Rate = 
DIVIDE(
    CALCULATE(
        COUNTROWS(powerbi_churn_analysis),
        powerbi_churn_analysis[predicted_churn] = 1
    ),
    COUNTROWS(powerbi_churn_analysis),
    0
) * 100

Revenue at Risk = 
CALCULATE(
    SUM(powerbi_customer_metrics[monetary_value]),
    powerbi_churn_analysis[churn_risk_category] = "High Risk"
)
```

### Conditional Formatting

```dax
Churn Risk Color = 
SWITCH(
    TRUE(),
    powerbi_churn_analysis[churn_probability] >= 0.7, "Red",
    powerbi_churn_analysis[churn_probability] >= 0.4, "Yellow",
    "Green"
)

Segment Color = 
SWITCH(
    powerbi_customer_metrics[customer_segment],
    "Champions", "#2ECC71",
    "Loyal Customers", "#3498DB",
    "At Risk", "#F39C12",
    "Lost", "#E74C3C",
    "#95A5A6"
)
```

---

## Dashboard Design Best Practices Applied

### Color Scheme
- **Primary:** Blue (#0A4EE4) - Olist main color
- **Success:** Green (#2ECC71) - Positive metrics
- **Warning:** Yellow (#F39C12) - Needs attention
- **Danger:** Red (#E74C3C) - Critical issues
- **Text Neutral:** Gray (#605E5C) - Text and secondary info
- **Background Neutral:** Gray (#E6E6E6) - Background color

### Typography
- **Headers:** Arial Unicode MS, 12pt
- **KPIs:** Arial Unicode MS Bold, 16pt
- **Body text:** Arial Unicode MS, 9pt
- **Data labels:** Arial Unicode MS, 9pt

### Layout Principles
1. **F-Pattern:** Most important info top-left
2. **Z-Pattern:** Guide eye across dashboard
3. **Grouping:** Related metrics together
4. **White space:** Prevent visual clutter
5. **Consistency:** Same chart types for similar data

### Interactivity
- **Slicers:** Date, State, Segment
- **Cross-filtering:** Click any visual → Filter others
- **Drill-through:** Right-click → See details
- **Tooltips:** Hover for additional context
- **Bookmarks:** Save specific views

---

## Performance Optimization

### Power BI File Size
- **Current:** ~50 MB
- **Optimization techniques:**
  - Import only necessary columns
  - Aggregate data in Databricks (not Power BI)
  - Remove unused columns/tables
  - Compress images

### Query Performance
- **Typical refresh time:** 30-60 seconds
- **Optimization:**
  - Use DirectQuery for real-time data
  - Schedule refresh during off-peak hours
  - Enable incremental refresh for large tables

---

## Sharing & Collaboration

### Databricks SQL Dashboards
- **Internal:** Anyone with Databricks access
- **External:** Generate shareable links (read-only)
- **Scheduling:** Email PDF snapshots daily/weekly

### Power BI Dashboards
- **Publish to workspace:** Share with team
- **Embed in website:** For public dashboards
- **Power BI Mobile:** iOS/Android apps
- **Export options:** PDF, PowerPoint, Excel

---

## Maintenance & Updates

### Weekly Tasks
- ✅ Verify data refresh completed
- ✅ Check for query errors
- ✅ Review performance metrics

### Monthly Tasks
- ✅ Add new KPIs as requested
- ✅ Update color schemes if needed
- ✅ Archive old report versions

### Quarterly Tasks
- ✅ User feedback review
- ✅ Dashboard redesign if needed
- ✅ Train new users

---

## User Guide

### For Executives
1. Start with **Executive Overview** dashboard
2. Review KPI cards (top metrics)
3. Check revenue trend (is it growing?)
4. Drill into problem areas (state, category)

### For Marketing Teams
1. Open **Customer Analytics** dashboard
2. Identify target segments (Champions, Loyalists)
3. Export customer lists for campaigns
4. Review **Churn Management** for retention

### For Product Managers
1. Open **Product Performance** dashboard
2. Identify best/worst categories
3. Analyze price vs. volume
4. Make inventory/pricing decisions

---

## Future Enhancements

### Planned Features
1. **Real-time alerting:** Email when churn risk spikes
2. **Predictive analytics:** Forecast next month's revenue
3. **A/B testing dashboard:** Compare marketing campaigns
4. **Customer journey mapping:** Visualize purchase paths

### Advanced Visuals
1. **Sankey diagram:** Customer flow between segments
2. **Network graph:** Product association rules
3. **Geographic heatmap:** Delivery performance by region
4. **Timeline:** Customer lifecycle milestones

---

## Screenshots

### Databricks - Product Performance
![Product Dashboard](../images/olist-dashboard-databricks.png)
*Real-time monitoring for all SQL Queries*

### Power BI - Executive Overview
![Executive Dashboard](../images/olist-dashboard-powerbi-1-executive-overview.png)
*KPIs: Revenue, Orders, Customers, AOV | Revenue trends and geographic distribution*

### Power BI - Customer Analytics
![Customer Dashboard](../images/olist-dashboard-powerbi-2-customer-analytics.png)
*RFM segmentation, customer lifetime value, and behavioral patterns*

### Power BI - Churn Management
![Churn Dashboard](../images/olist-dashboard-powerbi-3-churn-management.png)
*At-risk customers, churn probability distribution, and retention strategies*

### Power BI - Product Performance
![Product Dashboard](../images/olist-dashboard-powerbi-4-product-performance.png)
*Category performance, product rankings, and revenue contribution*

---

## Conclusion

These dashboards transform raw data into **actionable business intelligence**. They enable:
- **Data-driven decision making** at all levels
- **Proactive customer retention** through churn prediction
- **Optimized marketing spend** via segmentation
- **Product portfolio optimization** based on performance

The combination of **Databricks SQL** (real-time) and **Power BI** (presentation) provides a comprehensive analytics solution.