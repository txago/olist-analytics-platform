# 🚀 Setup Guide - Reproduce This Project

This guide will help you recreate the entire Olist Analytics Platform from scratch in about **2-3 hours**.

---

## Prerequisites

### Required Accounts (All Free)
- ✅ **Databricks Community Edition** (or trial): [signup link](https://community.cloud.databricks.com/login.html)
- ✅ **Kaggle account** (for dataset): [kaggle.com](https://www.kaggle.com/)
- ✅ **Power BI Desktop** (optional): [download](https://powerbi.microsoft.com/desktop/)

### Required Knowledge
- Basic SQL
- Basic Python (PySpark helpful but not required)
- Basic understanding of data pipelines

---

## Step 1: Get the Dataset (5 minutes)

1. **Go to Kaggle:**
   - Visit: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
   - Click "Download" button (18 MB ZIP file)
   - You'll need a free Kaggle account

2. **Extract files:**
   ```bash
   unzip brazilian-ecommerce.zip
   ```

3. **Verify you have 9 CSV files:**
   - olist_customers_dataset.csv
   - olist_orders_dataset.csv
   - olist_order_items_dataset.csv
   - olist_order_payments_dataset.csv
   - olist_order_reviews_dataset.csv
   - olist_products_dataset.csv
   - olist_sellers_dataset.csv
   - olist_geolocation_dataset.csv
   - product_category_name_translation.csv

---

## Step 2: Set Up Databricks (10 minutes)

### Option A: Community Edition (Recommended for Learning)

1. **Sign up:**
   - Go to: https://community.cloud.databricks.com/login.html
   - Click "Sign up" and create free account
   - Confirm email

2. **Access workspace:**
   - Log in to your workspace
   - You'll see the Databricks interface

3. **Check your SQL Warehouse:**
   - Left sidebar → Click "SQL Warehouses"
   - You should see "Starter Warehouse" (auto-created)
   - If not started, click "Start"
   - Wait ~30 seconds for green status

### Option B: Free Trial (14 days, full features)

1. Go to: https://databricks.com/try-databricks
2. Select your cloud provider (AWS recommended)
3. Follow signup wizard
4. Create a "Serverless SQL Warehouse" when prompted

---

## Step 3: Upload Data to Databricks (10 minutes)

### Create Unity Catalog Volume

1. **Navigate to Catalog:**
   - Left sidebar → Click "Catalog"
   - You'll see a tree structure

2. **Find your catalog:**
   - Look for "workspace" or your workspace name
   - Expand it → Click "default" schema

3. **Create Volume:**
   - Click "Create" button (top right)
   - Select "Volume"
   - Name: `olist_data`
   - Volume type: External (if asked)
   - Click "Create"

4. **Upload CSV files:**
   - Click on the new `olist_data` volume
   - Click "Upload files" button
   - Select all 9 CSV files
   - Wait for upload to complete (1-2 minutes)

5. **Note the path:**
   - Your path is: `/Volumes/workspace/default/olist_data/`
   - (Replace `workspace` with your catalog name if different)

---

## Step 4: Create and Run Notebooks (45 minutes)

### Notebook 1: Environment Check (5 minutes)

1. **Create notebook:**
   - Left sidebar → "Workspace"
   - Navigate to your user folder
   - Right-click → "Create" → "Notebook"
   - Name: `00_Environment_Check`
   - Language: Python
   - Connect to: Your SQL Warehouse

2. **Paste and run this code:**

```python
# Environment Check
print("🔍 Checking Databricks Environment...\n")

# Check Spark version
print(f"✅ Spark Version: {spark.version}")

# Check catalogs
print("\n📚 Available Catalogs:")
spark.sql("SHOW CATALOGS").show()

# Check current catalog
print(f"\n📍 Current Catalog: {spark.catalog.currentCatalog()}")
print(f"📍 Current Schema: {spark.catalog.currentDatabase()}")

# Check volume and files
try:
    files = dbutils.fs.ls("/Volumes/workspace/default/olist_data/")
    csv_files = [f for f in files if f.name.endswith('.csv')]
    print(f"\n✅ Found {len(csv_files)} CSV files in volume")
    for f in csv_files:
        print(f"  📄 {f.name}")
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("Make sure you created the volume and uploaded files!")

print("\n🎉 Environment ready!")
```

3. **Click "Run All"**
   - Should see ✅ for all checks
   - If errors, review Steps 2-3

---

### Notebook 2: Bronze Layer (10 minutes)

1. **Create new notebook:** `01_Bronze_Layer_Ingestion`

2. **Copy the complete Bronze layer code from the project document** (Section 2.2)

3. **Run the notebook:**
   - Click "Run All"
   - Wait 2-3 minutes
   - Should see: "✅ Success: 9/9 tables created"

4. **Verify tables:**
```python
# Check created tables
spark.sql("SHOW TABLES IN workspace.default").filter(
    "tableName LIKE 'bronze_%'"
).show()
```

---

### Notebook 3: Silver Layer (10 minutes)

1. **Create notebook:** `02_Silver_Layer_Transformation`

2. **Copy the Silver layer code** (Section 2.3 from project document)

3. **Run and verify:**
   - Should create 5 tables: `silver_orders`, `silver_order_items`, etc.
   - Check `silver_orders_master` has ~96K rows

---

### Notebook 4: Gold Layer (10 minutes)

1. **Create notebook:** `03_Gold_Layer_Business_Metrics`

2. **Copy the Gold layer code** (Section 2.4)

3. **Run and verify:**
   - Should create 8 tables
   - Check `gold_customer_metrics` has ~96K customers

---

### Notebook 5: ML Segmentation (5 minutes)

1. **Create notebook:** `04_ML_Customer_Segmentation`

2. **Copy the ML segmentation code** (Section 2.5)

3. **Run:**
   - Training takes ~2 minutes
   - Should see segment analysis output

---

### Notebook 6: ML Churn Prediction (5 minutes)

1. **Create notebook:** `05_ML_Churn_Prediction`

2. **Copy the churn prediction code** (Section 2.6)

3. **Run:**
   - Training takes ~2-3 minutes
   - Should see AUC-ROC score (0.80-0.90 typical)

---

## Step 5: Create SQL Queries (20 minutes)

1. **Navigate to SQL Editor:**
   - Left sidebar → "SQL" → "Queries"

2. **Create these 8 queries** (copy from project docs):

   **Query 1: KPIs Overview**
   ```sql
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
   ```

   **Query 2: Revenue Trends**
   ```sql
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
   ```

   **Query 3-8:** Copy from project document Section 3.3-3.8

3. **Run each query** to verify data

---

## Step 6: Create Databricks Dashboards (30 minutes)

1. **Create visualizations:**
   - For each query, click "Add Visualization"
   - Choose appropriate chart type:
     - KPIs → Counter cards
     - Revenue trends → Line chart
     - Segments → Pie/Bar chart
     - Geographic → Map (or Bar if map unavailable)

2. **Create dashboard:**
   - Left sidebar → "Dashboards" → "Create Dashboard"
   - Name: "Olist Analytics Platform"
   - Add visualizations one by one
   - Arrange in logical layout

3. **Add filters:**
   - Add date range picker
   - Add state dropdown
   - Configure to filter all visuals

---

## Step 7: Power BI Setup (Optional, 45 minutes)

### Connect Power BI to Databricks

1. **Get connection details:**
   - In Databricks: SQL Warehouses → Your warehouse
   - Click "Connection details" tab
   - Copy:
     - Server hostname
     - HTTP path

2. **Open Power BI Desktop:**
   - Click "Get Data"
   - Search for "Databricks"
   - Select "Databricks SQL"

3. **Enter connection info:**
   - Server hostname: (paste from step 1)
   - HTTP path: (paste from step 1)
   - Data connectivity mode: DirectQuery or Import

4. **Authentication:**
   - Select "User + Password" or "Azure Active Directory"
   - Enter credentials
   - Click "Connect"

### Load Tables

5. **Select tables:**
   - Navigator will show all your tables
   - Select these Gold tables:
     - `gold_customer_metrics`
     - `gold_daily_metrics`
     - `gold_category_metrics`
     - `gold_geographic_metrics`
     - `gold_customer_churn_predictions`
   - Click "Load"

### Create Dashboards

6. **Follow Power BI guide** (Section 5-7 from PowerBI document):
   - Create DAX measures
   - Build 4 dashboard pages
   - Add visualizations
   - Apply formatting

---

## Step 8: Test & Validate (15 minutes)

### Data Quality Checks

1. **Row counts match:**
```sql
SELECT 'bronze_orders' as table_name, COUNT(*) as row_count 
FROM bronze_orders
UNION ALL
SELECT 'silver_orders', COUNT(*) FROM silver_orders
UNION ALL
SELECT 'gold_customer_metrics', COUNT(*) FROM gold_customer_metrics;
```

Expected:
- bronze_orders: ~99K
- silver_orders: ~96K (only delivered)
- gold_customer_metrics: ~96K

2. **Revenue reconciliation:**
```sql
-- Should match across layers
SELECT SUM(order_total_value) as total_revenue
FROM silver_orders_master;

SELECT SUM(monetary_value) as total_revenue
FROM gold_customer_metrics;
```

3. **Churn predictions exist:**
```sql
SELECT COUNT(*) FROM gold_customer_churn_predictions;
-- Should be ~96K customers
```

### Dashboard Checks

4. **Databricks SQL Dashboard:**
   - All visualizations load
   - Filters work
   - No errors in query logs

5. **Power BI Dashboard:**
   - Data refreshes successfully
   - All visuals render
   - Slicers filter correctly

---

## Step 9: Document & Share (30 minutes)

1. **Take screenshots:**
   - Each Databricks dashboard
   - Each Power BI page
   - Save as PNG files

2. **Create GitHub repo:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/olist-analytics.git
   git push -u origin main
   ```

3. **Add documentation:**
   - Copy README.md from this project
   - Update with your specific details
   - Add screenshots to `images/` folder

4. **Record demo video:**
   - Use Loom, QuickTime, or OBS
   - 3-5 minute walkthrough
   - Upload to YouTube
   - Add link to README

---

## Troubleshooting

### Issue: "Volume not found"

**Solution:**
- Verify catalog name (might not be "workspace")
- Check schema name (might not be "default")
- Run: `SELECT current_catalog(), current_database()`
- Update paths in notebooks accordingly

### Issue: "Permission denied"

**Solution:**
- Ensure you're the volume owner
- Check Unity Catalog permissions
- Try using your personal schema instead of default

### Issue: "SQL Warehouse won't start"

**Solution:**
- Check quota limits (Community Edition has limits)
- Wait a few minutes and retry
- Try creating a new warehouse

### Issue: "Power BI can't connect"

**Solution:**
- Verify SQL Warehouse is running (green status)
- Check firewall isn't blocking port 443
- Try "Import" mode instead of "DirectQuery"
- Ensure you copied connection details exactly

### Issue: "Tables exist but queries fail"

**Solution:**
- Check table names match exactly (case-sensitive)
- Verify catalog.schema.table format
- Run `DESCRIBE TABLE tablename` to check schema

---

## Time Estimates

| Task | Time | Difficulty |
|------|------|-----------|
| Setup Databricks | 10 min | Easy |
| Upload data | 10 min | Easy |
| Bronze layer | 10 min | Easy |
| Silver layer | 15 min | Medium |
| Gold layer | 15 min | Medium |
| ML models | 10 min | Medium |
| SQL queries | 20 min | Easy |
| Databricks dashboards | 30 min | Medium |
| Power BI setup | 45 min | Medium |
| Testing | 15 min | Easy |
| Documentation | 30 min | Easy |
| **Total** | **3-4 hours** | - |

---

## Success Criteria

By the end, you should have:
- ✅ 9 Bronze tables with raw data
- ✅ 5 Silver tables with cleaned data
- ✅ 8 Gold tables with business metrics
- ✅ 2 ML models (segmentation, churn)
- ✅ 8 SQL queries with visualizations
- ✅ 1-2 interactive dashboards
- ✅ GitHub repo with documentation
- ✅ Screenshots and demo video

---

## Next Steps

After completing the setup:

1. **Customize:**
   - Add your own business questions
   - Create additional ML models
   - Design custom visualizations

2. **Optimize:**
   - Add Z-ordering for performance
   - Implement incremental refresh
   - Set up alerting

3. **Share:**
   - Add to your portfolio
   - Present in interviews
   - Write blog post about learnings

---

## Additional Resources

- **Databricks Documentation:** https://docs.databricks.com/
- **Delta Lake Guide:** https://docs.delta.io/
- **PySpark API:** https://spark.apache.org/docs/latest/api/python/
- **Power BI Documentation:** https://docs.microsoft.com/power-bi/
- **Kaggle Dataset:** https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

---

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review error messages carefully
3. Verify all prerequisites are met
4. Check Databricks community forums
5. Open an issue on GitHub (if using this repo)

---

**Good luck! 🚀**

Remember: The goal is learning, not perfection. Don't be afraid to experiment and break things - that's how you learn best!