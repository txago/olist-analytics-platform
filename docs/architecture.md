# 🏗️ Technical Architecture

## Overview

The Olist Analytics Platform is built on **Databricks** using the **Medallion Architecture** pattern, a best-practice approach for organizing data in a lakehouse. This architecture ensures data quality, scalability, and maintainability.

---

## Architecture Layers

### 1. Bronze Layer (Raw Data)

**Purpose:** Ingest raw data exactly as received from source systems.

**Characteristics:**
- No transformations applied
- All fields stored as strings initially
- Full audit trail with `ingestion_timestamp`
- Delta Lake format for ACID guarantees
- Unity Catalog governance

**Tables:**
| Table name | Total rows |
|--|--|
| bronze_customers | 99,441 |
| bronze_orders | 99,441 |
| bronze_order_items | 112,650 |
| bronze_order_payments | 103,886 |
| bronze_order_reviews | 99,224 |
| bronze_products | 32,951 |
| bronze_sellers | 3,095 |
| bronze_geolocation | 1,000,163 |
| bronze_product_category_translation  | 71 |

**Code Example:**
```python
df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "false") \
    .csv(f"/Volumes/workspace/default/olist_data/{file_name}")

df = df.withColumn("ingestion_timestamp", current_timestamp())

df.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable(f"workspace.default.bronze_{table_name}")
```

---

### 2. Silver Layer (Cleaned & Validated)

**Purpose:** Clean, validate, join, and enrich data for analytics.

**Transformations Applied:**
- ✅ Data type casting (strings → timestamps, decimals)
- ✅ Data quality checks (nulls, duplicates)
- ✅ Deduplication
- ✅ Table joins (orders + customers + items)
- ✅ Feature engineering (delivery_days, delivery_delay)
- ✅ English translations for categories

**Tables:**
| Table name | Output data |
|--|--|
| silver_orders | 96,478 delivered orders |
| silver_order_items | 112,650 enriched items |
| silver_order_payments | 99,441 aggregated payments |
| silver_order_reviews | 98,410 aggregated reviews |
| silver_orders_master | 96,478 complete order facts |

**Key Features:**
- **silver_orders_master** is the main fact table
- All dimensions joined (customer, payment, review, items)
- Calculated metrics added (order_total_value, delivery_days)

**Code Example:**
```python
# Filter only delivered orders
orders_clean = orders.filter(F.col("order_status") == "delivered")

# Convert timestamps
orders_clean = orders_clean.withColumn(
    "order_purchase_timestamp", 
    F.to_timestamp("order_purchase_timestamp")
)

# Calculate delivery time
orders_clean = orders_clean.withColumn(
    "delivery_days",
    F.datediff(
        F.col("order_delivered_customer_date"), 
        F.col("order_purchase_timestamp")
    )
)
```

---

### 3. Gold Layer (Business Metrics)

**Purpose:** Create aggregated, business-ready tables optimized for analytics and reporting.

**Tables:**
| Table name | Output data |
|--|--|
| gold_customer_metrics | 96,096 unique customers |
| gold_product_metrics | 32,951 products |
| gold_category_metrics | 71 categories |
| gold_geographic_metrics | 4,119 city-state combinations |
| gold_daily_metrics | 793 days |
| gold_customer_segments_ml | ML-based segments |
| gold_customer_churn_predictions | Churn risk scores |

**Business Metrics Calculated:**

#### Customer Metrics
- **RFM Scores:** Recency, Frequency, Monetary (1-5 scale)
- **Customer Segments:** Champions, Loyal, At Risk, Lost (10 segments)
- **CLV (Customer Lifetime Value):** 12-month projection
- **Customer Lifetime:** Days between first and last order
- **Average metrics:** AOV, delivery days, review score

#### Product Metrics
- **Revenue:** Total and average per product/category
- **Sales volume:** Total orders, units sold
- **Freight analysis:** Total and average shipping costs
- **Rankings:** Revenue rank within category

#### Geographic Metrics
- **State/city performance:** Orders, revenue, customers
- **Delivery performance:** Average days and delays by region
- **Customer satisfaction:** Review scores by location

#### Time-Series Metrics
- **Daily trends:** Orders, revenue, AOV
- **Moving averages:** 7-day, 30-day
- **Growth rates:** MoM, YoY comparisons

**Code Example:**
```python
customer_metrics = master.groupBy(
    "customer_id", 
    "customer_unique_id"
).agg(
    # Recency
    F.datediff(F.lit(max_date), F.max("order_purchase_timestamp"))
        .alias("recency_days"),
    
    # Frequency
    F.count("order_id").alias("frequency"),
    
    # Monetary
    F.sum("order_total_value").alias("monetary_value"),
    
    # Additional metrics
    F.avg("order_total_value").alias("avg_order_value"),
    F.avg("delivery_days").alias("avg_delivery_days"),
    F.avg("avg_review_score").alias("avg_review_score")
)
```

---

## Machine Learning Layer

### 1. Customer Segmentation (K-Means)

**Input Features:**
- `recency_days`
- `frequency`
- `monetary_value`

**Process:**
1. Feature assembly and standardization
2. K-Means clustering (k=3)
3. Segment analysis and labeling

**Output:**
- 3 distinct customer segments
- Segment characteristics (avg recency, frequency, monetary)
- Customer-to-segment mapping

**Code:**
```python
# Feature engineering pipeline
assembler = VectorAssembler(
    inputCols=features,
    outputCol="features_raw"
)
scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withStd=True,
    withMean=True
)

# K-Means clustering (k=3 segments)
kmeans = KMeans(
    k=3,
    seed=42,
    featuresCol="features",
    predictionCol="ml_segment"
)

# Create pipeline
pipeline = Pipeline(
    stages=[assembler, scaler, kmeans]
)

# Train model
model = pipeline.fit(ml_data_sampled)
predictions = model.transform(ml_data)
```

---

### 2. Churn Prediction (Random Forest)

**Target Variable:**
- `is_churned` = 1 if recency_days > 90, else 0

**Input Features:**
- `recency_days`
- `frequency`
- `monetary_value`
- `avg_order_value`
- `avg_delivery_days`
- `avg_delivery_delay`
- `avg_review_score`
- `customer_lifetime_days`
- `total_items_purchased`

**Model:**
- Random Forest Classifier
- 100 trees, max depth 5
- 80/20 train-test split

**Output:**
- Churn probability (0-1)
- Risk category (High/Medium/Low)
- Binary prediction (0/1)

**Performance:**
- AUC-ROC: 0.85+ (typical range)
- Precision/Recall balanced for business needs

**Code:**
```python
# Random Forest
rf = RandomForestClassifier(
    labelCol="is_churned",
    featuresCol="features",
    numTrees=10,
    maxDepth=5,
    seed=42
)

# Extract churn probability
predictions = predictions.withColumn(
    "churn_probability",
    extract_probability(F.col("probability"))
)

# Divide categories based in percentiles
predictions = predictions.withColumn(
    "churn_risk_category",
    F.when(F.col("churn_percentile") <= 0.20, "High Risk")      # Top 20%
     .when(F.col("churn_percentile") <= 0.50, "Medium Risk")    # Next 30%
     .otherwise("Low Risk")                                     # Bottom 50%
)
```

---

## Data Governance

### Unity Catalog Structure

```
Catalog: workspace
├── Schema: default
│   ├── Volume: olist_data (CSV files)
│   ├── Bronze Tables (9)
│   ├── Silver Tables (5)
│   └── Gold Tables (8)
```

**Benefits:**
- ✅ **Centralized metadata:** All tables discoverable
- ✅ **Access control:** Fine-grained permissions
- ✅ **Lineage tracking:** Data flow visibility
- ✅ **Audit logs:** Who accessed what, when

---

## Storage Layer: Delta Lake

**Key Features Used:**

1. **ACID Transactions**
   - Ensures data consistency during concurrent writes
   - No partial writes or corrupted data

2. **Time Travel**
   ```sql
   SELECT * FROM bronze_orders VERSION AS OF 3
   SELECT * FROM bronze_orders TIMESTAMP AS OF '2024-01-15'
   ```

3. **Schema Evolution**
   - Add new columns without breaking existing queries
   - Automatic schema inference and validation

4. **Optimized Performance**
   - Z-ordering for query performance
   - Data skipping using statistics
   - Compaction for small files

---

## Query Performance Optimization

### Serverless SQL Warehouse

**Configuration:**
- Auto-scaling: On
- Min cluster size: 1 node
- Max cluster size: 4 nodes
- Auto-suspend: 10 minutes

**Query Optimization Techniques:**

1. **Predicate Pushdown**
   ```sql
   -- Filters applied at storage level
   SELECT * FROM gold_customer_metrics 
   WHERE customer_state = 'SP'
   ```

2. **Column Pruning**
   ```sql
   -- Only read necessary columns
   SELECT customer_id, monetary_value 
   FROM gold_customer_metrics
   ```

3. **Partitioning** (for time-series data)
   ```python
   df.write \
       .format("delta") \
       .partitionBy("order_date") \
       .saveAsTable("gold_daily_metrics")
   ```

---

## Scalability Considerations

### Current Scale
- **Data volume:** ~1M rows total
- **Processing time:** <5 minutes end-to-end
- **Query response:** <2 seconds (most queries)

### Future Scale (100x growth)
- **Data volume:** 100M+ rows
- **Processing:** Distributed across multiple nodes
- **Optimization needed:**
  - Z-ordering on frequently filtered columns
  - Partitioning by date
  - Materialized views for complex aggregations
  - Liquid clustering for Delta tables

---

## Security & Compliance

### Data Masking
- PII (customer_id, customer_unique_id) are hashed
- No direct personal information exposed

### Access Control
```sql
-- Example: Grant read-only to analysts
GRANT SELECT ON SCHEMA workspace.default TO `analysts@company.com`

-- Example: Grant full access to data engineers
GRANT ALL PRIVILEGES ON SCHEMA workspace.default TO `engineers@company.com`
```

### Audit Logging
- All table access logged via Unity Catalog
- Query history tracked in SQL Warehouse
- Notebook runs logged with timestamps

---

## Monitoring & Observability

### Key Metrics Tracked

1. **Data Quality:**
   - Row counts per table
   - Null percentages
   - Duplicate rates
   - Schema drift detection

2. **Pipeline Performance:**
   - Execution time per notebook
   - Query latency
   - Cluster utilization

3. **Business Metrics:**
   - Daily revenue trends
   - Customer churn rate
   - Order volume

### Alerting (Production Setup)
- Email alerts on pipeline failures
- Slack notifications for data quality issues
- Dashboard refresh failures

---

## Technology Choices: Why Databricks?

### vs. Traditional Data Warehouse (e.g., Snowflake)
✅ **Unified platform** for data + ML (no separate tools)
✅ **Delta Lake** for ACID + time travel
✅ **Lower cost** for large-scale analytics
✅ **Native support** for Python/Spark

### vs. Cloud Data Lakes (e.g., S3 + Athena)
✅ **Built-in governance** (Unity Catalog)
✅ **Optimized performance** (caching, indexing)
✅ **Managed infrastructure** (no DevOps overhead)
✅ **Collaboration** (shared notebooks, queries)

---

## Deployment Architecture

### Development Environment
```
Databricks Workspace
├── Notebooks (Python/SQL)
├── SQL Queries
├── Dashboards
└── ML Experiments
```

### Production (Hypothetical)
```
Databricks Workspace (Production)
├── Workflows (scheduled jobs)
├── SQL Endpoints (always-on)
├── MLflow Model Registry
└── Monitoring Dashboards
```

**CI/CD Pipeline (if implemented):**
1. Code changes pushed to GitHub
2. Databricks CLI deploys notebooks
3. Automated tests run on sample data
4. Workflows triggered on schedule

---

## Cost Optimization

### Strategies Used

1. **Serverless SQL Warehouse**
   - Only pay when running queries
   - Auto-suspend after 10 minutes idle

2. **Delta Lake Compression**
   - Reduced storage costs
   - Faster query performance

3. **Efficient Data Types**
   - Use `INT` instead of `BIGINT` where possible
   - Use `DECIMAL` instead of `DOUBLE` for money

4. **Query Result Caching**
   - Identical queries return cached results
   - No re-execution needed

---

## Future Enhancements

### Architecture Improvements

1. **Streaming Ingestion**
   - Use Delta Live Tables for real-time updates
   - Kafka integration for event streaming

2. **Advanced ML**
   - Product recommendation engine
   - Dynamic pricing optimization
   - Inventory forecasting

3. **Data Quality Framework**
   - Great Expectations integration
   - Automated data validation tests
   - Anomaly detection

4. **Multi-Region Support**
   - Global customer base
   - Data residency compliance
   - Geo-distributed queries

---

## References

- [Databricks Lakehouse Architecture](https://www.databricks.com/glossary/medallion-architecture)
- [Delta Lake Documentation](https://docs.delta.io/)
- [Unity Catalog Guide](https://docs.databricks.com/data-governance/unity-catalog/index.html)
- [PySpark API Reference](https://spark.apache.org/docs/latest/api/python/)