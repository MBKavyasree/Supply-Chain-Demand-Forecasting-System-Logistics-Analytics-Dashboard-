# Supply Chain Data Analysis Report

## Executive Summary

This analysis covers data cleaning, exploratory analysis, demand forecasting, and dashboard design for the retail supply chain dataset. The source file contains 100 records and 33 columns with no missing values and no duplicate rows. Because the dataset does not include a real transaction or shipment date, demand forecasting was implemented as a supervised machine learning problem rather than a classical time-series model.

## Data Cleaning and Preprocessing

- Standardized all column names into machine-friendly snake_case.
- Resolved the ambiguous lead-time fields by preserving both as `order_lead_time_days` and `supplier_lead_time_days`.
- Added derived features: `inventory_gap`, `inventory_turnover_ratio`, `logistics_cost_total`, `total_lead_time_days`, `estimated_unit_margin`, and `revenue_per_unit_sold`.
- Validated that the dataset has 0 missing values and 0 duplicate rows.
- Retained all observations because no rows were structurally invalid.

## KPI Snapshot

- Total revenue: $577,604.82
- Total products sold: 46,099
- Average shipping cost: $5.55
- Average end-to-end lead time: 37.60 days
- Inspection quality score: 0.43 on a 0 to 1 scale

## Exploratory Findings

### Product Demand and Revenue

```
              total_revenue  total_products_sold  avg_stock_level  avg_inventory_gap
product_type                                                                        
skincare          241628.16                20731            40.20            -478.08
haircare          174455.39                13611            48.35            -351.97
cosmetics         161521.27                11757            58.65            -393.54
```

### Logistics Efficiency by Transportation Mode

```
                avg_shipping_cost  avg_route_cost  avg_shipping_time  avg_total_lead_time
transport_mode                                                                           
Sea                          4.97          417.82               7.12                35.35
Rail                         5.47          541.75               6.57                41.68
Road                         5.54          553.39               4.72                37.24
Air                          6.02          561.71               5.12                35.08
```

### Supplier Performance Comparison

```
            avg_products_sold  avg_revenue  avg_shipping_cost  avg_total_lead_time  avg_defect_rate  inspection_pass_rate  supplier_score
supplier                                                                                                                                 
Supplier 3             538.87      6519.73               4.79                40.27             2.47                  0.13            0.76
Supplier 1             410.37      5834.41               5.51                33.44             1.80                  0.48            0.69
Supplier 2             503.09      5703.06               5.74                39.64             2.36                  0.23            0.66
Supplier 5             481.22      6130.19               5.79                40.61             2.67                  0.17            0.57
Supplier 4             400.33      4803.83               5.76                36.11             2.34                  0.00            0.32
```

## Demand Forecasting

- Modeling target: `products_sold`
- Candidate models: Linear Regression, Random Forest Regressor, Gradient Boosting Regressor
- Selected model: random_forest
- Cross-validated R2: -0.308
- Test R2: -0.670
- Test MAE: 355.32
- Test RMSE: 399.26

### Top Forecast Drivers

```
                     feature  importance
 shipping_carriers_Carrier C        0.16
             capacity_buffer        0.09
                availability        0.07
           production_volume        0.06
     supplier_lead_time_days        0.05
        total_lead_time_days        0.05
          manufacturing_cost        0.04
                  route_cost        0.04
                 defect_rate        0.04
                 stock_level        0.04
manufacturing_lead_time_days        0.04
                       price        0.03
              location_Delhi        0.03
        order_lead_time_days        0.03
               shipping_cost        0.03
```

### Highest Forecast Demand Candidates

```
  sku product_type  predicted_products_sold  stock_level
SKU94    cosmetics                   826.46           77
SKU37     skincare                   810.22           25
 SKU9     skincare                   806.71           14
SKU36     skincare                   783.32           18
SKU78     haircare                   770.35            5
SKU47     skincare                   759.35            4
SKU11     skincare                   754.58           46
SKU74     haircare                   743.90           41
SKU46     haircare                   742.10           92
SKU91    cosmetics                   714.60           98
```

### Largest Forecast Errors to Review

```
  sku product_type  actual_products_sold  predicted_products_sold  absolute_error
SKU44    cosmetics                   919                   277.84          641.16
SKU80     skincare                   872                   249.50          622.50
SKU10     skincare                   996                   398.40          597.60
SKU45     haircare                    24                   585.58          561.58
SKU39     skincare                   176                   662.69          486.69
SKU70     haircare                    32                   502.51          470.51
SKU33    cosmetics                   616                   187.45          428.55
 SKU4     skincare                   871                   443.16          427.84
SKU76     haircare                   241                   662.69          421.69
SKU18     haircare                   620                   214.20          405.80
```

## Dashboard Design

The interactive dashboard is saved as `outputs/supply_chain_dashboard.html` and includes:

- KPI cards for total revenue, total products sold, and average lead time.
- Demand trend across the observation sequence with a rolling mean overlay.
- Revenue by product type.
- Average shipping cost by transportation mode.
- Supplier performance comparison using a composite score.
- Inventory levels for the highest-stock products.
- Forecast quality annotation with model metrics.

Note: the requested "demand trend over time" is approximated using record sequence because the source dataset has no date field. For production use, replace this with order date or shipment date.

## Business Insights

- Skincare is the strongest revenue contributor at $241,628, so assortment and promotional planning should prioritize that category.
- Sea is the lowest-cost transport mode on average, while lead-time variation across modes suggests mode selection should be segmented by urgency rather than standardized.
- Supplier 3 ranks highest on the composite supplier score, while Supplier 4 is the weakest performer; supplier allocation and corrective action plans should reflect that gap.
- SKU12 carries the highest stock position, whereas SKU9 has one of the tightest inventory gaps; inventory policy should rebalance excess stock away from slow-moving SKUs and protect constrained items.
- The demand forecast model selected random_forest with test R2=-0.67; the strongest predictive signal is shipping_carriers_Carrier C, indicating operational variables materially shape demand outcomes.
- SKU94 has the highest predicted demand under current operating conditions, making it a priority candidate for replenishment, supplier capacity checks, and logistics planning.

## Recommendations

- Rebalance inventory by reducing stock exposure on overstocked SKUs and increasing safety stock for items with tight inventory gaps and high predicted demand.
- Use supplier scorecards in quarterly business reviews, prioritizing volume allocation to higher-performing suppliers and remediation plans for low-scoring ones.
- Shift non-urgent shipments toward the lowest-cost transportation mode where service levels allow, and reserve faster modes for constrained or high-margin items.
- Add a transaction date, promised delivery date, and actual delivery date to the source data model to unlock true time-series demand forecasting and service-level analytics.
- Investigate the weakest forecast cases to determine whether product-level factors or hidden external drivers are missing from the dataset.

## Generated Outputs

- Dashboard: `outputs/supply_chain_dashboard.html`
- Clean dataset: `outputs/supply_chain_cleaned.csv`
- Forecast results: `outputs/demand_forecast_results.csv`
- Forecast metrics: `outputs/forecast_metrics.json`
- Plots: `outputs/plots/`
