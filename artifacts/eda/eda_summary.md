# EDA Summary — Feature Selection & Data Preparation Decisions

## 1. Leakage Prevention

- Review-related features are excluded from model training because they may contain information that would not be available at prediction time and could introduce data leakage.
- Target-derived geographic mean-delay encoding is calculated using training data only and then applied to validation, test, and future/inference data.

## 2. Order / Seller Features

### seller_count
- Values greater than 3 are capped at 3 because higher values are rare.

### unique_seller_states
- The value `3` is removed because it is rare.

## 3. Price Features

### total_price
- `total_price` is retained as the representative price feature.
- `max_price`, `mean_price`, `median_price`, and `min_price` are removed as redundant alternatives.
- Outliers are removed only from observations where `delayed = 0`.

## 4. Freight Features

### min_freight
- `min_freight` is retained as the representative freight feature.
- `max_freight`, `mean_freight`, `median_freight`, and `total_freight` are removed.
- Outliers are removed only from observations where `delayed = 0`.

## 5. Item Features

### item_count
- The value `21` is removed as an extreme/rare value.
- Values greater than 6 are capped at 6.

### unique_products
- Values greater than 4 are capped at 4.

### unique_category
- The value `6` is removed because it is rare.

## 6. Product Weight and Volume

### item_weight
- `min_item_weight` is retained.
- Other item-weight aggregation features are removed.
- Outliers are removed only from observations where `delayed = 0`.

### volume
- The minimum volume feature is retained.
- Other volume aggregation features are removed.

## 7. Photos

- Photo-related features are excluded from model training.

## 8. Payment Features

- `max_installments` is retained.
- `total_payment` is retained.
- Other payment aggregation features are removed.

## 9. Purchase Time Features

### purchase_time
- The minimum purchase timestamp is retained as the representative purchase time.

The following time-based features are created:

- `purchase_year` — year of purchase
- `purchase_month` — month of purchase
- `purchase_weekday` — weekday of purchase
- `purchase_hour` — hour of purchase
- `purchase_quarter` — quarter of purchase

## 10. Approval Time Features

### approval_time
- The minimum approval timestamp is retained as the representative approval time.

The following time-based features are created:

- `approval_year`
- `approval_quarter`
- `approval_month`
- `approval_week`
- `approval_day`
- `approval_hour`

## 11. Missing Values

- Missing values are removed from the training data where appropriate.
- Missing values are not removed from future/inference data solely for the purpose of matching the training data.
- Missing-value handling for inference must be performed using the fitted preprocessing objects from the training stage.

## 12. Delay Days

### delay_days
- Delay values greater than 70 days are capped at 70 days.
- A value of 70 is therefore treated as the maximum delay value.

## 13. Geographic Features

The following geographic categorical features are transformed using mean delay encoding:

### customer_zip_code_prefix
- Each ZIP-code prefix is replaced with its historical mean `delay_days`.

### customer_city
- Each customer city is replaced with its historical mean `delay_days`.

### customer_state
- Each customer state is replaced with its historical mean `delay_days`.

### seller_city
- Each seller city is replaced with its historical mean `delay_days`.

For all geographic features:

- The mean `delay_days` is calculated separately for each category.
- The encoding is calculated using training data only to prevent target leakage.
- Each category is represented by its corresponding historical mean delay value.
- The original categorical geographic feature is removed after encoding.
- This preserves continuous information about the historical delay level of each geographic category instead of reducing categories into discrete groups.

## 14. Geographic Coordinates and Distance

The following geographic features are retained:

- `customer_lat`
- `customer_long`
- `min_seller_distance_km`

The following features are removed:

- `max_seller_distance_km`
- `mean_seller_distance_km`
- `median_seller_distance_km`
- `seller_lat`
- `seller_long`

## 15. Final Feature Selection Principle

Feature selection is based on:

1. Avoiding data leakage.
2. Removing redundant aggregations.
3. Reducing the effect of rare/extreme values where justified by EDA.
4. Retaining representative features from highly correlated feature groups.
5. Preserving useful continuous information through mean delay encoding where appropriate.
6. Keeping only information that is available at prediction time.
7. Ensuring target-derived transformations are fitted on training data only.
8. Preparing the data so that the same feature definitions and preprocessing decisions can be reproduced in production.