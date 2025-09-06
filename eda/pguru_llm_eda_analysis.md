## Property Price Prediction: Comprehensive EDA Summary and Recommendations

This report provides a detailed analysis of the provided property dataset, focusing on insights relevant to price prediction modeling.

### 1. Data Quality Assessment

**Missing Values:**

*   **Critical Missing Values:**
    *   `Property Type`: 40.09% missing. This is a significant amount and will require careful handling. Properties with missing `Property Type` are likely to be excluded or imputed with a majority class if appropriate, but imputation might skew results.
    *   `Tenure`: 5.73% missing. While not as critical as `Property Type`, this still needs addressing.
*   **Minor Missing Values:**
    *   `Bedrooms`: 1.80% missing.
    *   `Price`: 0.92% missing. Crucial for our target variable.
    *   `Location`: 0.66% missing.
    *   `Size`: 0.95% missing.
    *   `Bathrooms`: 0.95% missing.
*   **No Missing Values:** All amenity-related features (binary indicators) have no missing values, which is excellent.

**Missing Value Patterns and Recommendations:**

*   **Observed Patterns:**
    *   The properties with missing `Property Type` also tend to have missing values in `Price`, `Location`, `Size`, `Bedrooms`, and `Bathrooms`. This suggests a potential issue with data collection for a specific subset of properties. Row 2 in the sample data perfectly illustrates this, showing `NaN` across multiple key features.
    *   There doesn't appear to be a systematic missingness pattern for `Tenure` or the core numerical features (`Price`, `Location`, `Size`, `Bedrooms`, `Bathrooms`) beyond the cluster identified with missing `Property Type`.
*   **Recommendations:**
    1.  **`Property Type`:** Due to the high percentage of missing values, direct imputation might introduce significant bias.
        *   **Primary Recommendation:** **Remove properties with missing `Property Type`**. This is the safest approach to avoid distorting the model, especially if the missingness is not random. If removal drastically reduces the dataset size, consider creating a new category like "Unknown Property Type" and encoding it.
        *   **Secondary (if removal is too aggressive):** Impute with the mode or a predictive model if a strong relationship can be established with other features (e.g., based on size, number of rooms). However, this should be done with caution.
    2.  **`Price`:** Since `Price` is the target variable, any row with a missing `Price` cannot be used for training. These rows should be excluded from the training set. They *could* be used for predicting prices if their features are complete, but for initial model development, focusing on complete target data is best.
    3.  **`Location`, `Size`, `Bedrooms`, `Bathrooms`:** For the few missing values in these core features, consider imputation:
        *   **`Location`:** Impute with the mode (most frequent location) or "Unknown Location".
        *   **`Size`, `Bedrooms`, `Bathrooms`:** Impute using median imputation, as these are numerical features and medians are robust to outliers. Alternatively, consider K-Nearest Neighbors (KNN) imputation based on similar properties.
    4.  **`Tenure`:** Impute with the mode ("Leasehold" or "Freehold" depending on frequency) or create an "Unknown Tenure" category.

**Data Type Issues:**

*   **All data types appear to be correctly inferred:** `Price`, `Size`, `Bedrooms`, `Bathrooms` are `float64`, categorical features (`Location`, `Property Type`, `Tenure`) are `object`, and amenities are `int64` (binary). No immediate data type conversion issues are apparent.

**Potential Data Quality Problems:**

*   **Outliers:** The `Price` feature has a very wide range (43,000 to 35,000,000), suggesting potential outliers or a highly skewed distribution. Similarly, `Size` can also have outliers. This will need to be investigated and handled during feature engineering and modeling.
*   **Inconsistent `Location` data:** With 94 unique values for `Location`, there's a high chance of minor variations in spelling or naming conventions (e.g., "Cheras" vs. "Cheras Baru"). Data cleaning and standardization of location names might be necessary.
*   **Redundant Amenities:** It's possible that some amenities are highly correlated or represent similar concepts (e.g., "Covered car park" and "Multi-storey car park" might overlap in functionality or perception). This can be checked with correlation analysis.

### 2. Key Insights for Pricing

**Price Distribution Characteristics:**

*   **Highly Skewed:** The wide range of `Price` (from 43,000 to 35,000,000) with a mean of 1.4M and a standard deviation of 2.6M strongly indicates a **right-skewed distribution**. This means there are a few very expensive properties pulling the mean up, while most properties are likely clustered at lower price points. A histogram or box plot would confirm this visually.
*   **Outliers Present:** The large standard deviation relative to the mean confirms the presence of significant outliers, which will influence model performance if not handled.

**Most Influential Features on Property Price:**

Based on common real estate principles and the features provided, the most influential features are likely to be:

1.  **`Size` (Square Footage):** Larger properties generally command higher prices. This is a fundamental driver.
2.  **`Bedrooms` & `Bathrooms`:** More bedrooms and bathrooms typically correlate with larger properties and thus higher prices. The `Bedrooms` and `Bathrooms` ratio might also be informative.
3.  **`Location`:** This is often the most critical factor. Premiums are associated with desirable and convenient locations. The sheer number of unique locations (94) suggests significant price variation across geographical areas.
4.  **`Property Type`:** Different property types (e.g., landed houses, apartments, condos) have inherent price differences based on exclusivity, maintenance, and amenities.

**Facility Amenities that Command Premiums:**

While the prevalence of most amenities is low (indicated by low mean and 25th percentile values of 0 for most), those that are present are likely to add value. Features with a higher presence or that are typically associated with luxury or convenience will likely command premiums:

*   **High Impact Amenities (Likely to command premiums):**
    *   **`Swimming pool`:** A common indicator of premium living.
    *   **`Gym`:** Essential for modern lifestyles and health-conscious buyers.
    *   **`Badminton hall` / `Basketball court` / `Tennis courts`:** Sports facilities often attract a premium, especially in developments catering to families or active individuals.
    *   **`Clubhouse` / `Lounge` / `Sky lounge`:** Offer communal spaces and a sense of community/luxury.
    *   **`24 hours security` / `Perimeter fencing`:** Crucial for safety and peace of mind, often standard in higher-end properties.
    *   **`Covered car park` / `Multi-storey car park`:** Convenience and protection for vehicles are valued.
    *   **`Jacuzzi` / `Sauna`:** Indicate luxury and wellness features.
*   **Lower Impact Amenities (Potentially less premium, or baseline):**
    *   Amenities like `Bus stop`, `Cafes`, `Eateries`, `Retail stores` can increase desirability due to convenience but might not directly add a significant monetary premium compared to core property features or high-end leisure facilities.
    *   `Community garden`, `Landscaped garden`, `Recreation lake` contribute to aesthetics and well-being but their direct price impact can vary.

**Location and Tenure Impact:**

*   **`Location`:** As highlighted, `Location` is expected to be a primary driver of price. Further analysis would involve grouping properties by location and comparing their average prices, or using location as a categorical feature in a model. Some locations will undoubtedly be significantly more expensive than others.
*   **`Tenure`:** With only three categories (`Freehold`, `Leasehold`), there will likely be a price difference. `Freehold` properties are generally more valuable than `Leasehold` properties, assuming equal lease remaining duration for the latter. This difference needs to be quantified.

### 3. Feature Engineering Recommendations

**Suggested Transformations:**

1.  **Log Transformation of Target Variable (`Price`):** Due to the highly skewed `Price` distribution, applying a log transformation (e.g., `np.log1p(Price)`) to the target variable is highly recommended. This will normalize the distribution, making it more suitable for many regression algorithms and improving model performance. Remember to inverse-transform the predictions back to the original scale.
2.  **Log Transformation of `Size`:** Similar to `Price`, `Size` might also benefit from a log transformation to handle its potentially skewed distribution and reduce the impact of very large properties.
3.  **Feature Scaling (Standardization/Normalization):** For algorithms sensitive to feature scales (e.g., Linear Regression, SVM, Neural Networks), scaling is crucial.
    *   **StandardScaler:** Scales features to have zero mean and unit variance. This is generally preferred if your data has outliers.
    *   **MinMaxScaler:** Scales features to a specific range, typically [0, 1]. This can be useful for algorithms like neural networks.

**Interaction Terms to Create:**

1.  **`Size` per `Bedroom` / `Bathroom`:** Create features like `Size_per_Bedroom` (`Size / Bedrooms`) and `Size_per_Bathroom` (`Size / Bathrooms`). This can capture the efficiency of space utilization. For example, a 1000 sq ft apartment with 3 bedrooms might be priced differently than a 1000 sq ft apartment with 2 bedrooms. Handle division by zero if `Bedrooms` or `Bathrooms` can be 0.
2.  **Facility Counts:** Sum up the number of amenities for each property. This can create a `Total_Amenities` feature.
3.  **Premium Facility Counts:** Create a subset of "premium" amenities (e.g., pool, gym, courts, clubhouse) and sum them up as `Premium_Amenities_Count`.
4.  **Interactions between Core Features and Location:** If specific amenities are more valued in certain locations, interaction terms could be explored (e.g., `Pool_x_Prestigious_Location`). This is more advanced and might require feature selection first.
5.  **Tenure x Property Type:** Some property types might be more likely to be freehold.

**Categorical Encoding Strategies:**

1.  **`Location`:**
    *   **Target Encoding (Mean Encoding):** This is highly recommended for high-cardinality categorical features like `Location`. It replaces each category with the average `Price` (or log-transformed `Price`) for that location. This can capture the price-predictive power of location effectively. Be cautious of overfitting by using cross-validation within the encoding process.
    *   **Frequency Encoding:** Replace categories with their frequency in the dataset. Less powerful than target encoding for capturing price relationships.
    *   **One-Hot Encoding:** Not recommended for 94 unique locations due to creating a very high-dimensional sparse feature space, which can lead to the "curse of dimensionality" and overfitting.
2.  **`Property Type`:**
    *   **One-Hot Encoding:** With 13 unique values, one-hot encoding is a viable option. It will create 12 new binary columns.
    *   **Target Encoding:** Could also be used here, especially if there are significant price differences between types.
3.  **`Tenure`:**
    *   **One-Hot Encoding:** With only 3 categories, one-hot encoding is suitable.

**Outlier Handling Approaches:**

1.  **Winsorizing:** Cap extreme values at a certain percentile (e.g., 95th or 99th percentile for `Price` and `Size`). This reduces the impact of outliers without removing data.
2.  **Log Transformation:** As mentioned, log transformation can significantly compress the range of values, naturally reducing the influence of extreme outliers.
3.  **Removal (Last Resort):** Only remove data points identified as extreme outliers if they are clearly erroneous or have a disproportionately negative impact on model training. This should be done with caution as it reduces the dataset size.
4.  **Robust Models:** Use models that are inherently less sensitive to outliers (e.g., Tree-based models like Random Forest or Gradient Boosting Machines).

### 4. Modeling Strategy

**Recommended Algorithms for Price Prediction:**

Given the dataset's characteristics (mix of numerical and categorical features, potential for non-linear relationships, and the need to handle outliers and interactions), the following algorithms are recommended:

1.  **Gradient Boosting Machines (GBMs):**
    *   **XGBoost, LightGBM, CatBoost:** These are powerful and widely used algorithms for tabular data. They handle non-linear relationships, feature interactions, and are generally robust to outliers. LightGBM is known for its speed. CatBoost can directly handle categorical features, which might simplify the pipeline.
2.  **Random Forest:** An ensemble of decision trees that is robust to outliers and can capture complex relationships.
3.  **Linear Regression with Regularization (Ridge/Lasso):** A good baseline model. Use with feature engineering (log transformations, interaction terms) and appropriate encoding/scaling. Lasso can also perform feature selection.
4.  **Support Vector Regression (SVR):** Can handle non-linear relationships using kernels, but can be computationally expensive and sensitive to feature scaling and hyperparameter tuning.
5.  **Neural Networks (MLPs):** Can learn complex patterns but require more data and careful tuning, especially for feature engineering and scaling.

**Validation Approach Suggestions:**

1.  **K-Fold Cross-Validation:** This is the standard and most recommended approach. Split the data into 'k' folds. Train the model on k-1 folds and validate on the remaining fold. Repeat this k times, using each fold as the validation set once. Average the performance metrics across all folds. A common choice is k=5 or k=10.
2.  **Time Series Split (if temporal component exists):** If there's an implicit or explicit time aspect to the data (e.g., listing date), a time-series split might be appropriate to ensure the model is evaluated on data from the future. However, based on the current description, this seems unlikely.
3.  **Stratified K-Fold (for Target Variable):** If the `Price` distribution is highly skewed, using `StratifiedKFold` on the target variable's bins can ensure that each fold has a similar distribution of prices, leading to more reliable validation.

**Potential Pitfalls to Avoid:**

1.  **Data Leakage:**
    *   **Target Encoding:** Ensure target encoding is done *within* each fold of cross-validation, not on the entire dataset before splitting. Otherwise, information from the target variable will leak into the features.
    *   **Scaling/Imputation:** Apply scaling and imputation transformers *after* splitting the data into training and testing sets (fit on training, transform on both).
2.  **Overfitting:**
    *   **High-Cardinality Categorical Features:** Over-reliance on one-hot encoding for `Location` will lead to overfitting.
    *   **Too Many Features:** With 43 features (and potential engineered ones), the model might learn noise. Feature selection (e.g., using Lasso, feature importance from tree models) or regularization techniques are crucial.
    *   **Complex Models without Validation:** Using complex models without proper cross-validation can lead to models that perform well on the training data but poorly on unseen data.
3.  **Ignoring Data Quality Issues:** Proceeding with modeling without addressing the significant missing values in `Property Type` or the potential outliers in `Price` will result in a flawed and inaccurate model.
4.  **Incorrect Inverse Transformation:** Forgetting to inverse-transform log-transformed predictions can lead to misinterpreting model performance.
5.  **Ignoring Feature Interactions:** Not creating interaction terms for features like `Size_per_Bedroom` might miss important predictive relationships.

This comprehensive analysis and recommendations should provide a strong foundation for building a robust and accurate property price prediction model. The next steps should involve implementing the recommended data cleaning, feature engineering, and model selection processes.