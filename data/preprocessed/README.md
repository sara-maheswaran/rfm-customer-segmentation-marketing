# Data Preprocessing

## Why We Cleaned the Data

Before building segmentation and predictive models, we cleaned the dataset to ensure that insights would be accurate, reliable, and meaningful.  

Raw marketing data often contains inconsistencies, extreme values, and formatting issues that can distort clustering algorithms and regression models. Proper preprocessing ensures that patterns we identify reflect real customer behavior — not data noise.

---

## Key Issues Identified

- Missing values in the `Income` column  
- Unrealistic birth years (e.g., 1893) resulting in incorrect ages  
- Extreme income outliers (e.g., $666,666)  
- Constant columns with no predictive value  
- Inconsistent categorical labels in `Education` and `Marital_Status`  
- Highly skewed spending variables  

---

## How We Addressed These Issues

### 1. Income Imputation
Missing income values were replaced with the **median**.

- The median was used because income is right-skewed.
- It prevents extreme high-income values from biasing the dataset.

---

### 2. Outlier Removal
Extreme values in **Age** and **Income** were removed using the **Interquartile Range (IQR)** method.

- This statistical method identifies unusually high or low values.
- Removing these prevents distortion in clustering and predictive models.
- This ensures segments represent realistic customer profiles.

---

### 3. Removal of Constant Columns
Columns with no variation (e.g., `Z_CostContact`, `Z_Revenue`) were removed.

- Variables with a single constant value add no predictive value.
- Removing them simplifies models and improves efficiency.

---

### 4. Categorical Standardization
`Education` and `Marital_Status` were standardized and consolidated into broader, meaningful categories.

- Reduced inconsistent formatting (e.g., capitalization differences).
- Grouped similar categories to improve interpretability.
- Reduced noise in clustering models.

---

### 5. Datetime Conversion & Tenure Creation
`Dt_Customer` was converted to a datetime format.

From this, we created:

- **Customer_Tenure_Days** – number of days since enrollment.

Tenure helps measure loyalty and lifecycle stage.

---

### 6. Feature Engineering
We created new features to better summarize customer behavior:

- **Age**
- **Total_Spending** (sum of all product categories)
- **Total_Purchases** (sum across channels)
- **Children_at_home**
- **Family_Size**

These derived features improve segmentation accuracy and strengthen predictive modeling (RFM and CLV).

---

### 7. Log Transformation of Spending
Spending variables were highly skewed.

- Applied a **log transformation** to reduce skewness.
- Prevents extreme spenders from dominating clusters.
- Helps models capture general behavioral trends more effectively.

---

## Final Outcome

After preprocessing:

- Dataset reduced from 2,240 to 2,229 high-quality observations  
- All missing values resolved  
- Extreme outliers removed  
- Behavioral and demographic features enhanced  
- Data optimized for clustering, RFM segmentation, and CLV modeling  

This preprocessing step ensured that subsequent segmentation and predictive models were based on clean, structured, and business-relevant data.
