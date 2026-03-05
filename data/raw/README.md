# Exploratory Data Analysis on raw

## Dataset Overview
- **Total records:** 2,240 customers
- **Total features:** 29 columns
- **Duplicates:** None
- **Missing values:** Only the Income column has 24 missing entries, All other variables are complete.
     - Income will require imputation during data cleaning.

## Data Types
- Education, Marital_Status, and Dt_Customer are **string** datatype.
- Dt_Customer represents dates and should be converted to **datetime** in preprocessing.


**Demographics**

The majority of customers are married and hold a graduation-level education. The average customer age is approximately 57 years. A small number of unrealistic age outliers were identified and will need to be handled. Understanding age distribution helps determine how purchasing patterns vary across low-spending ones.

![Demographics](../../images/age_distribution_with_mean_and_outlier_bounds.png)

**Income Analysis**

The average income is approximately 51,381; however, income distribution is highly skewed due to a small number of extremely high-income customers. This creates imbalance and requires scaling and outlier treatment before segmentation. Income is expected to be one of the strongest predictors of spending behavior.

![Income_Analysis](../../images/income_distribution_with_mean_and_outlier_bounds.png)

**Spending Behavior**

Customers spend the most on wines and meat products compared to other categories. Overall spending patterns are not evenly distributed, meaning some customers spend significantly more than others. This variation is valuable for segmentation, as it helps distinguish high-value customers from low-spending ones.

![Spending_Behavior](../../images/total_spending_by_product_category.png)

**Household Structure and Purchasing**

Spending behavior changes depending on whether customers have children at home. Households with significant family sizes show differences in important segmentation and product category preferences. This suggests that family structure is an important segmentation factor.

![Household](../../images/percentage_stacked_bar_plot_product_vs_children_at_home.png)

**Purchase Channels**

Customers most frequently make purchases in physical stores, followed by web and catalog channels. This indicates that offline presence remains important and can still play a significant role. Understanding channel preference allows for better campaign targeting.

![Purchase_channels](../../images/purchases_by_channel.png)

**Age Group Spending Patterns**

Spending habits vary across age groups. Certain age segments spend more overall, while others show stronger preferences for specific product categories. Segmenting by age group can therefore improve marketing personalization and promotional targeting.

![Age_Group](../../images/ave_spending_by_age_group.png)

**Correlation Insights**

Spending categories are strongly related to each other, meaning customers who spend heavily in one category often spend heavily in others. Income, recency of purchase, and age group can serve to be particularly important variables for identifying meaningful customer segments.

### **Key Insights for Segmentation**

The analysis shows clear differences in income levels, spending behavior, product preferences, and purchasing channels. Wines and meat products dominate overall spending, and store purchases are the most common transaction method. Age, income, and household structure strongly influence buying behavior. Before building segmentation models, income gaps must be filled, outliers must be treated, and features must be scaled appropriately.

![Allfeatureshistogram](https://github.com/sara-maheswaran/Machine_Learning_Group_8/blob/main/images/all_features_histogram.png?raw=true)

---

### Categorial Variables
- **Education** (5 categories):
Most common: **Graduation** - 1,127 customers (~50%)

- **Marital Status** (8 categories):
Most common: **Married** - 864 customers

![Countedmaritalstatus](https://github.com/sara-maheswaran/Machine_Learning_Group_8/blob/main/images/count_education_marital_status.png?raw=true)

### Numerical Variables
- **Age(Year_Birth):**
     - Mean age: 57.19
     - Some outliers present

![Agedistmeanoutliers](https://github.com/sara-maheswaran/Machine_Learning_Group_8/blob/main/images/age_distribution_with_mean_and_outlier_bounds.png?raw=true)

- **Income Distribution:**
     - Mean: 51,381
     - Max: 666,666
     - Strong right skew due to high-income outliers
     - Descriptive statistics show high skewness and kurtosis

![Incomedistmeanoutliers](https://github.com/sara-maheswaran/Machine_Learning_Group_8/blob/main/images/income_distribution_with_mean_and_outlier_bounds.png?raw=true)

- **Outliers & Scaling:**
     - Boxplots and Distribution charts highlight outliers in Income and Age
     - Histograms indicate that Spending, Income, and purchase-related variables are highly skewed with large maximum values --> scaling may be required

## Columns to Drop
- Z_CostContact (constant = 3)
- Z_revenue (constant = 11)
Both provide no predictive value and should be removed during preprocessing

![Columnstodrop](https://github.com/sara-maheswaran/Machine_Learning_Group_8/blob/main/images/columns_to_drop.png?raw=true)

## Correlation Analysis
- Heatmap shows **no strong correlations**, indicating no multicollinearity among variables

![Correlation](https://github.com/sara-maheswaran/Machine_Learning_Group_8/blob/main/images/correlation_heatmap.png?raw=true)

## Customer Spending Behaviour

- Spending by Product: 
     - Customers spend the most on **wine** and **meat

![Totalspendproductcat](https://github.com/sara-maheswaran/Machine_Learning_Group_8/blob/main/images/total_spending_by_product_category.png?raw=true)

- Purchases vs Children at Home: 
     - Customers with **no children** purchase more
     - Purchases decrease as the number of children increases

![avgtotalpurchbykidhome](https://github.com/sara-maheswaran/Machine_Learning_Group_8/blob/main/images/ave_total_purchase_by_kidhome.png?raw=true)

- Purchasing Channel:
     - Majority of customers purchase **in-store**, followed by **web** and **catalog**

![purchasebychannel](https://github.com/sara-maheswaran/Machine_Learning_Group_8/blob/main/images/purchases_by_channel.png?raw=true)


