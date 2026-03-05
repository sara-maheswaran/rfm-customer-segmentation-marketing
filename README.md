# **Customer Segmentation for Targeted Marketing Optimization** (ML8 Project)

# 📍 **Content Navigation:**
[Project Summary](#project-summary)

[Installation and Run](#installation-and-run)

[Project Model:](#project-model)

- [Dataset Description](#dataset-description)

- [Methodology](#methodology)

- [Data Cleaning](#data-cleaning)

- [Exploratory Data Analysis](#exploratory-data-analysis)

- [Model Development](#model-development)

- [Ethical Implications and Biases](#ethical-implications-and-biases)

- [Next Steps](#next-steps)

[Team Collaboration](#team-collaboration)

[Repository Structure](#repository-structure)


# **Project Summary:**
## Project Overview
This project utilizes advanced machine learning techniques to transition marketing strategies from generalized outreach to highly targeted, data-driven campaigns. The analysis involves identifying hidden customer segments through unsupervised clustering, which empowers the business to allocate marketing spend efficiently, maximize campaign ROI, and engage the right customers with the right offers.

### Business Problem
The company currently applies broad, non-personalized marketing campaigns across its entire customer base. This leads to inefficient marketing spend, poor differentiation between high-value and low-value customers, and missed upselling opportunities.

Thus, there is a critical need for data-driven segmentation to allow for tailoring of marketing strategies.

### Business Objective
To leverage customer segmentation for targeted marketing, enhancing campaign effectiveness and driving long-term retention and revenue growth.

The primary project goals are to:
- Identify distinct customer clusters based on their transactional recency, purchase frequency, and total monetary value (RFM).
- Increase campaign effectiveness and ROI by allowing for precise resource allocation and optimized marketing spend.

**Success is measured by the clarity, validity, and business applicability of these clusters, ensuring they provide actionable insights that align stakeholders and confidently guide revenue strategies.**

>### **Full project plan is outlined [here](google.com).**

# **Installation and Run:**
>- Read and follow instructions in [SETUP.md](https://github.com/sara-maheswaran/Machine_Learning_Group_8/blob/main/SETUP.md) file

###

# **Project Model**:
## Dataset Description
### Source: [Kaggle – Customer Personality Analysis Dataset](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis?resource=download)
The dataset provides insights into customer demographics, spending habits, and purchasing behavior.
### Shape
- Rows: 2,240 customers
- Columns: 29 variables

Each row represents a unique customer, and each column represents demographic information, purchasing behavior, or campaign interaction history.

### Summary
**Demographics:** Income, Year_Birth, Education, Marital_Status, Kidhome, Teenhome  
**Spending Behavior:** MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds  
**Purchase Channels:** NumWebPurchases, NumCatalogPurchases, NumStorePurchases, NumWebVisitsMonth  
**Engagement:** Recency, AcceptedCmp1–AcceptedCmp5, Response 

### Key Variables and Attributes:

<details>

<br>

| Variable Name      | Type          | Description                                                        | Category            |
|--------------------|---------------|--------------------------------------------------------------------|---------------------|
| ID                 | Integer       | Customer's unique identifier                                       | Identification      |
| Year_Birth         | Integer       | Customer's birth year                                              | Demographic          |
| Education          | Categorical   | Customer's education level                                         | Demographic          |
| Marital_Status      | Categorical   | Customer's marital status                                         | Demographic          |
| Income             | Integer       | Customer's yearly household income                                  | Demographic          |
| Kidhome            | Integer       | Number of children in customer's household                          | Demographic          |
| Teenhome           | Integer       | Number of teenagers in customer's household                         | Demographic          |
| Dt_Customer        | Date          | Date of customer's enrollment with the company                     | Temporal             |
| Recency            | Integer       | Number of days since customer's last purchase                     | Behavioral           |
| Complain           | Integer       | 1 if customer complained in the last 2 years, 0 otherwise         | Behavioral           |
| MntWines           | Integer       | Amount spent on wine in the last 2 years                          | Spending Behavior    |
| MntFruits          | Integer       | Amount spent on fruits in the last 2 years                        | Spending Behavior    |
| MntMeatProducts    | Integer       | Amount spent on meat in the last 2 years                          | Spending Behavior    |
| MntFishProducts    | Integer       | Amount spent on fish in the last 2 years                          | Spending Behavior    |
| MntSweetProducts    | Integer       | Amount spent on sweets in the last 2 years                        | Spending Behavior    |
| MntGoldProds      | Integer       | Amount spent on gold in the last 2 years                          | Spending Behavior    |
| NumDealsPurchases  | Integer       | Number of purchases made with a discount                         | Promotion Behavior   |
| AcceptedCmp1       | Integer       | 1 if customer accepted the offer in the 1st campaign, 0 otherwise | Promotion Activity   |
| AcceptedCmp2       | Integer       | 1 if customer accepted the offer in the 2nd campaign, 0 otherwise | Promotion Activity   |
| AcceptedCmp3       | Integer       | 1 if customer accepted the offer in the 3rd campaign, 0 otherwise | Promotion Activity   |
| AcceptedCmp4       | Integer       | 1 if customer accepted the offer in the 4th campaign, 0 otherwise | Promotion Activity   |
| AcceptedCmp5       | Integer       | 1 if customer accepted the offer in the 5th campaign, 0 otherwise | Promotion Activity   |
| Response           | Integer       | 1 if customer accepted the offer in the last campaign, 0 otherwise | Promotion Activity   |
| NumWebPurchases    | Integer       | Number of purchases made through the company's website             | Activity             |
| NumCatalogPurchases | Integer       | Number of purchases made using a catalogue                         | Activity             |
| NumStorePurchases   | Integer       | Number of purchases made directly in stores                       | Activity             |
| NumWebVisitsMonth   | Integer       | Number of visits to the company's website in the last month       | Activity             |

</details>

## Methodology

Analyses for this customer segmentation project were conducted using **Python 3.11.14** in Jupyter Notebooks.  

The project leveraged key libraries including **pandas**, **numpy**, **scikit-learn**, **matplotlib** and **seaborn** for data manipulation, visualization, clustering, and predictive modeling.  

Any random seeds or states used to ensure reproducibility are documented directly in the notebooks within the `model` folder. **add link**

## Data Cleaning

Purpose of Data Cleaning: Ensure accurate and actionable customer insights by addressing inconsistencies and anomalies in the marketing dataset.

Key Issues Identified:
- Missing values in the Income column: 24 records (~1.1%)
- Unrealistic birth years (e.g., 1893) leading to age outliers: 7 records (~0.3%)
- Extreme high-income outliers (e.g., $666,666): 1 record (~0.04%)
- Constant or non-informative columns: 2 columns (Z_CostContact, Z_Revenue).
- Categorical inconsistencies in Education and Marital_Status.
- Skewed spending variables: All spending variables (Wines, Fruits, Meat, Fish, Sweets, Gold) were highly right-skewed, affecting the distribution across all 2,240 records (~100%).

Data Cleaning & Preprocessing Strategy:
- Income Imputation: Missing Income values were replaced with the median because income is right-skewed.
- Outlier Removal: Extreme values in Age and Income were removed using the Interquartile Range (IQR) method.
- Drop Constant Columns: Columns with no variance were removed.
- Categorical Standardization & Consolidation: Standardized Education and Marital_Status and grouped categories into meaningful levels.
- Datetime Conversion: Dt_Customer was converted to datetime and used to calculate Customer_Tenure_Days.
- Feature Engineering: Created Age, Total_Spending, Total_Purchases, Children_at_home, Family_Size.
- Log Transformation of Spending: Spending variables were log-transformed to reduce skew.

## Exploratory Data Analysis

- Skewed distributions: Apply scaling or transformation for numeric variables like income and purchase amounts.
- No multicollinearity: Features are largely independent, simplifying modeling.
- Customer behavior insights:
    - High spending in wine and meat
    - Fewer children correlates with higher purchases
    - In-store purchases dominate
    - Majority of customers are married and hold a graduation-level education
    - Average customer age is approximately 57 years

>**For more insights, see [Exploratory Data Analysis README.md](https://github.com/sara-maheswaran/Machine_Learning_Group_8/tree/main/data/raw).**

## Model Development

### Customer Segmentation (RFM KMeans Clustering)

Below is the model development process for segmenting customers using RFM (Recency, Frequency, Monetary) KMeans clustering. The goal is to group customers based on their purchasing behavior to make informed marketing strategies and improve customer engagement.

### Model Objectives & Success Criteria

**Objectives:**
- Segment customers into meaningful groups using their purchasing behavior.
- Identify high value and loyal customers for targeted campaigns.
- Provide actionable insights to improve retention and sales.

**Success Criteria:**
- Optimal number of clusters determined via **Elbow Method** and **Silhouette Score**.
- Distinct and interpretable customer segments with business relevance.

### Relevant Features for Training
- Recency : Days since last purchase
- Frequency  : Total number of purchases
- Monetary  : Total spending amount

These features capture customer engagement, loyalty, and value essential for RFM based segmentation.

### Algorithm Selection

**KMeans Clustering** was chosen because:

- The task is unsupervised (no labeled output).
- RFM features are numerical and suitable for distance-based clustering.
- KMeans is computationally efficient and widely used for customer segmentation.

### Hyperparameter Tuning & Validation
- Evaluated K values from 2 to 10 using:
    - Elbow Method: Looked for the “elbow” point in the Inertia curve.
    - Silhouette Score: Measured cluster cohesion and separation.
- Result:
    - Optimal K = 4 provides clear, distinct customer segments.

![Elbow and Silhoutte Score](images/elbow_and_silhoutte_score.png)

### Initialization
- Number of initializations (n_init): Tested multiple runs to avoid local minima.
- Random state: Set for reproducibility.

### Validation Techniques
- Silhouette Score : measures how similar an object is to its own cluster vs other clusters.

![Silhouette Score](images/silhoutte_k_4.png)

- Business interpretation : Segments evaluated for actionable insights
    - Cluster 0 : New / Occasional
    - Cluster 1 : Potential Loyals
    - Cluster 2 : Loyal Customers
    - Cluster 3 : At Risk/Low Value

![PCA](images/customer_segments_visualized_in_2D.png)

### Key Results
- Customers are segmented into 4 distinct clusters based on recency, frequency, and monetary value.
- New / Occasional, Potential Loyals, Loyal Customers and At Risk/Low Value customer segments are easily identifiable.
- The model provides a foundation for targeted marketing, personalized campaigns, and loyalty strategies

### Cluster Profiling
There are four distinct customer segments which can be used to move from broad marketing campaigns to data-driven, personalized strategies:

| Cluster | Segment Type | No. of Customers | Profile Characteristics | Marketing Strategy |
| ---- | ---- | ---- | ---- |  ---- |
| 0 | New / Occasional | 524 | Low recency, low frequency, low monetary | Engagement: Incentivize 2nd purchase, cross-sell, and small bundle offers |
| 1 | Potential Loyals | 616 | High recency, high frequency, high monetary | Re-engagement: "We miss you" emails, loyalty points, and win-back discounts |
| 2 | Loyal Customers | 573 | Low recency, high frequency, high monetary | Retention: Exclusive VIP offers, early access, and premium campaigns |
| 3 | At-Risk / Low Value | 516 | High recency, low frequency, low monetary |	Reactivation: Low-cost reminders; suppress from high-cost marketing if inactive |

### Cluster Spending
The total purchaces and spending of each cluster is:

| Cluster |	Segment |	Customers | Total Revenue ($) | Avg_Revenue per Customer ($) | Avg_Purchases (Qty.) |	Revenue Share (%) |
| --- |	--- | --- | --- | --- |	--- | --- |
| 1 | Loyal Customers |	573 |	14142.042267 | 24.680702 | 20.727749 | 33.454533 |
| 0 | At Risk/Low Value | 516 |	6432.098454 | 12.465307 |	8.482558 | 15.215826 |
| 2	| New / Occasional | 524 | 5967.641575 | 11.388629 | 7.519084 | 14.117103 |
| 3	| Potential Loyals | 616 | 15730.641844 | 25.536756 | 21.021104 | 37.212538 |

## Ethical Implications and Biases
>Possible ethical implications and biases are described in the notebook [here](https://github.com/sara-maheswaran/Machine_Learning_Group_8/blob/main/reports/Ethical%20Implications_Biases%20rpt.ipynb) under [Reports](reports). 

## Next Steps
- First Step: share segments with marketing team; tag customers in Customer Relationship Manager (CRM)
- Next Campaign: design one test campaign per segment, measure response lift
- Quarterly: track segment migration - i.e. if customers are moving between groups
- Annual: re-run clustering with new data to validate and update segments

# **Team Collaboration:**
>The team’s collaboration methods are described in the notebook [here](https://github.com/sara-maheswaran/Machine_Learning_Group_8/blob/main/reports/Team%20Collaboration%20rpt.ipynb) under [Reports](reports). 

### Project Showcase Pitch Deck
>- See [Reports](reports) folder

### Individual Reflection Videos - **placeholder links to update*
- [Shaifali Tailor](http://google.com)
- [Iris Jiongco](http://google.com) 
- [Saraneya Maheswaran](http://google.com)
- [David Ancor](http://google.com)
- [Anika Chowdhury](http://google.com)
- [Stella Hoang](http://google.com)



# **Repository Structure:**

```
├── data
    └── preprocessed
    └── raw
├── experiments
├── images
├── models
├── reports
├── .gitignore
├── README.md
├── SETUP.md
├── pyproject.toml
├── uv.lock
```

- **data:** Includes raw data and Exploratory Data Analysis, and preprocessed data.
- **experiments:** Data experiments contributed by all team members.
- **images:** Includes charts, graphs, tables from analysis.
- **models:** Project models.
- **reports:** Project reports.
- **src:** Source code, databases, logs.
- **.gitignore:** Files to exclude.
- **README.md:** This file.
- **SETUP.md:** Contains the steps required to set up this repo for the module.
- **pyproject.toml:** Tells Python which packages this repo needs to run.
- **uv.lock:** Project environment.