# **Project Model**: Customer Segmentation for Targeted Marketing Optimization

Data Science Institute, UofT - Cohort 8 - ML Team 8


## Table of Contents


## Project Overview
This project utilizes advanced machine learning techniques to transition marketing strategies from generalized outreach to highly targeted, data-driven campaigns. The analysis is split into two core objectives: identifying hidden customer segments through unsupervised clustering, and building predictive classification models to determine a customer’s likelihood to accept future promotions. Ultimately, this approach empowers the business to allocate marketing spend efficiently, maximize campaign ROI, and engage the right customers with the right offers.

## Business Problem
The company currently applies broad, non-personalized marketing campaigns across its entire customer base. This leads to:

- Inefficient marketing spend
    * Utilizing a generalized strategy increases marketing spend by wasting budget on demographic segments with a near-zero probability of conversion
- Low campaign conversion rates
    * One-size-fits-all approaches for campaigns results in ignored promotions and stagnant sales throughput, despite high-volume outreach
- Poor differentiation between high-value and low-value customers
    * Without customer segmentation, customers who are highly profitable are treated the same as one-off buyers, thus leaving revenue streams open to competitor offers 
- Missed upselling and retention opportunities
    * High-probability cross-sell opportunities are missed and instead the company waits for customers to churn before trying to save them

There is a critical need for data-driven segmentation and predictive modeling to understand customer behavior, anticipate their actions, and tailor marketing strategies accordingly.

## Stakeholders
The primary audience is:
- Marketing Managers
    * Customer segmentation will allow marketing managers to know who their most profitable customers are and how to reach them efficiently with ads tailored to their purchasing habits
- CRM / Customer Experience Teams
    * Loyalty programs and automated emails with tailored campaigns can be developed to decrease recency and build re-engagement where needed
- Business Strategy Executives
    * Overall company business strategies can be driven based on customer purchasing behaviour
- Sales Leadership
    * Insights into where customers make purchases can guide where investments can be made to increase sales - ex. store expansions vs e-commerce infrastructure 

Secondary audience:
- Data team / Analytics team (for implementation)
    * Results from clustering and the models created can be used to update other tools used by the company as required (ex. Salesforce), modify to process larger datasets, or retrain the model when new data is available

## Business Objective
To leverage customer segmentation for targeted marketing, enhancing campaign effectiveness and driving long-term retention and revenue growth.

The primary project goals are to:
- Identify distinct customer clusters based on purchasing behavior, demographics, and engagement metrics.
- Increase campaign effectiveness and ROI by allowing for precise resource allocation and optimized marketing spend.

**Success is measured by the clarity, validity, and business applicability of these clusters, ensuring they provide actionable insights that align stakeholders and confidently guide revenue strategies.**

## Risks and Unknowns

### Risks

<details>
**1. Poor Cluster Interpretability:** Clusters may not be clearly distinct or actionable.

>Mitigation plan: Use validation metrics, limit cluster count, perform clear profiling, and validate with business stakeholders. Integrate visualization techniques to intuitively show cluster separations to stakeholders.

**2. Data Quality Issues:** Missing income values, outliers (extreme spending), and/or inaccurate customer demographics.
>Mitigation plan: Clean data systematically (imputation, outlier treatment), conduct EDA, and document preprocessing steps. Employ automated monitoring systems to track data quality and report anomalies in real-time.

**3. Over-Segmentation:** Too many clusters may confuse marketing teams and/or complicate execution.
>Mitigation plan: Select optimal cluster number using statistical methods and align with business execution capacity. Use automated tools to assist in determining the ideal number of clusters (e.g. elbow method).

**4. Static Segmentation:** Customer behavior changes over time.
>Mitigation plan: Retrain models periodically and incorporate recency-based features to capture behavioral changes. Implement dynamic segmentation that adapts to real-time data, ensuring ongoing relevance and accuracy.

**5. External Factors:** Economic downturn, seasonality, product changes may impact behavior, consumer preferences.
>Mitigation plan: Monitor model performance over time and reassess assumptions during economic or seasonal shifts. Users should create contingency plans and flexible strategies based on external factors, ensuring resilience and adaptability.

</details>

### Unknowns
<details>
- Customer Lifetime Value (CLV): *Uncertainty regarding the true CLV of segments, impacting marketing priorities.*
- Customer Satisfaction: *Unknown satisfaction levels, influencing retention and campaign effectiveness.*
- Competitor Influence: *Uncertainty about how competitor actions may affect customer behavior.*
- Macro-Economic Conditions: *Potential impact of economic factors on purchasing behavior and marketing success.*
- Variability in Customer Response: *Unclear how different segments will respond to targeted marketing strategies.*

- Resource Availability: *Uncertain levels of resources for implementing targeted strategies.*
</details>

***
# Repository Structure - **to update*

```
├── data
    └── preprocessed
    └── raw
    └── sql
├── experiments
├── models
├── reports
├── src
├── .gitignore
├── README.md
```

### Requirements:
- Specific libraries/frameworks suited to project requirements: [pyproject.toml](pyproject.toml) file

### Installation and Run - **placeholder link to update*
- See [SETUP.md](SETUP.md) file

***
## Dataset Description
### Source: [Kaggle – Customer Personality Analysis Dataset](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis?resource=download)
The dataset contains customer demographic information, purchasing behavior, and campaign response history.
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

Any random seeds or states used to ensure reproducibility are documented directly in the notebooks within the `model` folder.

### Exploratory Data Analysis

#### Demographics
The majority of customers are married and hold a graduation-level education. The average customer age is approximately 57 years. A small number of unrealistic age outliers were identified and will need to be handled. Understanding age distribution helps determine how purchasing patterns vary across life stages.

![alt text](image.png)

---

#### Income Analysis
The average income is approximately 51,381, but income distribution is highly skewed due to a small number of extremely high-income customers. This creates imbalance and requires scaling and outlier treatment before segmentation. Income is expected to be one of the strongest predictors of spending behavior.

![alt text](image-1.png)

---

#### Spending Behavior
Customers spend the most on wines and meat products compared to other categories. Overall spending patterns are not evenly distributed, meaning some customers spend significantly more than others. This variation is valuable for segmentation, as it helps distinguish high-value customers from low-spending ones.

![alt text](image-2.png)

---

#### Household Structure and Purchasing
Spending behavior changes depending on whether customers have children at home. Households with different family sizes show differences in total purchases and product category preferences. This suggests that family structure is an important segmentation factor.

![alt text](image-3.png)
![alt text](image-4.png)

---

#### Purchase Channels
Customers most frequently make purchases in physical stores, followed by web and catalog channels. This indicates that offline presence remains important, although digital channels still play a significant role. Understanding channel preference allows for better campaign targeting.

![alt text](image-6.png)

---

#### Age Group Spending Patterns
Spending habits vary across age groups. Certain age segments spend more overall, while others show stronger preferences for specific product categories. Segmenting by age group can therefore improve marketing personalization and promotional targeting.

![alt text](image-5.png)

---

#### Correlation Insights
Spending categories are strongly related to each other, meaning customers who spend heavily in one category often spend heavily in others. Income, recency of purchase, and total spending appear to be particularly important variables for identifying meaningful customer segments.

![alt text](image-7.png)

---

#### Key Insights for Segmentation
The analysis shows clear differences in income levels, spending behavior, product preferences, and purchasing channels. Wines and meat products dominate overall spending, and store purchases are the most common transaction method. Age, income, and household structure strongly influence buying behavior. Before building segmentation models, income gaps must be filled, outliers must be treated, and features must be scaled appropriately.

---

### Data Cleaning
- **Purpose of Data Cleaning:** Ensure accurate and actionable customer insights by addressing inconsistencies and anomalies in the marketing dataset.

- **Key Issues Identified:**
  - Missing values in the `Income` column: 24 records (~1.1%)
  - Unrealistic birth years (e.g., 1893) leading to age outliers:  7 records (~0.3%) 
  - Extreme high-income outliers (e.g., $666,666): 1 record (~0.04%)
  - Constant or non-informative columns: 2 columns (`Z_CostContact`, `Z_Revenue`).
  - Categorical inconsistencies in `Education` and `Marital_Status`.
  - Skewed spending variables: All spending variables (Wines, Fruits, Meat, Fish, Sweets, Gold) were highly right-skewed, affecting the distribution across all 2,240 records (~100%).


- **Data Cleaning & Preprocessing Strategy:**
  - **Income Imputation:** Missing `Income` values were replaced with the median because income is right-skewed.
  - **Outlier Removal:** Extreme values in `Age` and `Income` were removed using the Interquartile Range (IQR) method.  
  - **Drop Constant Columns:** Columns with no variance were removed.  
  - **Categorical Standardization & Consolidation:** Standardized `Education` and `Marital_Status` and grouped categories into meaningful levels.  
  - **Datetime Conversion:** `Dt_Customer` was converted to datetime and used to calculate `Customer_Tenure_Days`.  
  - **Feature Engineering:** Created `Age`, `Total_Spending`, `Total_Purchases`, `Children_at_home`, `Family_Size`.  
  - **Log Transformation of Spending:** Spending variables were log-transformed to reduce skew.  

For more details, see [Data Preprocessing](data/preprocessed/README.md).
   

This is documentation of the machine learning pipeline and model architecture for future reference:

#### Model Description

This project uses K-Means Clustering, an unsupervised machine learning algorithm.
Unlike classification models (which predict categories like “Yes” or “No”), clustering finds natural groupings in the data without predefined labels.


We built a customer segmentation model to group customers based on three key behavioral metrics: **Recency** (how recently they purchased), **Frequency** (how often they purchase), and **Monetary value** (how much they spend). This approach is known as **RFM Analysis**, a widely used marketing framework that helps businesses identify high-value, loyal, at-risk, and low-engagement customers. By focusing on real purchasing behavior rather than assumptions, the model provides meaningful and actionable customer groups.

Before clustering, all numerical variables were scaled using **StandardScaler** to ensure fair comparison between features, since spending values can otherwise dominate the analysis. A **pipeline** was built so that preprocessing and clustering occur together, ensuring consistency, stability, and reproducibility of results.

To determine the optimal number of customer segments, we evaluated multiple cluster sizes using the **Elbow Method** (which measures cluster compactness through inertia) and the **Silhouette Score** (which measures how clearly separated the clusters are). Both techniques indicated that **four clusters (k = 4)** provided the best balance between compactness and separation.

To make the results easier to interpret visually, we applied **Principal Component Analysis (PCA)** to reduce the data to two dimensions for plotting purposes. This step does not change the clustering results; it simply allows us to clearly see the separation between customer groups.

We selected **K-Means clustering** because it works exceptionally well with numeric, distance-based RFM data. It efficiently identifies natural behavioral patterns, scales well to large datasets, and produces segments that are easy for business teams to understand and act on. The final output delivers clear, interpretable customer segments that directly support targeted marketing and optimization strategies.

## Results | Final Products

### Key Findings and Instructions

This project produced a validated customer segmentation model that grouped customers into **four clear behavioral segments** based on how recently they purchase, how often they buy, and how much they spend. The model was tested using both the **Elbow Method** and **Silhouette Score**, which confirmed that four segments provide the best balance between clarity and accuracy. The final deliverables include the cleaned dataset, RFM feature framework, clustering model, validation analysis, PCA visualization, and executive-ready materials.  

![alt text](image-8.png) 
![alt text](image-9.png)

The key finding is that customers naturally fall into four meaningful groups: high-value customers, loyal repeat buyers, at-risk customers who may stop purchasing, and low-engagement customers with growth potential. These segments are easy to understand and can be directly used by the marketing team to create targeted campaigns instead of sending the same message to everyone. To implement this, customers should be tagged by segment in the CRM system, campaigns should be tailored to each group, and performance should be tracked regularly to measure ROI and monitor customer movement between segments.

#### Risks, Unknowns and Limitations
There are some limitations to consider. The model is based only on purchasing behavior and does not include demographic or external factors such as seasonality or economic changes. It also groups customers based on past behavior rather than predicting future actions. For this reason, the model should be refreshed periodically to stay relevant.

#### Next Steps
Next steps include integrating the segmentation into marketing systems, launching segment-specific campaigns, tracking results through A/B testing, and periodically retraining the model as new data becomes available. With proper implementation, this segmentation provides a strong foundation for personalized marketing, improved retention, and long-term revenue growth.


### Visuals and Credits

![alt text](image-10.png)

<br>

![alt text](image-11.png)

<br>

![alt text](image-12.png)


### Models
- See [Models](models/) folder

#### Ethical Implications / Biases:
<input type="checkbox">
<label>Ethical implications or biases associated with our machine learning model</label><br>


Although the model uses only purchasing behavior (Recency, Frequency, and Monetary value), it may still introduce bias because it relies on historical data. Customers who already spend more may continue receiving better offers, while lower-spending customers may receive fewer benefits, potentially widening engagement gaps. The model also reflects past marketing strategies and does not account for personal or economic circumstances.

Customer data must be handled securely and transparently, and the segmentation should be used to improve customer experience — not restrict opportunities. Regular monitoring and human oversight are important to ensure fair and responsible use.

### Project Showcase Pitch Deck
- See [Reports](reports/) folder

## Team Roles & Responsibilities
**Please edit as necessary
- **Stella Hoang** – project documentation and reporting (README), model experimentation and evaluation, GitHub repository management
- **Shaifali Tailor** – technical lead, EDA, model development and evaluation (customer segmentation, clustering, RFM analysis, CLV), FINAL model design, README contributor
- **Saraneya Maheswaran** – project management/task coordination, GitHub repository management, project documentation and reporting (README), model experimentation and evaluation
- **Iris Jiongco** – data preprocessing, README contributor, model experimentation and evaluation
- **David Ancor** –  model development and evaluation (customer segmentation, clustering, RFM analysis, CLV), slide deck preparation, team meeting organization
- **Anika Chowdhury** – model development and evaluation (clustering, customer segmentation), code/peer review and quality assurance



### Individual Reflection Videos - **placeholder links to update*

- [Stella Hoang](http://google.com)
- [Shaifali Tailor](http://google.com)
- [Saraneya Maheswaran](http://google.com)
- [Iris Jiongco](http://google.com)
- [David Ancor](http://google.com)
- [Anika Chowdhury](http://google.com)




