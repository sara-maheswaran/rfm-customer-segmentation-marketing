# **Project Model**: Machine_Learning_Group_8

## Project Title
Customer Segmentation for Targeted Marketing Optimization


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

### Unknowns
- Customer Lifetime Value (CLV): *Uncertainty regarding the true CLV of segments, impacting marketing priorities.*
- Customer Satisfaction: *Unknown satisfaction levels, influencing retention and campaign effectiveness.*
- Competitor Influence: *Uncertainty about how competitor actions may affect customer behavior.*
- Macro-Economic Conditions: *Potential impact of economic factors on purchasing behavior and marketing success.*
- Variability in Customer Response: *Unclear how different segments will respond to targeted marketing strategies.*

- Resource Availability: *Uncertain levels of resources for implementing targeted strategies.*
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

### Requirements - **placeholder link to update*
- Specific libraries/frameworks suited to project requirements: [pyproject.toml](http://google.com) file

### Installation and Run - **placeholder link to update*
- See [SETUP.md](http://google.com) file

***
# **Project Model**: Customer Segmentation for Targeted Marketing Optimization
## Dataset Description
### Source: [Kaggle – Customer Personality Analysis Dataset](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis?resource=download)
The dataset contains customer demographic information, purchasing behavior, and campaign response history.
### Shape
- Rows: 2,240 customers
- Columns: 29 variables

Each row represents a unique customer, and each column represents demographic information, purchasing behavior, or campaign interaction history.

### Key Variables and Attributes:
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

***
# <code style="background:white;color:black">* WEEK 2 COMING SOON: * WIP BELOW</code>

## Methodology

### Data Cleaning


<input type="checkbox">
<label>Why we cleaned our data, and the best strategy </label><br>

- **Purpose of Data Cleaning:** Ensure accurate and actionable customer insights by addressing inconsistencies and anomalies in the marketing dataset.

- **Key Issues Identified:**
  - Missing values in the `Income` column.
  - Unrealistic birth years (e.g., 1893) leading to age outliers.
  - Extreme high-income outliers (e.g., $666,666).
  - Constant or non-informative columns (`Z_CostContact`, `Z_Revenue`).
  - Categorical inconsistencies in `Education` and `Marital_Status`.
  - Skewed spending variables.

- **Data Cleaning & Preprocessing Strategy (Justified):**
  - **Income Imputation:** Missing `Income` values were replaced with the median.  
    *Justification:* Median is robust to outliers and preserves central tendency without being skewed by extreme values.
  - **Outlier Removal:** Extreme values in `Age` and `Income` were removed using the Interquartile Range (IQR) method.  
    *Justification:* IQR is a standard and effective technique to detect extreme values, preventing them from disproportionately influencing clustering and predictive models.
  - **Drop Constant Columns:** Columns with no variance were removed.  
    *Justification:* Columns with a single constant value provide no predictive power, so removing them simplifies models and improves efficiency.
  - **Categorical Standardization & Consolidation:** Standardized `Education` and `Marital_Status` and grouped categories into meaningful levels.  
    *Justification:* Reduces noise from inconsistent labeling and small categories, improving clustering accuracy and interpretability.
  - **Datetime Conversion:** `Dt_Customer` was converted to datetime and used to calculate `Customer_Tenure_Days`.  
    *Justification:* Enables calculation of tenure-related features, which are important for understanding customer loyalty and lifecycle stage.
  - **Feature Engineering:** Created `Age`, `Total_Spending`, `Total_Purchases`, `Children_at_home`, `Family_Size`.  
    *Justification:* Derived features summarize behavior and demographics, improving segmentation, RFM analysis, and CLV modeling.
  - **Log Transformation of Spending:** Spending variables were log-transformed to reduce skew.  
    *Justification:* Reduces the influence of extreme spenders, allowing clustering and regression models to better capture general patterns.

-### Exploratory Data Analysis

<input type="checkbox">
<label>How we explored the relationships between different variables </label><br>
<input type="checkbox">
<label>Types of patterns/trends in our data </label><br>

### Model Development

This is documentation of the machine learning pipeline and model architecture for future reference:

<input type="checkbox">
<label>Specific objectives and success criteria for our machine learning model</label><br>
<input type="checkbox">
<label>Relevant features for training</label><br>
<input type="checkbox">
<label>Machine learning algorithms that were suitable for our problem domain</label><br>
<input type="checkbox">
<label>Techniques used to validate and tune the hyperparameters</label><br>
<input type="checkbox">
<label>How data was split into training, validation, and test sets</label><br>

## Results | Final Products

### Models
- See [Models](models) folder

#### Ethical Implications / Biases:
<input type="checkbox">
<label>Ethical implications or biases associated with our machine learning model</label><br>

### Project Showcase Pitch Deck
- See [Reports](reports) folder

### Individual Reflection Videos - **placeholder links to update*
- [Shaifali Tailor](http://google.com)
- [Iris Jiongco](http://google.com) 
- [Saraneya Maheswaran](http://google.com)
- [David Ancor](http://google.com)
- [Anika Chowdhury](http://google.com)
- [Stella Hoang](http://google.com)