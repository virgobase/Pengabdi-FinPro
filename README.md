# Pengabdi-FinPro
Rakamin final projects repository of Team Pengabdi FinPro 

## STAGE 0 
On this stage, we basically prepare the framework of this project. We have selected the Employee Attrition dataset beforehand, which we downloaded from Kaggle [Employee Attrition](https://www.kaggle.com/datasets/patelprashant/employee-attrition)  or you can check on the folder list above.

### Problem
The problem we want to solve is the attrition rate of the company that reached 16.1%. While based on Maier (2015) attrition rate of a company considered to be high if it surpasses 5% within one period (yearly). So, the attrition rate from our dataset is considered high and that becomes our main problem.

### Role
We act as a team of HC Data Scientist who are given order to intervene and analyze the causes of high attrition rates in our company. 

### Goal
Our main goal is to decrease the attrition rate up to 12% within 12 months with structured business recommendation programmes

### Objectives
Here are some objectives to reach our goal:
- Identify factors that cause atrition (features importance)
- Develop predictive models that can estimate the level of attrition of individuals or groups of employees
- Intervening on the causal factors that lead to high employee attrition
- Designing program recommendations for company based on model results

### Business Metrics
Our business metric is Attrition rate. 
***

## STAGE 1
Stage 1 is Exploratory Data Analysis or EDA. In this stage we explore data from descriptive statistic into visualization for a better data and business understanding.

### Descriptive Statistic
We conducted some analysis for checking if there are any missing data, duplicates, incorrect datatype and strange data value in our dataset. Based on the dataset we have [Descriptive Statistic](https://github.com/zerobase-one/Pengabdi-FinPro/blob/1d34687c670307119564ab8319ad5f524cf875bc/Stage%201/Descriptive%20Statistic.ipynb), there are no missing value and duplicates in our dataset. 
Hence there are few interesting columns such as, EmployeeCount and StandardHours that show 1 unique value only. While for 'EmployeeNumber', we assumed that the number refers to the employee ID. The only reason why it has a higher number than the total current employees is that the missing IDs are those employees who no longer work at the company. 

### Univariate Analysis
After doing descriptive and checking, we have to do analyze each column separately. Looking at the distribution value's detail. First step defines attrition as a target, got any visualization for each column in our dataset, make a box plot for each feature and then conclude the results of observations. For the result we can access [here](https://github.com/zerobase-one/Pengabdi-FinPro/blob/76fddd6d1fb99002a4c480a19aaee0b64f6c15c0/Stage%201/Univariate%20Analysis.ipynb).
- We have an unimodal data distribution including (Age, DistanceFromHome, MonthlyIncome, NumCompaniessWorked, YearsSinceLastPromotion, TotalWorkingYears, PercentSalaryHike, YearsAtCompany), a bimodal distribution including (PerformanceRating, HourlyRate, MonthlyRate, TrainingTimesLastYear, YearsInCurrentRole, YearsWithCurrManager), a multimodal including (Education, Environment Satisfaction, JobInvolvement, JobLevel, JobSatisfaction, RelationshipSatisfaction, StockOptionLevel, WorkLifeBalance). 
- There are also types of distribution based on their skewness in our dataset, such as Normal(Age) and Positively Skewed(DistanceFromHome, MonthlyIncome, NumCompaniessWorked, PercentSalaryHike, YearsSinceLastPromotion, TotalWorkingYears, YearsAtCompany).
- We have some outliers in the MonthlyIncome, NumCompaniesWorked, PerformanceRating, StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear, YearsAtCompany, YearsSinceLastPromotion & YearsWithCurrManager features.

Then we come up with few recommendations related to the analysis for pre-processing stage:
- Need to handle features which categorized as positively skewed distribution using logtransformation
- Need to handle features that having outliers using Z-score > 3 after doing the transformation

### Multivariate Analysis
#### Feature Selection Phase ####
At this stage, we used the Chi-Square test to see the significance of the categorical type features to the 'Attrition' target. Meanwhile for numerical features we use Heatmap Correlation. For more visually pleasing you can [check here.](https://github.com/zerobase-one/Pengabdi-FinPro/blob/main/Stage%201/Multivariate%20Analysis.ipynb)
By finishing this step we could finally decide which categorical and numerical features will be used in the next stage.

Here are the list of categorical features:
- BusinessTravel
- Department
- EducationField
- MaritalStatus
- JobRole
- OverTime

While down below are numerical features that will be processed:
- Age
- DailyRate
- DistanceFromHome
- EnvironmentSatisfaction
- JobInvolvement
- JobLevel
- JobSatisfaction
- MonthlyIncome
- RelationshipSatisfaction
- StockOptionLevel
- TotalWorkingYears
- TrainingTimesLastYear
- WorkLifeBalance
- YearsAtCompany
- YearsInCurrentRole
- YearsWithCurrManager

### Insights and Business Recommendations
In this phase, we focus on presenting the effect of Low Monthly Income on Attrition and the effect of employees' satisfaction on job, environment and relationship on their decision of attrition.

#### What we have done on MonthlyIncome feature? #### 
1. We modified this feature and classify into 3 classes (low, medium, high) based on its quartile.
2. We checked its Attrition ratio for each classes
3. We checked its relation with JobRole
4. We modified Age feature into Age_Cat and checked its relation with 'Low' MonthlyIncome, JobRole and Attrition
5. We checked how the ratio of Attrition based on Age_Cat
6. Lastly, we concluded some recommendations

#### What we have done on Satisfaction features? ####
1. Firstly, we checked the Attrition's ratio on JobSatisfaction, EnvironmentSatisfaction and RelationshipSatisfaction features
2. Then, we grouped those features and checked mean value of all features on Attrition
3. Lastly, we concluded some recommendations

For more visual pleasing works you can check it [here](https://github.com/zerobase-one/Pengabdi-FinPro/blob/main/Stage%201/Stage%201%20-%20Pengabdi%20FinPro.pdf)

***
# STAGE 2
Down below are brief explanation of how this stage be handled in jupyter notebook and you can [check here](https://github.com/zerobase-one/Pengabdi-FinPro/blob/main/Stage%202/Stage%202%20-%20Pengabdi%20FinPro%20(1).ipynb) for the python code.

## Data Cleansing & Extracting Features

**Handling Missing & Duplicated Data**
There is no missing and duplicated data. 

**Extracting Features**
- First of all, we dropped some features that only have 1 unique value like,'EmployeeNumber','Over18','StandardHours' or we assumed has no relevancy towards target 'Attrition' like 'EmployeeCount'.

- Furthermore, we have added new features based on the available features. The first feature is 'Accumulated_Satisfaction' which is obtained from the average of 'Environment Satisfaction', 'JobInvolvement', 'JobSatisfaction', 'RelationshipSatisfaction', and 'WorkLifeBalance' features. This feature is an average value, accumulated employee satisfaction from various aspects. Based on the heatmap correlation, it is known that the new feature 'Accumulated_Satisfaction' has a higher correlation with Attrition (target) than the five features above. So that the next five features are dropped.
  
- The second extraction feature is 'YearsWorkingPerCompany' which is obtained from the 'TotalWorkingYears' divided by 'NumCompaniesWorked'. Because data with the value “inf” was found in this new feature, row data was deleted in the 'NumCompaniesWorked' feature which has a value of 0 (zero) and in the 'TotalWorkingYears' and 'YearsAtCompany' features which have different row values between the two features. Furthermore, it is known that this new feature has a positively skewed distribution, which will be transformed later for this feature.
So that the dimensions of the dataframe after extraction and drop are made up of 1273 rows & 28 columns. This feature is considered important because it can simultaneously show employee loyalty to the company. The greater the employee's value on this feature, the more loyal the employee is to his/her job and company.

**Features Transformation**
Log1p Transformation is performed for features with a positive skewed distribution so that they are close to or normally distributed, namely: 'DistanceFromHome', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsWithCurrManager', 'PercentSalaryHike', 'YearsWorkingPerCompany'.

**Handle Outliers**
The Z-Score is calculated for each feature, then the rows of data that have a Z-score > 3 (Outlier) will be removed. The deleted data rows are as many as 15 rows.

**Encode Categorical Features**
One-Hot Encoded for the following categorical features : 'BusinessTravel', 'Department', 'EducationField', 'MaritalStatus', 'JobRole'. It was found that the number of features formed increased to 46 features. In addition, for the Overtime and Gender features, the value is changed from boolean to binary 1/0.

**Split & Scalling Data**
Then, we pplit Data Train and Data Test. After that, a scaler 'fit.transform' is performed on the Data Train while a scaler 'transform' on the Data Test.

**Handle Imbalance Class**
Handle imbalance target (Attrition) on Data Train by using oversampling technique.

## Feature Selection
After the data cleansing has been completed, feature filtering will then be carried out in the training data because there are still 46 features and 1 target which causes the data to become quite complex.

Based on the results of the correlation between features with the correlation of each feature with the target (Attrition), features that have a correlation value <= 0.07 will be dropped. Thus, 28 features are obtained that are correlated and relevant to the target which will then be used in the machine learning model.
These features are:
1. Age
2. DistanceFromHome
3. JobLevel
4. MonthlyIncome
5. OverTime
6. StockOptionLevel
7. TotalWorkingYears
8. TrainingTimesLastYear
9. YearsAtCompany
10. YearsInCurrentRole
11. YearsWithCurrManager
12. Accumulated_Satisfaction
13. YearsWorkingPerCompany
14. BusinessTravel_Non-Travel
15. BusinessTravel_Travel_Frequently
16. BusinessTravel_Travel_Rarely
17. Department_Research & Development
18. Department_Sales
19. EducationField_Medical
20. MaritalStatus_Divorced
21. MaritalStatus_Married
22. MaritalStatus_Single
23. JobRole_Healthcare Representative
24. JobRole_Laboratory Technician
25. JobRole_Manager
26. JobRole_Manufacturing Director
27. JobRole_Research Director
28. JobRole_Sales Representative 

## Feature Recommendations
These features below are recommendations of new feature that are no exist in the dataset and might have correlation whether directly or indirectly towards target 'Attrition':
1. **PerformanceFeedback**: Frequency and quality of feedback received by the employee.
2. **JobStability**: The stability or likelihood of future job changes.
3. **CareerGrowthOpportunities**: The perceived potential for career advancement within the company.
4. **CompanyCulture**: Employee perceptions of the overall company culture and values.
5. **EmployeeEngagement**: The level of employee commitment and emotional attachment to the company.

***
