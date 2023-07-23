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
We conducted some analysis for checking if there are any missing data, duplicates, incorrect datatype and strange data value in our dataset. Based on the dataset we have [Descriptive Statistic](https://github.com/zerobase-one/Pengabdi-FinPro/blob/1d34687c670307119564ab8319ad5f524cf875bc/Stage%201/Descriptive%20Statistic.ipynb), there are no missing value and duplicates in our dataset. Hence there are few interesting columns such as, EmployeeCount and StandardHours that show 1 unique value only. While for 'EmployeeNumber', we assumed that the number refers to the employee ID. The only reason why it has a higher number than the total current employees is that the missing IDs are those employees who no longer work at the company. 

### Univariate Analysis
After doing descriptive and checking, we have to do analyze each column separately. Looking at the distribution value's detail. First step defines attrition as a target, got any visualization for each column in our dataset, make a box plot for each feature and then conclude the results of observations. For the result we can access [here](https://github.com/zerobase-one/Pengabdi-FinPro/blob/76fddd6d1fb99002a4c480a19aaee0b64f6c15c0/Stage%201/Univariate%20Analysis.ipynb).
We have an unimodal data distribution including (Age, DistanceFromHome, MonthlyIncome, NumCompaniessWorked, YearsSinceLastPromotion, TotalWorkingYears, PercentSalaryHike, YearsAtCompany), a bimodal distribution including (PerformanceRating, HourlyRate, MonthlyRate, TrainingTimesLastYear, YearsInCurrentRole, YearsWithCurrManager), a multimodal including (Education, Environment Satisfaction, JobInvolvement, JobLevel, JobSatisfaction, RelationshipSatisfaction, StockOptionLevel, WorkLifeBalance). 
There are also types of distribution based on their skewness in our dataset, such as Normal(Age) and Positively Skewed(DistanceFromHome, MonthlyIncome, NumCompaniessWorked, PercentSalaryHike, YearsSinceLastPromotion, TotalWorkingYears, YearsAtCompany). We have some outliers in the MonthlyIncome, NumCompaniesWorked, PerformanceRating, StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear, YearsAtCompany, YearsSinceLastPromotion & YearsWithCurrManager features.

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

