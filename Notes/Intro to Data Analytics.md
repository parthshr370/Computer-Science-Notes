# Intro to Data Analytics (IDA)
#### By Parth Sharma DSE A

## Data Analysis

1. **Data analytics** is the process of *manipulating data to extract useful trends* and hidden patterns which can help us derive valuable insights to make business predictions.
2. Types of **Data analytics**
  - **Predictive** - uses *historical data to predict future trends*. For example, you might use predictive analytics to forecast your sales for the next quarter, or to predict how many customers are likely to churn.
  - **Descriptive** - *It tells you what happened in the past*. For example, you might use descriptive analytics to track your sales over time, or to see how many visitors came to your website.
  - **Prescriptive** - *Recommend actions that you can take to achieve your desired* outcomes. For example, if your predictive analytics model shows that your sales are likely to decline in the next quarter, prescriptive analytics might recommend that you launch a new marketing campaign or offer discounts to existing customers.
  - **Diagnostic** - Answers the question of why something happened. For example, if you see that your sales have declined in the past month, you might use diagnostic analytics to identify the root cause of the decline.

3. **Data Collection** - Process of *collecting the info from relevant sources in order to find a solution to the given statistical enquiry*. Collection of data is the first and the foremost step in a statistical investigation.
4. **Statistical enquiry** - Search of truth by using s*tatistical methods of collection , compiling, analysis, interpretation* etc.
   - Investigator 
   - Enumerator 
   - Respondents
   - Survey

5. Data is a tool that helps the investigator in understanding the problem by providing him with the information required . It has two types
   - *Primary* and *Secondary*

6. Methods of collecting secondary data
  - Published sources 
  - Govt publication 
  - Semi govt Publication ( *Municipalities and metropolitan councils* )
  - Publication of Trade Associations ( *Sugar mills association about diff sugar mills*)

7. **Data preprocessing** is the process of *transforming raw data into a format that is suitable for data analysis*. It is an important step in any data analytics project, as it ensures that the data is accurate, complete, and consistent.
8. Process involved in Data preprocessing 
- **Data cleaning:** This involves identifying and correcting errors or inconsistencies in the data, such as missing values, outliers, and duplicates.
- **Data integration:** This involves combining data from multiple sources into a single dataset.
- **Data transformation:** This involves converting the data into a desired format, such as changing the data types or scaling the data.
- **Feature engineering:** This involves creating new features from the existing data, or transforming existing features in a way that makes them more useful for the intended analysis.

9. **Missing Data** - It can be cured in some ways 
  - Ignore the tuples - do this when the *data set is very large and multiple values* are missing within a tuple 
  - Fill the missing value - You can do this manually or by *most probable value* 

10. **Noisy Data** - it is meaningless data that cant be interpreted by machines. It can be generated due to faulty data collection .
11. Ways to handle **Noisy Data**
  - **Binning Method -** Works on a *sorted data in order to smooth it* . Whole data is divided into segments of equal size and then various methods are employed to complete the task
  - **Regression -** Data can be made *smooth by fitting in to a regression function* . It can be linear or multiple(Multiple independent variables)
  ![Reg](https://miro.medium.com/v2/resize:fit:1400/1*EGGSComC0XLWPGq_oSHV-g.jpeg)

  - **Clustering** is collecting data in a a cluster 
![Clus](https://media.geeksforgeeks.org/wp-content/uploads/20190318035145/Drawing1.png)


12. **Data Transformation** - the process of converting raw data into a format that is suitable for analysis. This may involve cleaning the data to remove errors and inconsistencies, integrating data from multiple sources, or converting the data to a different format.
13. Steps for data transformation
  - **Normalisation** - Scaling data value into certain scale 
  - **Attribute selection** - New attributes are constructed from the given set of attributes to help the mining process 
  - **Discretization -** Done to replace the raw values of numeric attributes by interval level or conceptual level 
  - **Concept hierarchy** - Attributes are converted into lower level hierarchy to higher level ( Eg city can be converted to a country)

14. **Data reduction** is crucial step in the data mining process that involves reducing the size of the dataset while preserving the important information 
15. Some steps for **data reduction**
  - Feature Selection
  - Feature Extraction
  - Sampling 
  - Clustering 
  - Compression

16. Common **data formats -** 
  - **CSV** - Comma seperated values where each line is a data record 
  - **XLSX** - MS excel open XML format for spreadsheet 
  - **ZIP** - used at data containers they store one or more than one file in compressed form 
  - **TXT** - Used for storing plane text (`.txt`)
  - **JSON** - Javascript object notation Used for representing structured data based on javascript objects 
  - **PDF** is portable document file 

## Data Cleaning

17. Data cleaning is process of identifying and correcting inaccurate records from a set table or data base and refers to identifying incomplete incorrect or irrelevant parts of the data
18. **Makes your data efficient** . where an uncleaned data can give bad results on a good algorithm . On the other hand a high quality data can cause simple algorithm to give you outstanding results. 
19. There are many different types of *data cleaning tasks*, including:
  - **Identifying and correcting errors:** This could include fixing typos, correcting spelling mistakes, and correcting incorrect dates.
  - **Removing duplicate data:** This is important because duplicate data can skew the results of data analysis.
  - **Formatting data consistently:** This could involve converting all dates to the same format, or ensuring that all addresses are in the same format.
  - **Filling in missing values:** This could involve using statistical methods to estimate missing values, or simply removing records with missing values.
  - **Standardising data:** This could involve converting all measurements to the same units, or converting all names to the same format.

20. **Benefits of data cleaning**
There are many benefits to data cleaning, including:
  - **Improved data quality:** Data cleaning helps to ensure that your data is accurate, complete, and consistent. This leads to more reliable and accurate results from data analysis.
  - **Reduced costs:** Data cleaning can help to reduce the costs associated with storing and processing data. Dirty data can take up more storage space and require more processing power.
  - **Improved decision-making:** By improving the quality of your data, data cleaning can help you to make better decisions.
  - **Increased customer satisfaction:** Data cleaning can help you to better understand your customers and their needs. This can lead to improved customer satisfaction and increased sales.

## EDA (Exploratory Data Analysis)

21. **Exploratory data analysis (EDA)** is a statistical process of investigating *data sets and summarizing their main characteristics*, often using statistical graphics and other data visualization methods.
22. It is often the first step n data analytics
23. There are *many different techniques that can be used in EDA*, including:
  - **Univariate analysis:** This involves *examining the distribution of each individual variable in the data set*. This can be done using *histograms*, *boxplots*, and other statistical graphics.
  - **Bivariate analysis:** This involves *examining the relationship between two variables in the data set*. This can be done using scatter plots, correlation matrices, and other statistical graphics.
  - **Multivariate analysis:** This involves *examining the relationship between three or more variables in the data set*. This can be done using principal component analysis, cluster analysis, and other statistical techniques.

24. Here are some **examples of how EDA can be used:**
  - A marketing analyst might use *EDA to understand the demographics of their customer base*, identify trends in customer behaviour, and develop targeted marketing campaigns.
  - A financial analyst might use EDA to *identify patterns in stock prices, predict market trends*, and make investment decisions.
  - A medical researcher might use EDA to *identify risk factors for diseases, develop new treatments*, and improve patient care.

25. Steps of ***EDA*** 
  - *Data collection* 
  - *Data cleaning* 
  - *Univariate analysis*
  - *Bivariate Analysis*
  - *Multivariate Analysis*

26. **Univariate analysis** uni means one . Where one variable is used for analysing the dataset. It deals with how the data for one quantity is changed.
27. Like finding the pattern of the height of students in schools across India
28. In *Univariate* analysis you're generally looking for 
  - **Measures of Central Tendency:** *Mean, median, and mode* provide insights into where the centre of the data lies.
  - **Measures of Dispersion:** *Range, variance, and standard deviation* help you understand how spread out the data is.
  - **Frequency Distribution:** *Creating histograms, bar charts, and pie charts* allows you to visualise the data’s distribution.

29. **Bivariate analysis -** The examination of two variables simultaneously.Where two variables are observed and one variable here is independent while the other is dependant 
30. It helps us to understand trends and pattern 
31. Tells us about cause effect relationship
32. It helps researchers make predictions and allows researchers to predict future results by modelling link between two variables 
33. It helps in decision making 
34. Types of **Bivariate Analysis -**
  - Scatterplots
  - Correlation
  - Regression

35. **Chi square test -**  is a statistical hypothesis test used to compare observed results with expected results. It is a non-parametric test, meaning that it does not make any assumptions about the distribution of the data.
36. **T - test** is a statistical test that compares the means of two groups to see if they have a big difference. 
37. The analysis is appropriate when comparing the averages of two categories in a variable 
38. **ANOVA** test decides whether averages of two groups differ from one another statistically . This comparision of averages of numerical variable for more than two categories of a categorical variable is appropriate.
39. **Multivariate analysis**  offers all possible independent variables and their relationship with one another.
40. It measures the effect of multiple independent variables on two or more dependent variables.
41. Like level of education can tell us about **life satisfaction** and **job satisfaction**
42. Statistics plays a crucial role in data analytics, which is the process of *examining, cleaning, transforming, and interpreting data* to discover valuable *insights, patterns,* and *trends*.
43. **Hypothesis testing** is a form of statistical inference that uses data from a sample to draw conclusions about a *population parameter or a population probability.*
44. Types of **hypothesis testing -**
   - Null Hypothesis 
   - Alternate Hypothesis
   - Level of significance
   - P Value

45. ***Steps in Hypothesis testing***
   - **Step 1–** We first *identify the problem* about which we want to make an assumption keeping in mind that our assumption should be contradictory to one another  
   - **Step 2 –** We consider *statically assumption such that the data is normal or not*, statistical independence between the data. 
   - **Step 3 –** We decide our *test data on which we will check* our hypothesis 
   - **Step 4 –** The data for the tests are evaluated in this step we look for various *scores in this step like z-score and mean values.*
   - **Step 5 –** In this stage, we decide *where we should accept the null hypothesis or reject the null hypothesis*

![Hypothesis](https://miro.medium.com/v2/resize:fit:1400/1*1cNvVqKvO9XWiTZ-sj2pkA.png)

STUDY ABOUT BASIC STATS 
## Chi Square Test

46. The chi-square test is a *non-parametric statistical test* used to compare observed results with expected results. It is used to determine whether there is a statistically significant relationship between two categorical variables, or whether the observed distribution of a categorical variable matches the expected distribution.
47. One way to use this test is to find the footfall of a restaurant during lunch time. The number of people you expected vs the ones who actually gave the footfall can be solved by chi square test 

![chi](https://cdn1.byjus.com/wp-content/uploads/2020/10/Chi-Square-Test.png)

48. This method is to find a certain pattern in the data ( *do students with GPA more than 8 get placed ?*)
49. There are broadly two types of categorical variables:
  - Nominal Variable: A nominal variable has no natural ordering to its categories. They have two or more categories. For example, Marital Status (*Single, Married, Divorcee*), Gender (*Male, Female, Transgender*), etc.
  - Ordinal Variable: A variable for which the categories can be placed in an order. For example, Customer Satisfaction (*Excellent, Very Good, Good, Average, Bad*), and so on

## Data Analytics Life cycle

50. Life cycle of data analytics 
   - **Business understanding:** This phase involves understanding the business problem that you are trying to solve with data analytics. What are the specific questions that you need to answer? What are the business goals that you are trying to achieve?
   - **Data collection:** This phase involves collecting the data that you need to answer your questions and achieve your business goals. The data can come from a variety of sources, such as internal databases, external databases, and web scraping.
   - **Data preparation:** This phase involves cleaning and preparing the data for analysis. This may involve removing errors, correcting inconsistencies, and transforming the data into a format that is compatible with your analytics tools.
   - **Model building:** This phase involves building a model to answer your questions or achieve your business goals. The model can be a simple statistical model or a more complex machine learning model.
   - **Model evaluation:** This phase involves evaluating the model to ensure that it is accurate and reliable. This can be done by using a holdout test set or by using cross-validation.
   - **Deployment:** This phase involves deploying the model to production so that it can be used to make predictions or decisions.

## Data Analytics Methods and Techniques

51. **Big data:** Big data refers to large datasets that are difficult to process with traditional data processing tools. Big data can be structured, unstructured, or semi-structured.
52. **Meta data:** Meta data is data about data. It is used to describe the characteristics of data, such as its format, type, and source.
53. **Real time data:** Real time data is data that is generated and processed in real time. It is often used to make real-time decisions.
54. **Machine data:** Machine data is data that is generated by machines. It is often used to monitor and manage machines.
55. **Quantitative data:** Quantitative data is data that can be measured and expressed in numerical terms.
56. **Qualitative data:** Qualitative data is data that is non-numerical and cannot be measured.
57. **Regression analysis** is a statistical method that is used to identify the relationship between two or more variables. It is often used to predict the value of one variable based on the values of other variables.*Some exaples*
  - *Predict the sales of a product* based on the price of the product and the amount of advertising that is spent.
  - Determine the relationship between *a student's test score and their study habits.*
  - Predict the *risk of a patient developing a disease* based on their age, gender, and medical history.


## Descriptive Statistics in Python

58. **Descriptive statistics** are used to *summarize and describe a data set*. They can be used to *calculate measures of central tendency*, such as the mean and median, as well as measures of dispersion, such as the standard deviation and range.
59. The *Pandas* `describe()` method can be used to calculate a variety of descriptive statistics for a data set, including the *mean, median, standard deviation, minimum, maximum, and percentiles.*
```python
import pandas as pd

df = pd.DataFrame({'age': [25, 30, 35, 40, 45], 'height': [170, 175, 180, 185, 190]})

# Calculate descriptive statistics for the age and height columns
df.describe()

```

```
# the output is 

 age    height
count  5.000000   5.000000
mean  35.000000  177.500000
std   8.164966   7.071067
min   25.000000  170.000000
max   45.000000  190.000000
25%   30.000000  172.500000
50%   35.000000  177.500000
75%   40.000000  182.500000
```

60. The *Pandas* `value_counts()` method can be used to count the number of occurrences of each unique value in a data set.
```python
import pandas as pd

df = pd.DataFrame({'gender': ['male', 'female', 'male', 'female', 'female']})

# Count the number of occurrences of each gender
df['gender'].value_counts()

```

```
Output:

female    3
male      2
```

61. **Box plots** are a type of *data visualization that can be used to summarize the distribution of a data set*. They show the five-number summary of the data set, which includes the *minimum, maximum, median, first quartile, and third quartile.*
```python
import matplotlib.pyplot as plt

data = [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]

# Create a box plot of the data
plt.boxplot(data)
plt.show()
```

62. **Scatter plots** are a type of data visualization that can be *used to show the relationship between two variables*. They are created by *plotting each data point as a point on a graph.*

```python
import matplotlib.pyplot as plt

x = [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
y = [170, 175, 180, 185, 190, 195, 200, 205, 210, 215, 220]

# Create a scatter plot of the data
plt.scatter(x, y)
plt.show()
```

63. The Pandas `groupby()` function is used to g*roup data by one or more columns*. This allows you to aggregate the data and calculate summary statistics for each group.
64. For example, the following code groups the data in the `df` DataFrame by the `gender` column
```python
import pandas as pd

df = pd.DataFrame({'gender': ['male', 'female', 'male', 'female', 'female'], 'age': [25, 30, 35, 40, 45]})

# Group the data by gender
grouped_df = df.groupby('gender')
```
Once you have grouped the data, *you can use the Pandas aggregation functions to calculate summary statistics* for each group. For example, the following code calculates the *mean age for each gender group*:

```python
# Calculate the mean age for each gender group
mean_age_by_gender = grouped_df['age'].mean()

# Print the results
print(mean_age_by_gender)
```

```
Output:

gender
female    37.5
male      32.5
```

65. A pivot table is a type of data visualization that allows you *to summarize and group data* in a *tabular format*. Pivot tables can be created using the Pandas `pivot_table()` function.
66. To create a pivot table, you first need to *create a Pandas DataFrame*. Then, you can call the `pivot_table()` function on the DataFrame, specifying the values that you want to summarize, the columns that you want to group by, and the aggregation function that you want to use.
```python
import pandas as pd

df = pd.DataFrame({'gender': ['male', 'female', 'male', 'female', 'female'], 'age': [25, 30, 35, 40, 45], 'age_group': ['20-29', '20-29', '30-39', '30-39', '40-49']})

# Create a pivot table
pivot_table = df.pivot_table(values='age', index='gender', columns='age_group', aggfunc='mean')

# Print the pivot table
print(pivot_table)
```

67. A **heatmap** is a type of data visualization that *uses colors to represent the values in a two-dimensional matrix*. Heatmaps can be used to visualize data such as *correlation matrices, gene expression data, and sales data.*
68. To create a **heatmap**, you first need to create a NumPy array containing the data that you want to visualise. Then, you can use the Matplotlib `imshow()` function to create the heatmap.
```python
import numpy as np
import matplotlib.pyplot as plt

df = pd.DataFrame({'age': [25, 30, 35, 40, 45], 'height': [170, 175, 180, 185, 190]})

# Calculate the correlation between the columns in the DataFrame
corr_matrix = df.corr()

# Create a heatmap of the correlation matrix
plt.imshow(corr_matrix)
plt.colorbar()
plt.show()
```


## Data Preprocessing 

69. **Data preprocessing** is the process of preparing raw data for machine learning analysis. It involves c*leaning, formatting, and transforming the data* to make it more *consistent* and *easier to model*.
70. **Data wrangling** is a general term for the process of transforming and manipulating data. It can include a variety of tasks, such as:
   - **Data cleaning -** Removing duplicates inconsistencies in data 
   - **Data formatting -** converting data into a constant format
   - **Data transformation -** Includes binng the data or converting the categoricla values to numeric variables. Makes it suitable for machine learning
```python
import pandas as pd

# Load the data
df = pd.read_csv('data.csv')

# Drop duplicate rows
df = df.drop_duplicates()

# Fill in missing values
df['age'].fillna(df['age'].mean(), inplace=True)

# Convert all of the text data to lowercase
df['name'] = df['name'].str.lower()

# Normalize the data
df['age'] = (df['age'] - df['age'].mean()) / df['age'].std()

# Save the cleaned data
df.to_csv('cleaned_data.csv', index=False)
```

71. Ways to deal with missing values in data preprocessing 
  - **Drop -** Drop the rows and columns that contain the missing values 
  - **Replace -** replace the values with a placeholder value such as mean or median of the column.
  - **Leave -** This involved leaving the missing values as they are . Applies 

72. **Data formatting** is important for ensuring that the data is consistent and easy to use. For example, *you might want to convert all of the dates in the data set to a standard format*, such as ISO 8601. You might also want to convert all of the text data to lowercase, or remove any punctuation or special characters.
73. **Data normalization** is the process of *transforming the data so that all of the features are on the same scale*. This is important for ensuring that the machine learning algorithm treats all of the features equally.
74. Types of Data Normalisation
- **Simple feature scaling:** This involves scaling each feature to have a mean of 0 and a standard deviation of 1.
- **Min-max scaling:** This involves scaling each feature to the range [0, 1].
- **Z score normalization:** This involves scaling each feature to have a mean of 0 and a standard deviation of 1.

75. **Data binning** is the process of grouping the data into buckets. This can be useful for reducing the dimensionality of the data, or for converting continuous data into categorical data.
76. For example, the following code shows how to use one-hot encoding to convert the `country` variable to numeric variables:
```python
import pandas as pd

# Load the data
df = pd.read_csv('data.csv')

# Create a one-hot encoding of the country variable
df = pd.get_dummies(df, columns=['country'])

# Save the cleaned data
df.to_csv('cleaned_data.csv', index=False)
```

77. There are a *number of different ways to convert categorical values into numeric variables*. One common approach is to use one-hot encoding. One-hot encoding *creates a new binary feature for each unique category* in the categorical variable.
78. Use of Data Binning 
  - Reducing the *dimensionality of the data*
  - Converting *continuous data into categorical data*
  - *Smoothing* the data
  - Improving the *performance of machine learning algorithms*

79. There are a number of different ways to bin data in Python. One common approach is to use the `cut()` function in the Pandas library. The `cut()` function takes a Pandas Series as input and returns a new Series containing the binned values.
```python 
import pandas as pd

# Load the data
df = pd.read_csv('data.csv')

# Bin the age variable into three bins
bins = [0, 18, 65, 100]
df['age_bin'] = pd.cut(df['age'], bins=bins, labels=['Child', 'Adult', 'Senior'])

# Print the binned data
print(df['age_bin'])
```

## Important Python Codes

81. I will write down code to do data cleaning and experimenting with various functions with the data
```python
import pandas as pd
import numpy as np
df= pd.read_csv('unclean_data1.csv')
```
*This does the task of importing the libraries and reading the csv file* 

82. To print the head of the *data or the first few entries* 
```python
df1 = pd.read_csv("unclean_data2.csv")
df1.head()
```
*In the output we can see the unclean and inconsistent data* 

83. Step 1 is to print the columns and then convert them to *UPPERCASE*
```python
df.columns.str.upper()
```

84. Now feed the UPPERCASE data in the variable 
```python
df.columns= df.columns.str.upper()
```

85. Rename the column names and put serial numbers 
```python
df.rename(columns={'1':'S.N'})
```

86. Now next step is to deal with *missing values* . Can be done by *deleting the row or column with missing data* or just *put the mean value in place* of the empty cells 
87. Check the missing Data 
```python
df.rename(columns={'1':'S.N'}) 
```
*false tells us there is no missing data* 

```python 
df.isnull().any().any()
```
*Gives out how many values show true*

```python 
df.isnull().sum()
```
*Gives the sum of all the missing values in each row*

```python 
df.isnull().sum().sum()
```
*Gives the sum of all the missing value in the whole dataset*

```python 
df['DURATION']
```
*prints the whole column content under DURATION*

```python
df['DURATION'].mean()
```
*finds the whole mean of the column DURATION*

```python
df_with_mean = df.DURATION.fillna(df['DURATION'].mean())
```
*tells the whole dataset to fill the null values with the mean value*

88. The method `dropna` removes the rows and columns that contain NULL values 
```python
df_drop_with_condition = df.dropna(how="any")
```
*this declares a variable and then tells us to remove any row with any keyword*

89. The method `df.duplicated` tells us the if there is a duplicated value in the dataset with *TRUE* and *FALSE*
```python 
df.duplicated('movie_title')
```
*tells us if there is any duplicate in movie_title*

```python
df_drop_dup = df.drop_duplicates()
```
*drops any value that has duplicates*

