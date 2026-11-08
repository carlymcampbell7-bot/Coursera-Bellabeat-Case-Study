# Coursera-Bellabeat-Case-Study
By Carly Campbell (Fall 2026)


## Overview

In my Google Data Analytics Capstone Project, I will be a junior data analyst working on the marketing analyst team at Bellabeat. Bellabeat is a high-tech manufacturer of health-focused products for women. My objective will be to analyze smart device data for Bellabeat with a focus on their app, and to analyze the data to gain valuable consumer insights for app usage. 

I will be using the data analysis process provided through Coursera's Google Data Analysis course: Ask, Prepare, Process, Analyze, Share, and Act. By following these steps, I aim to provide Bellabeat with data-backed recommendations to improve marketing strategy and growth opportunities. 

Through this case study I will be using **Python** since it is a powerful, high-level, versatile language that allows data professionals to analyze data in a fast and efficient manner.

The link to the Kaggle Notebook can be found here (

I will be using the data analysis process provided in the Google Data Analytics Course: ask, prepare, process, analyze, and share.

## Buisness Problem

Bellabeat, a successful company, is a high-tech manufacturer of health-focused products for women. Cofounder and Chief Creative Officer of Bellabeat, Urška Sršen, believes that by analyzing smart device fitness data (FitBit Fitness Tracker Data), the company could gain valuable insights on consumer usage to use for marketing strategies and potential growth opportunities. 

Bellabeat products include: 

  **Bellabeat app:** The Bellabeat app connects to their smart devices and provides users with health data related to their activity, sleep, stress, menstrual cycle, and mindfulness habits to help users understand their current habits and progress in healthy decisions. 

  **Leaf:** Bellabeat's leaf is a wellness tracker that can be utilized as a bracelet, clip, or necklace to track activity, sleep and stress levels through the Bellabeat app.

  **Time:** Bellabeat provides a wellness watch to track user activity, sleep, and stress levels, connecting to the Bellabeat app to provide users with daily insights.  

  **Spring:** Bellabeat offers a high-tech water bottle that tracks users daily water intake. Spring connects to the bellabeat app, tracking users hydration levels. 

  **Bellabeat membership:** In addition, Bellabeat offers subscription based membership program to give users 24/7 access to individualized advice on nutrition, activity, sleep, health and beauty, and mindfulness. 


## Ask: work to understand the task to be solved and/or questions to be answered.
**Key stakeholders**
Key stakeholders are Urška Sršen, Bellabeat's cofounder and Chief Creative Officer, Sando Mur, Mathematician and Bellabeat's cofounder, and the Bellabeat marketing and analytics team. 

**Deliverable**
The business task is to analyze smart device usage data on non-Bellabeat smart devices tracking users' activity, sleep, and weight data to identify insights to promote healthy habits, improve user engagement, and to inform product features individualized to customer needs.
## Prepare: find & collect data needed to answer the questions.
**Data source:**
Data was recommended through Google Data Analytics course, and is publicly avaiable on Kaggle, titled FitBit Fitness Tracker Data from Möbius:   (https://www.kaggle.com/datasets/arashnic/fitbit/data)

I will be using the ROCCC framework from the Google Data Analytics course to evaluate the data source:

**Relevant:** This dataset includes insights on minute-level output for physical activity, heart rate, and sleep monitoring, which is relevant to analyzing user wellness and health patterns for Bellabeat's products.

**Original:** This dataset is not and was collected a third-party  provider via Amazon Mechanical Turk who surveyed 30 real users that consented as respondents. The data is publicly available on Kaggle.

**Comprehensive:** The dataset covers steps, sleep hours, calories burned, weight, and other important variables that can be used to find solutions for Bellabeat's business tasks. 

**Current:** This dataset was generated between 03/12/2016-05/12/2016. Since it is now 2026, data is not current and may not include new factors and differing consumer insights that would be important to consider. 

**Cited:** Data source is collected from third-party source so it is unknown.

**Limitations in dataset:** 
- There is only data available from 30 different participants, the data shows that there are 35 different users for daily activity, 24 users for sleep, and 13 users for weight data. Possible explanations for the 35 different recorded users could be some users owning multiple products under different Id's, or an error in the data description. 
- Gender is not specified in this data which is an important factor when considering the data.
- The sample size of 30 FitBit users is not an accurate representation of the fitness population as a whole. 
- The data is not current, as it was collected in 2016 and it is now 2026.
- The data is collected via a distributed survey and we are unable to ensure its integrity or accuracy.
- The weight log dataset is incomplete, possibly suggesting low user engagement.

Overall, this dataset is considered poor quality and a higher quality dataset is recommended to be used in order to gain valuable consumer insights to improve business growth.

**Import libraries and load data**

After importing the data to my Kaggle notebook using Python, I imported the necessary libraries and loaded the datasets I needed for my data analysis. 
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
daily1 = pd.read_csv('/kaggle/input/datasets/arashnic/fitbit/mturkfitbit_export_3.12.16-4.11.16/Fitabase Data 3.12.16-4.11.16/dailyActivity_merged.csv')
daily2 = pd.read_csv('/kaggle/input/datasets/arashnic/fitbit/mturkfitbit_export_4.12.16-5.12.16/Fitabase Data 4.12.16-5.12.16/dailyActivity_merged.csv')
sleep = pd.read_csv('/kaggle/input/datasets/arashnic/fitbit/mturkfitbit_export_4.12.16-5.12.16/Fitabase Data 4.12.16-5.12.16/sleepDay_merged.csv')
weight1 = pd.read_csv('/kaggle/input/datasets/arashnic/fitbit/mturkfitbit_export_3.12.16-4.11.16/Fitabase Data 3.12.16-4.11.16/weightLogInfo_merged.csv')
weight2 = pd.read_csv('/kaggle/input/datasets/arashnic/fitbit/mturkfitbit_export_4.12.16-5.12.16/Fitabase Data 4.12.16-5.12.16/weightLogInfo_merged.csv')

print(daily1.head())
print(daily2.head())
print(sleep.head())
print(weight1.head())
print(weight2.head())
```



## Process: clean & organize data.
All data cleaning processes are contained in the project's kaggle notebook. 

```python
# Merge daily activity & weight datasets & clean duplicates
daily_combined = pd.concat([daily1, daily2], ignore_index=True)
daily_combined = daily_combined.drop_duplicates(subset=['Id', 'ActivityDate'])

weight_combined = pd.concat([weight1, weight2], ignore_index=True)
weight_combined = weight_combined.drop_duplicates(subset=['Id', 'Date'])

# Clean sleep dataset duplicates
sleep = sleep.drop_duplicates(subset=['Id', 'SleepDay'])

print(daily_combined.info())
print(weight_combined.info())
print(sleep.info())
```
```python
# Convert date columns to datetime format
daily_combined['ActivityDate'] = pd.to_datetime(daily_combined['ActivityDate'])
weight_combined['Date'] = pd.to_datetime(weight_combined['Date'], format='%m/%d/%Y %I:%M:%S %p')
sleep['SleepDay'] = pd.to_datetime(sleep['SleepDay'], format='%m/%d/%Y %I:%M:%S %p')

print(daily_combined.dtypes)
print(weight_combined.dtypes)
print(sleep.dtypes)
```
```python
# Check for missing values
print(daily_combined.isnull().sum())
print(weight_combined.isnull().sum())
print(sleep.isnull().sum())
```
Code returned 'Fat' missing values count of 94, I explored it further with summary statistics of the column excluding missing values, visualizing the missing values distribution, and comparing rows with and without missing Fat. Reference Kaggle notebook for more details.
```python
# Remove rows where 'Fat' is missing values
weight_cleaned = weight_combined.dropna(subset=['Fat'])
print(weight_cleaned['Fat'].describe())
```
```python
# Create new columns, convert minutes asleep to hours, extract day of week
sleep['TotalSleepHours'] = sleep['TotalMinutesAsleep'] / 60
daily_combined['DayOfWeek'] = daily_combined['ActivityDate'].dt.day_name()

print(sleep[['TotalMinutesAsleep', 'TotalSleepHours']].head())
print(daily_combined[['ActivityDate', 'DayOfWeek']].head())
```
Then I converted column names to snake_case format, refer to Kaggle notebook to see code. 
```python
# Number of individual users for each dataset
print(f"daily users: {daily_combined['Id'].nunique()}"),
print(f"sleep users: {sleep['Id'].nunique()}"),
print(f"weight users: {weight_combined['Id'].nunique()}"),
```
```python
# Explore numerical summaries
print(daily_combined['total_steps'].describe())
print(weight_combined['weight_kg'].describe())
print(sleep['total_sleep_hours'].describe())
```
```python
# Explore numerical summaries
print(daily_combined['total_steps'].describe())
print(weight_combined['weight_kg'].describe())
print(sleep['total_sleep_hours'].describe())
```
I then identified outliers for each dataset, refer to Kaggle notebook to see code. 

## Analyze & Share: data analysis to find trends and relationships to find answers, create visualizations to present. 

Refer to Kaggle notebook to see full code analysis.

I explored first explored summary statistics of each dataset. I found that 
- the average steps taken per day is 7,247.
- average distance is about 5.19 km which is closely matching tracker distance 5.17 km, which could indicate consistent tracking
- mean for logged_activities_distance is very low, 0.13 km and median is 0, most users did not log additional activities beyond automatic tracking
- average total_sleep_hours is about 7 hours 
Then I found the average steps per user and average steps by day of the week. The findings suggested users prefer to track activities during weekdays over weekends.

I grouped by date and calculated average steps over time, then visualized it in a line plot.
```python
# Group by date and calculate average steps
steps_over_time = daily_combined.groupby('activity_date')['total_steps'].mean().reset_index()

print(steps_over_time.head())

# Line plot of total steps over time
sns.lineplot(x='activity_date', y='total_steps', data=daily_combined)
plt.title('Total Steps Over Time')
plt.xticks(rotation=45)
plt.show()
```
 <img width="622" height="507" alt="download" src="https://github.com/user-attachments/assets/3ea46578-7677-4b8d-9f6a-7fb279b2c512" />
This data showed wide range of steps, showing some very active days and some with no steps recorded.
```python
# Scatter plot of steps vs. calories
plt.figure(figsize=(8,6))
sns.scatterplot(data=daily_combined, x='total_steps', y='Calories')
plt.title('Steps vs Calories Burned')
plt.show()
```
<img width="704" height="547" alt="download-1" src="https://github.com/user-attachments/assets/e18f77ac-b754-4a0f-b257-451112ba5a3e" />
There is a positive correlation. Wide range in calories being burned, explanation for this could be due to differing workouts and gender differences that were not recorded in the data. 
```python
# Distribution of activity levels
plt.figure(figsize=(6,4))
sns.countplot(data=daily_combined, x='activity_level', order=['Low', 'Moderate', 'High'])
plt.title('Activity Level Distribution')
plt.show()
```
<img width="540" height="393" alt="download-2" src="https://github.com/user-attachments/assets/3eb7031f-57df-480d-a9f2-ad00e140dda2" />
This visual shows most users spend majority of their activity level in the low category, but there are also significant amounts in the moderate and high categories.
```python
# How much time users spend inactive
daily_combined['sedentary_minutes_percent'] = (daily_combined['sedentary_minutes'] / (24*60)) * 100
print("Summary statistics for sedentary time percentage:")
print(daily_combined['sedentary_minutes_percent'].describe())

# [out]

Summary statistics for sedentary time percentage:
count    1373.000000
mean       68.987922
std        21.860763
min         0.000000
25%        50.625000
50%        73.472222
75%        86.597222
max       100.000000
Name: sedentary_minutes_percent, dtype: float64
```
This data shows that on average, users spend about 69% of their tracked time being sedentary. 
```python
# Align data by date
merged_data = daily_combined[['activity_date', 'total_steps']].merge(
    sleep[['sleep_day', 'total_sleep_hours']], left_on='activity_date', right_on='sleep_day', how='left'
).merge(
    weight_combined[['Date', 'weight_kg']], left_on='activity_date', right_on='Date', how='left'
)

correlation_matrix = merged_data[['total_steps', 'total_sleep_hours', 'weight_kg']].corr()
print(correlation_matrix)

# [out]
                     total_steps  total_sleep_hours  weight_kg
total_steps           1.000000          -0.018766        NaN
total_sleep_hours    -0.018766           1.000000        NaN
weight_kg                  NaN                NaN        NaN
```
This data shows there is essentially no linear relationship between total steps and total sleep hours in the data, and that the weight_kg column has missing values causing correlation to not be computed.
```python
# Boxplot for sleep hours
sns.boxplot(y=sleep['total_sleep_hours'])
plt.title('Total Sleep Hours Distribution')
plt.show()
```
<img width="563" height="416" alt="download-3" src="https://github.com/user-attachments/assets/660740ce-0c65-4c7a-9852-aab80f6fb888" />

```python
# Distribution with histograms
plt.figure(figsize=(8,5))
daily_combined['total_steps'].hist(bins=30, color='lightblue')
plt.title('Distribution of Total Steps')
plt.xlabel('Total Steps')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8,5))
weight_combined['weight_kg'].hist(bins=20, color='lightgreen')
plt.title('Distribution of Weight (kg)')
plt.xlabel('Weight (kg)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8,5))
sleep['total_sleep_hours'].hist(bins=20, color='lightpink')
plt.title('Distribution of Total Sleep Hours')
plt.xlabel('Sleep Hours')
plt.ylabel('Frequency')
plt.show()
```
<img width="686" height="470" alt="download" src="https://github.com/user-attachments/assets/e0e18b06-1a3d-4bd6-ab8a-c1d2cc606921" />
<img width="686" height="470" alt="download-1" src="https://github.com/user-attachments/assets/3394531f-5431-4eb2-b852-7a39441e65b1" />
<img width="695" height="470" alt="download-2" src="https://github.com/user-attachments/assets/ed2493e5-943d-443f-ba71-7593d9fb3a31" />
These distribution plots show averages for total steps, sleep hours, and weight, showing that users are not getting, on average, the recommended amount of steps and sleep per day. 

I also plotted the outliers for each dataset for columns total_steps, sleep_hours, and weight_kg. Refer to Kaggle notebook to see code.

## Act: 
**Data trends**
- High rates of sedentary activity levels suggests that majority of users are using the smart devices to track sedentary activities rather than for tracking fitness activities.
- Users are getting on average 7 hours of sleep, which indicates a large amount of users are not getting the recommended amount of healthy sleep hours per night.
- Data also shows that on average users are getting 7,247 steps per day, which is less than the recommended daily step amount.
**How Bellabeat could use insights to influence marketing strategy**
- Bellabeat can assist users by making an individualized fitness plan to increase daily step count, higher activity levels, better sleep, logging workouts and healthy motivation.
- Bellabeat's team could provide users with educational resources like recommendations, benefits, and at-home tutorials of exercise and fitness topics to help encourage users creating and meeting fitness goals. 
- Bellabeat could incorporate notifications and rewards to encourage users for creating and working towards their fitness goals.
- Bellabeat could foster and create a community of its users, for women who prioritize health and wellness, where they can encourage one another and share new workouts or healthy meals, or even local groups where they can share their favorite hikes or workout classes.
 
