# Coursera-Bellabeat-Case-Study
By Carly Campbell (Fall 2026)


## Overview

In my Google Data Analytics Capstone Project, I will be a junior data analyst working on the marketing analyst team at Bellabeat. Bellabeat is a high-tech manufacturer of health-focused products for women. My objective will be to analyze smart device data for Bellabeat with a focus on their app, and to analyze the data to gain valuable consumer insights for app usage. 

I will be using the data analysis process provided through Coursera's Google Data Analysis course: Ask, Prepare, Process, Analyze, Share, and Act. By following these steps, I aim to provide Bellabeat with data-backed recommendations to improve marketing strategy and growth opportunities. 

Through this case study I will be using **Python** since it is a powerful, high-level, versatile language that allows data professionals to analyze data in a fast and efficient manner.

The link to the [Kaggle notebook](https://www.kaggle.com/code/carlycampbell/bellabeat-case-study)

I will be using the data analysis process provided in the Google Data Analytics Course: ask, prepare, process, analyze, and share.

## Business Problem

Bellabeat develops wellness products designed to empower women to improve their health through better tracking and insights.
Bellabeat products include: 

  **Bellabeat app:** The Bellabeat app connects to their smart devices and provides users with health data related to their activity, sleep, stress, menstrual cycle, and mindfulness habits to help users understand their current habits and progress in healthy decisions. 

  **Leaf:** Bellabeat's leaf is a wellness tracker that can be utilized as a bracelet, clip, or necklace to track activity, sleep and stress levels through the Bellabeat app.

  **Time:** Bellabeat provides a wellness watch to track user activity, sleep, and stress levels, connecting to the Bellabeat app to provide users with daily insights.  

  **Spring:** Bellabeat offers a high-tech water bottle that tracks users daily water intake. Spring connects to the bellabeat app, tracking users hydration levels. 

  **Bellabeat membership:** In addition, Bellabeat offers subscription based membership program to give users 24/7 access to individualized advice on nutrition, activity, sleep, health and beauty, and mindfulness. 

The business task is to analyze usage patterns from non-Bellabeat smart devices to uncover trends in activity, sleep, and health behavior. These insights can help Bellabeat improve future product features and marketing strategies that encourage healthier habits among its users.

## Ask: work to understand the task to be solved and/or questions to be answered.
**Key stakeholders**
Key stakeholders are Urška Sršen, Bellabeat's cofounder and Chief Creative Officer, Sando Mur, Mathematician and Bellabeat's cofounder, and the Bellabeat marketing and analytics team. 

**Deliverable**
The goal is to analyze smart device data to identify usage trends, understand behaviors related to activity and sleep, and provide recommendations that Bellabeat could use to better engage users and grow its membership and product adoption.
## Prepare: find & collect data needed to answer the questions.
**Data source:**
Data was recommended through Google Data Analytics course, and is publicly avaiable on Kaggle, titled FitBit Fitness Tracker Data from Möbius:   (https://www.kaggle.com/datasets/arashnic/fitbit/data)

I will be using the ROCCC framework from the Google Data Analytics course to evaluate the data source:

**Relevant:** The dataset includes information on activity, sleep, calories, and weight, which is directly relevant to Bellabeat’s wellness-focused products.

**Original:** The data was collected from a third-party source via Amazon Mechanical Turk from 30 real users who consented to participate.

**Comprehensive:** The dataset covers steps, sleep hours, calories burned, weight, and other important variables that can be used to find solutions for Bellabeat's business tasks. 

**Current:** Current: The dataset was collected in 2016, so it is not current and may not reflect current user behavior or new product trends. 

**Cited:** The source is publicly available but not directly cited by Bellabeat, and its origin is not fully verified.

**Limitations in dataset:** 
- There is only data available from 30 different participants, which is a relatively small sample size
- The data shows that there are 35 different users for daily activity, 24 users for sleep, and 13 users for weight data. Possible explanations for the 35 different recorded users could be some users owning multiple products under different Id's, or an error in the data description. 
- Gender is not specified in this data which is an important factor when considering the data for a women-focused health product.
- The sample size of 30 FitBit users is not an accurate representation of the fitness population as a whole. 
- The data is not current, as it was collected in 2016 and it is now 2026.
- The data is collected via a distributed survey and we are unable to ensure its integrity or accuracy.
- The weight log dataset is incomplete, possibly suggesting limited user engagement with that feature.

Overall, the dataset is useful for a preliminary analysis, but it is not ideal for drawing strong conclusions about a broader population. A more current and representative dataset would provide better insights for Bellabeat’s business strategy.



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
The data was imported and cleaned using Python in a Kaggle notebook. The analysis included:

- Merging daily activity datasets

- Removing duplicate records

- Converting date columns to datetime format

- Checking missing values

- Cleaning incomplete weight entries

- Creating new variables such as total sleep hours and day-of-week

- Standardizing column names for consistency
 

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

# Check for missing values
print(daily_combined.isnull().sum())
print(weight_combined.isnull().sum())
print(sleep.isnull().sum())
```
```python
# Create new columns, convert minutes asleep to hours, extract day of week
sleep['TotalSleepHours'] = sleep['TotalMinutesAsleep'] / 60
daily_combined['DayOfWeek'] = daily_combined['ActivityDate'].dt.day_name()

print(sleep[['TotalMinutesAsleep', 'TotalSleepHours']].head())
print(daily_combined[['ActivityDate', 'DayOfWeek']].head())
```
```python
# Number of individual users for each dataset
print(f"daily users: {daily_combined['Id'].nunique()}"),
print(f"sleep users: {sleep['Id'].nunique()}"),
print(f"weight users: {weight_combined['Id'].nunique()}"),

# Explore numerical summaries
print(daily_combined['total_steps'].describe())
print(weight_combined['weight_kg'].describe())
print(sleep['total_sleep_hours'].describe())
```

## Analyze & Share: data analysis to find trends and relationships to find answers, create visualizations to present. 

The full analysis, including all code and visualizations, is available in the Kaggle notebook. The main analysis focused on:

- Summary statistics for activity, sleep, and weight

- Trends in steps and calories burned

- Sleep patterns and sedentary time

- Activity level distributions

- Relationships between daily activity and health outcomes



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
# Align data by date
merged_data = daily_combined[['activity_date', 'total_steps']].merge(
    sleep[['sleep_day', 'total_sleep_hours']], left_on='activity_date', right_on='sleep_day', how='left'
).merge(
    weight_combined[['Date', 'weight_kg']], left_on='activity_date', right_on='Date', how='left'
)

correlation_matrix = merged_data[['total_steps', 'total_sleep_hours', 'weight_kg']].corr()
print(correlation_matrix)

...

                     total_steps  total_sleep_hours  weight_kg
total_steps           1.000000          -0.018766        NaN
total_sleep_hours    -0.018766           1.000000        NaN
weight_kg                  NaN                NaN        NaN
```
This data shows there is essentially no linear relationship between total steps and total sleep hours in the data, and that the weight_kg column has missing values causing correlation to not be computed.


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
**Key findings**
I found that:

- Average daily steps were approximately 7,247

- Average daily sleep was about 6.9 hours

- Average sedentary time was high, suggesting many users spent a large portion of their day inactive

- Users generally logged lower levels of activity and did not consistently meet recommended fitness benchmarks

- There was a positive relationship between steps and calories burned

These visualizations indicated that while users were tracking activity, many were still not meeting daily health recommendations. The data also suggested a moderate to high level of inactivity, which presents an opportunity for Bellabeat to support users with habit-building tools and personalized guidance.

## Act: 
**Data trends**
- A large percentage of tracked time is sedentary, which indicates that many users are not consistently engaging in movement throughout the day and may be using the smart devices to track sedentary activities rather than for tracking fitness activities.

- Activity levels are concentrated in low-to-moderate ranges rather than sustained active routines.
  
- Users are getting on average 6.9 hours of sleep, which indicates a large amount of users are not getting the recommended amount of healthy [sleep hours per night](https://www.nia.nih.gov/health/sleep/sleep-and-older-adults).
  
- Data also shows that on average users are getting 7,247 steps per day, which is less than the [recommended daily step amount](https://www.calm.com/blog/how-many-steps-per-day).
  
**How Bellabeat could use insights to influence marketing strategy**
Bellabeat could use these insights to strengthen its product strategy and marketing approach:

- Personalized fitness plans: Create individualized daily goals that encourage users to increase step count, improve activity levels, and develop healthier routines.

- Sleep coaching: Offer insights and reminders that help users build better sleep habits and improve consistency.

- Wellness nudges: Use notifications and reminders to motivate users to move more, meet hydration goals, and maintain routines.

- Reward systems: Incorporate streaks, badges, or challenges to improve engagement and retention.

- Community building: Create supportive communities where women can share goals, workouts, nutrition ideas, and motivation.

- Educational content: Provide at-home workout tutorials, nutrition guidance, and healthy lifestyle resources tailored to the user’s goals.

- Goal tracking and accountability: Help users set realistic milestones and provide progress tracking to improve adherence.

## Conclusion
This case study demonstrates that wearable data can provide valuable insights into user behavior and habits. The analysis shows that many users are not meeting recommended levels of daily movement and sleep, which highlights a clear opportunity for Bellabeat to create more engaging, personalized, and supportive wellness experiences.
By leveraging these insights, Bellabeat can strengthen its product offerings, improve customer engagement, and build a stronger connection with women seeking healthier lifestyles.

## Tools used
- Python
 
- Pandas

- NumPy

- Matplotlib

- Seaborn

- Kaggle
## Project links
- Kaggle notebook: [Google Data Analytics Bellabeat Case Study](https://www.kaggle.com/code/carlycampbell/bellabeat-case-study)

- Data source: https://www.kaggle.com/datasets/arashnic/fitbit/data

**Additional data sources:**

- [How many steps per day should you take?](https://www.calm.com/blog/how-many-steps-per-day)

- [Sleep and Older Adults](https://www.nia.nih.gov/health/sleep/sleep-and-older-adults)
