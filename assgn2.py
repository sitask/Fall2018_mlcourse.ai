# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 10:10:23 2018

@author: sita
"""

import numpy as np
import pandas as pd
# pip install seaborn 
import seaborn as sns
import matplotlib.pyplot as plt
dtype = {'DayOfWeek': np.uint8, 'DayofMonth': np.uint8, 'Month': np.uint8 , 'Cancelled': np.uint8, 
         'Year': np.uint16, 'FlightNum': np.uint16 , 'Distance': np.uint16, 
         'UniqueCarrier': str, 'CancellationCode': str, 'Origin': str, 'Dest': str,
         'ArrDelay': np.float16, 'DepDelay': np.float16, 'CarrierDelay': np.float16,
         'WeatherDelay': np.float16, 'NASDelay': np.float16, 'SecurityDelay': np.float16,
         'LateAircraftDelay': np.float16, 'DepTime': np.float16}

# change the path if needed
path = '2008.csv.bz2'
flights_df = pd.read_csv(path, usecols=dtype.keys(), dtype=dtype)
print(flights_df.shape)
print(flights_df.columns)
flights_df.head().T

#1. Find top-10 carriers in terms of the number of completed flights (UniqueCarrier column)?
#Which of the listed below is not in your top-10 list?
q1 = flights_df['UniqueCarrier'].value_counts()
print(q1[:10]) 

#EV is not on the top 10 list

#2. Plot distributions of flight cancellation reasons (CancellationCode).
#What is the most frequent reason for flight cancellation? (Use this link to translate 
#codes into reasons)
q2 = flights_df.dropna(subset=['CancellationCode'])
q2.head().T
q2.groupby('CancellationCode').size().plot(kind='bar')
# B - weather is the answer

#3. Which route is the most frequent, in terms of the number of flights?
#(Take a look at 'Origin' and 'Dest' features. Consider A->B and B->A 
#directions as different routes)
(flights_df.groupby(['Origin','Dest']).size()).nlargest(5)
#Origin  Dest
#SFO     LAX     13788 --> this is the answer
#LAX     SFO     13390
#OGG     HNL     12383
#LGA     BOS     12035
#BOS     LGA     12029

#4. Find top-5 delayed routes (count how many times they were delayed on departure). 
#From all flights on these 5 routes, count all flights with weather conditions 
#contributing to a delay.
#build flag by joining origin and dest names
flights_df['flag'] = flights_df['Origin']+"-"+flights_df['Dest']
#take only those for which Departure Delay is >0
new_df = flights_df[flights_df['DepDelay']>0]
#group based on routes and find delayed routes, take top 5
delayed_routes = pd.DataFrame(new_df['flag'].value_counts().reset_index()).head()
delayed_routes.columns = ['flag', 'count']

#find flights that are in delayed routes
df = pd.DataFrame()
df = new_df[(new_df['flag'].isin(delayed_routes['flag']))]
#choose only those that have a +ve Weather delay
cnt = len(df[df['WeatherDelay']>0.0])

#answer comes out to be 668

#5. Examine the hourly distribution of departure times. For that, create a new series from DepTime, removing missing values.
#Choose all correct statements:
#Flights are normally distributed within time interval [0-23] (Search for: Normal distribution, bell curve).
#Flights are uniformly distributed within time interval [0-23].
#In the period from 0 am to 4 am there are considerably less flights than from 7 pm to 8 pm.
flights_df['deptime_split'] = flights_df['DepTime']/100
flights_df['deptime_split'].plot.hist(bins=24)
#Only 3rd statement is correct

#6. Show how the number of flights changes through time (on the daily/weekly/monthly basis) 
#and interpret the findings.
#Choose all correct statements:
#The number of flights during weekends is less than during weekdays (working days).
#The lowest number of flights is on Sunday.
#There are less flights during winter than during summer.
daily_flight_count = pd.DataFrame()
daily_flight_count['Count'] = flights_df.groupby(['Month','DayofMonth','DayOfWeek']).size()
df = daily_flight_count.reset_index()
daily_flight_count['Count'].plot(kind='bar')
print('Daily flight count = ', daily_flight_count['Count'].sum())

weekday_flight_count = pd.DataFrame()
weekday_flight_count['Count'] = flights_df.groupby(['DayOfWeek']).size()
#df = daily_flight_count.reset_index()
weekday_flight_count['Count'].plot(kind='bar')
weekday_count = weekday_flight_count['Count'][0:5].sum()
weekend_count = weekday_flight_count['Count'][5:7].sum()
print('Working day flight count =', weekday_count)
print('Weekend flight count =', weekend_count)
print('Weekday flight count = ', weekday_flight_count['Count'].sum())
print('Weekend flights less than weekday"', weekend_count < weekday_count)

print('Lowest number of flights is on Sunday"', \
      weekday_flight_count.min() == weekday_flight_count.loc[7]['Count'])

#flights_df['week_num'] = '2008-' + flights_df['DayofMonth'] +'-' + flights_df
monthly_flight_count = pd.DataFrame()
monthly_flight_count['Count'] = flights_df.groupby(['Month']).size()
#df = daily_flight_count.reset_index()
monthly_flight_count['Count'].plot(kind='bar')
summer_count = monthly_flight_count.loc[6:8]['Count'].sum()
winter_count = monthly_flight_count.loc[1:2]['Count'].sum() + monthly_flight_count.loc[12]['Count']
print('Less flights in winter than summer?', winter_count<summer_count)

#7. Examine the distribution of cancellation reasons with time. Make a bar plot of cancellation reasons aggregated by months.
#Choose all correct statements:
#December has the highest rate of cancellations due to weather.
#The highest rate of cancellations in September is due to Security reasons.
#April's top cancellation reason is carriers.
#Flights cancellations due to National Air System are more frequent than those due to carriers.

#A	Carrier
#B	Weather
#C	National Air System
#D	Security

q7 = flights_df.dropna(subset=['CancellationCode'])
q7_ans = q7.groupby(['Month', 'CancellationCode']).size().unstack()
q7_ans.plot(kind='barh')
q7_1 = q7_ans.idxmax()
print('December has the highest rate of cancellations due to weather:', q7_1['B']==12) #True
print('The highest rate of cancellations in September is due to Security reasons:',q7_ans.loc[9].max() == q7_ans.loc[9]['D'])
print('Aprils top cancellation reason is carriers:',q7_ans.loc[4].max() == q7_ans.loc[4]['A'])
print('Flights cancellations due to National Air System are more frequent than those due to carriers:',q7_ans['C'].sum() > q7_ans['A'].sum()) #4 - False

#true
#false
#true
#false

#8. Which month has the greatest number of cancellations due to Carrier?
#May, January, September, April
print(q7_ans.loc[q7_ans['A'].max()==q7_ans['A']]) #April

#9. Identify the carrier with the greatest number of cancellations due to carrier in the 
#corresponding month from the previous question.
# 9E, EV, HA, AA

q9 = flights_df[(flights_df['Month']==4) & (flights_df['CancellationCode']=='A')]
print(q9['UniqueCarrier'].value_counts()) #AA is the answer

#10. Examine median arrival and departure delays (in time) by carrier. Which carrier 
#has the lowest median delay time for both arrivals and departures? Leave only 
#non-negative values of delay times ('ArrDelay', 'DepDelay'). Boxplots can be helpful
# in this exercise, as well as it might be a good idea to remove outliers in order to 
#build nice graphs. You can exclude delay time values higher than a corresponding 
#.95 percentile.

#EV, OO, AA, AQ

print(len(flights_df.index))
#arrival delays
q10 = flights_df[['FlightNum','ArrDelay','DepDelay','UniqueCarrier']]
print(q10.head())
q10_a1=q10[['UniqueCarrier', 'ArrDelay']]
q10_a1 = q10_a1.drop(q10_a1[q10_a1['ArrDelay']<=0.0].index)
q10_a11=q10_a1.groupby(['UniqueCarrier', 'ArrDelay']).size().reset_index()
q10_a11.describe()
q10_a11 = q10_a11[q10_a11.ArrDelay < np.percentile(q10_a11.ArrDelay, 95)]
ax=sns.boxplot(x=q10_a11['UniqueCarrier'], y=q10_a11['ArrDelay'])

q10_a2=q10[['UniqueCarrier', 'DepDelay']]
q10_a2 = q10_a2.drop(q10_a2[q10_a2['DepDelay']<=0.0].index)
q10_a12=q10_a2.groupby(['UniqueCarrier', 'DepDelay']).size().reset_index()
q10_a12.describe()
q10_a12 = q10_a12[q10_a12.DepDelay < np.percentile(q10_a12.DepDelay, 95)]
ax = sns.boxplot(x=q10_a12['UniqueCarrier'],y=q10_a12['DepDelay'])

#AQ is the answer.

_, axes = plt.subplots(1, 2, sharey=True, figsize=(6, 4))
sns.boxplot(data=q10_a11['ArrDelay'], ax=axes[0])
sns.violinplot(data=q10_a11['ArrDelay'], ax=axes[1])

_, axes = plt.subplots(1, 2, sharey=True, figsize=(6, 4))
sns.boxplot(data=q10_a12['DepDelay'], ax=axes[0])
sns.violinplot(data=q10_a12['DepDelay'], ax=axes[1])
