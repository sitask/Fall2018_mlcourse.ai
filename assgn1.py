# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
data = pd.read_csv('athlete_events.csv')

data.describe()

#1. How old were the youngest male and female participants of the 1996 Olympics?
data_1996 = data[data['Year']==1996]

min_f = data_1996[data_1996['Sex']=='F']['Age'].min()
min_m = data_1996[data_1996['Sex']=='M']['Age'].min()
print(min_m, min_f)

#2. What was the percentage of male gymnasts among all the male 
# participants of the 2000 Olympics? Round the answer to the first decimal.
#Hint: here and further if needed drop duplicated sportsmen to count only unique ones.
data_2000 = data[data['Year']==2000]
data_2000 = data_2000[data_2000['Sex']=='M']
total = data_2000['ID'].nunique()
data_2000 = data_2000[data_2000['Sport']=='Gymnastics']
gymcount = data_2000['ID'].nunique()
pct = gymcount/total*100
print("%.1f"% pct)

#3. What are the mean and standard deviation of height for female basketball 
#players participated in the 2000 Olympics? Round the answer to the first decimal.
yr = data['Year']==2000
sex = data['Sex']=='F'
sport = data['Sport']=='Basketball'
data_q3 = data[yr & sex & sport]
std = data_q3['Height'].std()
mean = data_q3['Height'].mean()
print("%.1f"% mean, "%.1f"% std)

#4. Find a sportsperson participated in the 2002 Olympics, with the highest 
#weight among other participants of the same Olympics. What sport did he or she do?
yr = data['Year']==2002
data_q4 = data[yr]
max_wgt = data_q4['Weight'].max()
sp = data_q4[data_q4['Weight']==max_wgt]['Sport']
print(sp)

#5. How many times did Pawe Abratkiewicz participate in the Olympics held in different years?
data_q5 = data[data['Name']=='Pawe Abratkiewicz']
cnt = data_q5['Games'].nunique()
print(cnt)

#6. How many silver medals in tennis did Australia win at the 2000 Olympics?
yr = data['Year']==2000
team = data['Team']=='Australia'
sport = data['Sport']=='Tennis'
medal = data['Medal']=='Silver'
data_q6 = data[yr & team & sport & medal]
print(data_q6['ID'].count())

#7. Is it true that Switzerland won fewer medals than Serbia at the 2016 Olympics? 
#Do not consider NaN values in Medal column.
yr = data['Year']==2016
team1 = data['Team']=='Switzerland'
team2 = data['Team']=='Serbia'
data_q7 = data[yr & (team1 | team2)]
data_q7.dropna(0,inplace=True)
cnt1 = data_q7[data_q7['Team']=='Serbia']['ID'].count()
cnt2 = data_q7[data_q7['Team']=='Switzerland']['ID'].count()
print(cnt2<cnt1)

#8. What age category did the fewest and the most participants of the 2014 Olympics belong to?
bins=[0,15,25,35,45,55]
data_q8 = data[data['Year']==2014]
data_q8['binned'] = pd.cut(data_q8['Age'], bins)
cnts = pd.DataFrame(data_q8['binned'].value_counts())
print(cnts.count())
print(cnts.index[4], cnts.index[0])

#9. Is it true that there were Summer Olympics held in Lake Placid? Is it true 
#that there were Winter Olympics held in Sankt Moritz?
season = data['Season']=='Summer'
location = data['City']=='Lake Placid'
df = data[season & location]
print(df.shape[0] > 0)
season = data['Season']=='Winter'
location = data['City']=='Sankt Moritz'
df = data[season & location]
print(df.shape[0] > 0)

#10. What is the absolute difference between the number of unique sports at 
#the 1995 Olympics and 2016 Olympics?
yr1 = data[data['Year']==1995]['Sport'].nunique()
yr2 = data[data['Year']==2016]['Sport'].nunique()
print(yr2-yr1)




