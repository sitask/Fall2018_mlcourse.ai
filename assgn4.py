# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 12:31:12 2018

@author: sita
"""
# Import libraries and set desired options
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

# Read the training and test data sets, change paths if needed
train_df = pd.read_csv('train_sessions.csv', index_col='session_id')
test_df = pd.read_csv('test_sessions.csv', index_col='session_id')

# Convert time1, ..., time10 columns to datetime type
times = ['time%s' % i for i in range(1, 11)]
train_df[times] = train_df[times].apply(pd.to_datetime)
test_df[times] = test_df[times].apply(pd.to_datetime)

# Sort the data by time
train_df = train_df.sort_values(by='time1')

# Look at the first rows of the training set
train_df.head()

# Change site1, ..., site10 columns type to integer and fill NA-values with zeros
sites = ['site%s' % i for i in range(1, 11)]
train_df[sites] = train_df[sites].fillna(0).astype(np.uint16)
test_df[sites] = test_df[sites].fillna(0).astype(np.uint16)

# Load websites dictionary
with open(r"site_dic.pkl", "rb") as input_file:
    site_dict = pickle.load(input_file)

# Create dataframe for the dictionary
sites_dict = pd.DataFrame(list(site_dict.keys()), index=list(site_dict.values()), columns=['site'])
print(u'Websites total:', sites_dict.shape[0])
sites_dict.head()

#4.1. What are the dimensions of the training and test sets (in exactly this order)?
print(train_df.shape)
print(test_df.shape)
#(253561, 21)
#(82797, 20) - Option #3

# Top websites in the training data set
top_sites = pd.Series(train_df[sites].values.flatten()
                     ).value_counts().sort_values(ascending=False).head(5)
print(top_sites)
sites_dict.loc[top_sites.drop(0).index]

#4.2. What kind of websites does Alice visit the most?
#videohostings
#social networks
#torrent trackers
#news

alice_df = train_df[train_df['target']==1]
alice_df[sites] = alice_df[sites].fillna(0).astype(np.uint16)
top_alice_sites = pd.Series(alice_df[sites].values.flatten()
                     ).value_counts().sort_values(ascending=False).head(5)
print(top_alice_sites)
sites_dict.loc[top_alice_sites.index] #videohostings - Option #1

# Create a separate dataframe where we will work with timestamps
time_df = pd.DataFrame(index=train_df.index)
time_df['target'] = train_df['target']

# Find sessions' starting and ending
time_df['min'] = train_df[times].min(axis=1)
time_df['max'] = train_df[times].max(axis=1)

# Calculate sessions' duration in seconds
time_df['seconds'] = (time_df['max'] - time_df['min']) / np.timedelta64(1, 's')

time_df.head()

#4.3. Select all correct statements:
#on average, Alice's session is shorter than that of other users
#more than 1% of all sessions in the dataset belong to Alice
#minimum and maximum durations of Alice's and other users' sessions are approximately the same
#variation about the mean session duration for all users (including Alice) is approximately the same
#less than a quarter of Alice's sessions are greater than or equal to 40 seconds

alice_time = time_df[time_df['target']==1]
gen_time  = time_df[time_df['target']==0]

at = alice_time['seconds'].mean()
gt = gen_time['seconds'].mean()

print('on average, Alice session is shorter than that of other users:', at < gt )

print('more than 1% of all sessions in the dataset belong to Alice:', \
      len(alice_time) > .01*len(time_df))

alice_time['seconds'].describe() # 0/1763
gen_time['seconds'].describe() # 0/1800 Option #3 True

#Option 4: std of alice users = 153, non-alice = 296 - so False

q3 = alice_time[alice_time['seconds']>=40]
print('less than a quarter of Alices sessions are greater than or equal to 40 seconds:', \
      len(q3) < .25*len(alice_time))

# True, False, True, False, True

# Our target variable
y_train = train_df['target']

# United dataframe of the initial data 
full_df = pd.concat([train_df.drop('target', axis=1), test_df])

# Index to split the training and test data sets
idx_split = train_df.shape[0]

# Dataframe with indices of visited websites in session
full_sites = full_df[sites]
full_sites.head()

# sequence of indices
sites_flatten = full_sites.values.flatten()

# and the matrix we are looking for 
# (make sure you understand which of the `csr_matrix` constructors is used here)
# a further toy example will help you with it
full_sites_sparse = csr_matrix(([1] * sites_flatten.shape[0],
                                sites_flatten,
                                range(0, sites_flatten.shape[0]  + 10, 10)))[:, 1:]

full_sites_sparse.shape

#####
# data, create the list of ones, length of which equal to the number of elements in the initial dataframe (9)
# By summing the number of ones in the cell, we get the frequency,
# number of visits to a particular site per session
data = [1] * 9

# To do this, you need to correctly distribute the ones in cells
# Indices - website ids, i.e. columns of a new matrix. We will sum ones up grouping them by sessions (ids)
indices = [1, 0, 0, 1, 3, 1, 2, 3, 4]

# Indices for the division into rows (sessions)
# For example, line 0 is the elements between the indices [0; 3) - the rightmost value is not included
# Line 1 is the elements between the indices [3; 6)
# Line 2 is the elements between the indices [6; 9) 
indptr = [0, 3, 6, 9]

# Aggregate these three variables into a tuple and compose a matrix
# To display this matrix on the screen transform it into the usual "dense" matrix
small_csr = csr_matrix((data, indices, indptr)).todense()

df = pd.DataFrame(small_csr)
df = df.drop([0], axis=1)
df.values

#4.4. What is the sparseness of the matrix in our small example?
#Question 4.4 - 50%

def get_auc_lr_valid(X, y, C=1.0, seed=17, ratio = 0.9):
    # Split the data into the training and validation sets
    idx = int(round(X.shape[0] * ratio))
    # Classifier training
    lr = LogisticRegression(C=C, random_state=seed, solver='liblinear').fit(X[:idx, :], y[:idx])
    # Prediction for validation set
    y_pred = lr.predict_proba(X[idx:, :])[:, 1]
    # Calculate the quality
    score = roc_auc_score(y[idx:], y_pred)
    
    return score

# Select the training set from the united dataframe (where we have the answers)
X_train = full_sites_sparse[:idx_split, :]

# Calculate metric on the validation set
print(get_auc_lr_valid(X_train, y_train))

# Function for writing predictions to a file
def write_to_submission_file(predicted_labels, out_file,
                             target='target', index_label="session_id"):
    predicted_df = pd.DataFrame(predicted_labels,
                                index = np.arange(1, predicted_labels.shape[0] + 1),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)
# Train the model on the whole training data set
# Use random_state=17 for repeatability
# Parameter C=1 by default, but here we set it explicitly
lr = LogisticRegression(C=1.0, random_state=17, solver='liblinear').fit(X_train, y_train)

# Make a prediction for test data set
X_test = full_sites_sparse[idx_split:,:]
y_test = lr.predict_proba(X_test)[:, 1]

# Write it to the file which could be submitted
write_to_submission_file(y_test, 'baseline_1.csv')

#4.5. What years are present in the training and test datasets, if united?
#
#13 and 14
#2012 and 2013
#2013 and 2014
#2014 and 2015

# Use the merged df of train and test
df = pd.DataFrame(index=full_df.index)

# Find sessions' starting and ending
df['min'] = full_df[times].min(axis=1)
df['max'] = full_df[times].max(axis=1)

# Calculate sessions' duration in seconds
df['seconds'] = (df['max'] - df['min']) / np.timedelta64(1, 's')

df.head()

df['year'] =  pd.DatetimeIndex(df['min']).year
print(df['year'].value_counts().index) #2014 and 2013 - Option 3


# Dataframe for new features
full_new_feat = pd.DataFrame(index=full_df.index)

# Add start_month feature
full_new_feat['start_month'] = full_df['time1'].apply(lambda ts: \
    100 * ts.year + ts.month).astype('float64')
    
full_new_feat['target'] = train_df['target']

#4.6. Plot the graph of the number of Alice sessions versus the new feature, start_month. Choose the correct statement:
#Alice wasn't online at all for the entire period
#From the beginning of 2013 to mid-2014, the number of Alice's sessions per month decreased
#The number of Alice's sessions per month is generally constant for the entire period
#From the beginning of 2013 to mid-2014, the number of Alice's sessions per month increased

alice_df = full_new_feat[full_new_feat['target']==1]

alice_visits = pd.DataFrame(alice_df['start_month'].value_counts().reset_index())
alice_visits.columns = ['Start Month', 'Count']
alice_visits = alice_visits.sort_values('Start Month')
alice_visits['Start Month'] = alice_visits['Start Month'].astype(str)

plt.xticks(rotation=90)
plt.plot(alice_visits['Start Month'], alice_visits['Count'])

##Only Option 1 True

# Add the new feature to the sparse matrix
tmp = full_new_feat[['start_month']].values
X_train = csr_matrix(hstack([full_sites_sparse[:idx_split,:], tmp[:idx_split,:]]))

# Compute the metric on the validation set
print(get_auc_lr_valid(X_train, y_train)) #0.7508354860175162

# Add the new standardized feature to the sparse matrix
tmp = StandardScaler().fit_transform(full_new_feat[['start_month']])
X_train = csr_matrix(hstack([full_sites_sparse[:idx_split,:], tmp[:idx_split,:]]))

# Compute metric on the validation set
print(get_auc_lr_valid(X_train, y_train)) #0.9196830663159572

#4.7. Add to the training set a new feature "n_unique_sites" – the number of the 
#unique web-sites in a session. Calculate how the quality on the validation set 
#has changed
#
#It has decreased. It is better not to add a new feature.
#It has not changed
#It has decreased. The new feature should be scaled.
#I am confused, and I do not know if it's necessary to scale a new feature.
#Tips: use the nunique() function from pandas. Do not forget to include the 
#start_month in the set. Will you scale a new feature? Why?

# Add start_month feature
train_df['start_month'] = train_df['time1'].apply(lambda ts: \
    100 * ts.year + ts.month).astype('int')
train_df[sites]=train_df[sites].replace({0:np.NaN})
train_df['n_unique_sites'] = train_df[sites].nunique(axis=1)

# Add the new feature to the sparse matrix
tmp = train_df[['n_unique_sites']].values
tmp2 = full_new_feat[['start_month']].values

X_train = csr_matrix(hstack([full_sites_sparse[:idx_split,:], tmp[:idx_split,:], \
                             tmp2[:idx_split,:]]))

# Compute the metric on the validation set
print(get_auc_lr_valid(X_train, y_train)) #0.4297

#scale the new features added and re-compute
tmp = StandardScaler().fit_transform(train_df[['n_unique_sites']])
tmp2 = StandardScaler().fit_transform(full_new_feat[['start_month']])
X_train = csr_matrix(hstack([full_sites_sparse[:idx_split,:], tmp[:idx_split,:], \
                             tmp2[:idx_split,:]]))

# Compute the metric on the validation set
print(get_auc_lr_valid(X_train, y_train)) #0.91616

#Answer is Option 3 (can be used only with scaling)

#4.8. Add two new features: start_hour and morning. Calculate the metric. 
# Which of these features gives an improvement?
#The start_hour feature is the hour at which the session started (from 0 to 23), 
#and the binary feature morning is equal to 1 if the session started in the morning and 
#0 if the session started later (we assume that morning means start_hour is equal to 11 or less).
#
#Will you scale the new features? Make your assumptions and test them in practice.
#
#None of the features gave an improvement :(
#start_hour feature gave an improvement, and morning did not
#morning feature gave an improvement, and start_hour did not
#Both features gave an improvement

train_df['start_hour'] =  pd.DatetimeIndex(train_df['time1']).hour
train_df['morning'] = np.where(train_df['start_hour']<=11, 1, 0)

# we know that start_month has to be scaled for the results to be good, so using it scaled
tmp = train_df[['start_hour']].values
tmp2 = train_df[['morning']].values
tmp3 = StandardScaler().fit_transform(full_new_feat[['start_month']])
#with start_hour
X_train = csr_matrix(hstack([full_sites_sparse[:idx_split,:], tmp[:idx_split,:], \
                             tmp3[:idx_split,:]]))
print(get_auc_lr_valid(X_train, y_train)) # 0.9572718607645079

#with morning
X_train = csr_matrix(hstack([full_sites_sparse[:idx_split,:], tmp2[:idx_split,:], \
                             tmp3[:idx_split,:]]))
print(get_auc_lr_valid(X_train, y_train)) # 0.9486655018622379

#with start_hour and morning
X_train = csr_matrix(hstack([full_sites_sparse[:idx_split,:], tmp[:idx_split,:], \
                             tmp2[:idx_split,:], tmp3[:idx_split,:]]))
print(get_auc_lr_valid(X_train, y_train)) # 0.9584922183335024

# all 3 features scaled
tmp = StandardScaler().fit_transform(train_df[['start_hour']])
tmp2 = StandardScaler().fit_transform(train_df[['morning']])
X_train = csr_matrix(hstack([full_sites_sparse[:idx_split,:], tmp[:idx_split,:], \
                             tmp2[:idx_split,:], tmp3[:idx_split,:]]))
print(get_auc_lr_valid(X_train, y_train)) # 0.9591485907617544

#Option 4 is the answer - both features gave an improvement before and after scaling

full_new_feat['start_hour'] =  pd.DatetimeIndex(full_df['time1']).hour
full_new_feat['morning'] = np.where(full_new_feat['start_hour']<=11, 1, 0)

# Compose the training set
tmp_scaled = StandardScaler().fit_transform(full_new_feat[['start_month', 
                                                           'start_hour', 
                                                           'morning']])
X_train = csr_matrix(hstack([full_sites_sparse[:idx_split,:], 
                             tmp_scaled[:idx_split,:]]))

# Capture the quality with default parameters
score_C_1 = get_auc_lr_valid(X_train, y_train)
print(score_C_1) # 0.959152062833017

from tqdm import tqdm

# List of possible C-values
Cs = np.logspace(-3, 1, 10)
scores = []
for C in tqdm(Cs):
    scores.append(get_auc_lr_valid(X_train, y_train, C=C))

plt.plot(Cs, scores, 'ro-')
plt.xscale('log')
plt.xlabel('C')
plt.ylabel('AUC-ROC')
plt.title('Regularization Parameter Tuning')
# horizontal line -- model quality with default C value
plt.axhline(y=score_C_1, linewidth=.5, color='b', linestyle='dashed') 
plt.show()

#4.9. What is the value of parameter C (if rounded to 2 decimals) that corresponds 
#to the highest model quality?
max_score = max(scores)
max_score_index = scores.index(max_score)

print(Cs[max_score_index]) # 0.1668100537200059 is the answer

#full_new_feat['n_unique_sites'] =  train_df[['n_unique_sites']]

# Prepare the training and test data
#tmp_scaled = StandardScaler().fit_transform(full_new_feat[['start_month', 'start_hour', 
#                                                           'morning','n_unique_sites']])
tmp_scaled = StandardScaler().fit_transform(full_new_feat[['start_month', 'start_hour', 
                                                           'morning']])
X_train = csr_matrix(hstack([full_sites_sparse[:idx_split,:], 
                             tmp_scaled[:idx_split,:]]))
X_test = csr_matrix(hstack([full_sites_sparse[idx_split:,:], 
                            tmp_scaled[idx_split:,:]]))

# Train the model on the whole training data set using optimal regularization parameter
C = Cs[max_score_index]
lr = LogisticRegression(C=C, random_state=17, solver='liblinear').fit(X_train, y_train)

# Make a prediction for the test set
y_test = lr.predict_proba(X_test)[:, 1]

# Write it to the submission file
write_to_submission_file(y_test, 'baseline_2.csv')

