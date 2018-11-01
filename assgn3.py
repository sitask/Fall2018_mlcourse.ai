# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 17:29:34 2018

@author: sita
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.externals.six import StringIO  
from matplotlib import pyplot as plt
import pydotplus
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
data = pd.read_csv('mlbootcamp5_train.csv',sep=';')

data.describe()
data.head()

#Transform the features: create "age in years" (full age) and also create 3 binary 
#features based on cholesterol and 3 more on gluc, where they are equal to 1, 2 or 3. 
#This method is called dummy-encoding or One Hot Encoding (OHE). It is more convenient
# to use pandas.get_dummmies.. There is no need to use the original features cholesterol 
# and gluc after encoding.
data['age_yrs'] = (data['age']/365).round()

chol123 = pd.get_dummies(data['cholesterol'])
chol123.columns = ('chol1','chol2','chol3')

gluc123 = pd.get_dummies(data['gluc'])
gluc123.columns = ('gluc1','gluc2','gluc3')

data = data.join(chol123)
data = data.join(gluc123)

y = data['cardio']
y.value_counts(normalize=True)

X = data.drop(columns=['age','cholesterol','gluc','cardio'])

#Split data into train and holdout parts in the proportion of 7/3 using
# sklearn.model_selection.train_test_split with random_state=17.
X_train, X_valid, y_train, y_valid = train_test_split( X, y, test_size=0.3, random_state=17)

#Train the decision tree on the dataset (X_train, y_train) with max depth 
#equals to 3 and random_state=17. Plot this tree with sklearn.tree.export_graphviz, dot and pydot. 
dtree = DecisionTreeClassifier( random_state = 17, max_depth=3)
dtree.fit(X_train, y_train)
features = list(X.columns)

dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,
                feature_names=features)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

graph.write_png('tree.png')

#Make predictions for holdout data (X_valid, y_valid) with the trained decision tree. 
#Calculate accuracy.
y_pred = dtree.predict(X_valid)
accu1 = accuracy_score(y_valid, y_pred)
print("Validation Set Accuracy using Decision Tree (Depth=3): %.2f%%" % (accu1 * 100.0))

#Set up the depth of the tree using cross-validation on the dataset (X_train, y_train) 
#in order to increase quality of the model. Use GridSearchCV with 5 folds. Fix 
#random_state=17 and change max_depth from 2 to 10.

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import cross_val_score

params = {'max_depth': np.arange(2, 11)}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
cv_dtree = GridSearchCV(estimator=dtree, param_grid=params, cv=skf, verbose=1)
cv_score = cross_val_score(estimator=cv_dtree, X=X_train, y=y_train, cv=skf)
print(cv_score)
print(cv_score.mean())
cv_dtree.fit(X_train, y_train)
print( cv_dtree.best_params_ )
print( cv_dtree.best_estimator_ )
print( 'Best score by CV: ', cv_dtree.best_score_ )

cv_y_pred = cv_dtree.predict(X_valid)
accu2 = accuracy_score(y_valid, cv_y_pred)
print("Validation Set Accuracy after hyperparam tuning (Depth=2 to 10): %.2f%%" % (accu2 * 100.0))


#Draw the plot to show how mean accuracy is changing in regards to max_depth 
#value on cross-validation.
cv_accuracies_by_depth, validationset_accuracies_by_depth = [], []
for depth in np.arange(2, 11):
    tree = DecisionTreeClassifier(random_state=17, max_depth=depth)
    cv_score = cross_val_score(estimator=tree, X=X_train, y=y_train, cv=skf)
    print(cv_score)
    cv_accuracies_by_depth.append(cv_score.mean())
    
    tree.fit(X_train, y_train)
    pred = tree.predict(X_valid)
    validationset_accuracies_by_depth.append(accuracy_score(y_valid, pred))
    
plt.plot(np.arange(2, 11), cv_accuracies_by_depth, label='CV Accuracies',c='blue')
plt.plot(np.arange(2, 11), validationset_accuracies_by_depth, label='Validation Set Accuracies', c='orange')
plt.legend()    

#Question 4. Is there a local maximum of accuracy on the built validation curve? 
#Did GridSearchCV help to tune max_depth so that there's been at least 1% change in 
#holdout accuracy? (check out the expression (acc2 - acc1) / acc1 * 100%, where acc1 
#and acc2 are accuracies on holdout data before and after tuning max_depth with
# GridSearchCV respectively)?

print(max(cv_accuracies_by_depth)) #Yes - at depth=6 (not on the holdout curve)
print((accu2 - accu1) / accu1 * 100) #No - .58% improvement only (72.13% 50 72.55%)

#transform data as asked
#Question 5. What binary feature is the most important for heart disease detection 
#(it is placed in the root of the tree)?
q5_data = data[['age_yrs', 'ap_hi','smoke', 'gender', 'chol1', 'chol2', 'chol3']]

q5_data['age1'] = q5_data['age_yrs'].apply(lambda x: 1 if (x>=40) & (x<50) else 0)
q5_data['age2'] = q5_data['age_yrs'].apply(lambda x: 1 if (x>=50) & (x<55) else 0)
q5_data['age3'] = q5_data['age_yrs'].apply(lambda x: 1 if (x>=55) & (x<60) else 0)
q5_data['age4'] = q5_data['age_yrs'].apply(lambda x: 1 if (x>=60) & (x<65) else 0)

q5_data['bp1'] = q5_data['ap_hi'].apply(lambda x: 1 if (x>=120) & (x<140) else 0)
q5_data['bp2'] = q5_data['ap_hi'].apply(lambda x: 1 if (x>=140) & (x<160) else 0)
q5_data['bp3'] = q5_data['ap_hi'].apply(lambda x: 1 if (x>=160) & (x<180) else 0)
    
q5_data['gender'] = q5_data['gender']-1
q5_data = q5_data.drop(columns=['age_yrs', 'ap_hi'])

X_train, X_valid, y_train, y_valid = train_test_split( q5_data, y, test_size=0.3, random_state=17)
newtree = DecisionTreeClassifier( random_state = 17, max_depth=3)
newtree.fit(X_train, y_train)
features = list(q5_data.columns)
dot_data = StringIO()
export_graphviz(newtree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,
                feature_names=features)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

graph.write_png('newtree.png') #BP of 140-160 (Option 3)


