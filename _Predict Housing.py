#!/usr/bin/env python
# coding: utf-8

# # Project: Predict Housing Prices

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ##### 1.CRIM: per capita crime rate by town
# ##### 2.ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
# ##### 3.INDUS: proportion of non-retail business acres per town
# ##### 4.CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
# ##### 5.NOX: nitric oxides concentration (parts per 10 million)
# ##### 6.RM: average number of rooms per dwelling
# ##### 7.AGE: proportion of owner-occupied units built prior to 1940
# ##### 8.DIS: weighted distances to five Boston employment centres
# ##### 9.RAD: index of accessibility to radial highways
# ##### 10.TAX: full-value property-tax rate per 10,000
# ##### 11.PTRATIO: pupilteacher ratio by town
# ##### 12.B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
# ##### 13.LSTAT:  lower status of the population
# ##### 14.MEDV: Median value of owner-occupied homes in 1000's

# # Load Boston Dataset
# 

# In[2]:


from sklearn.datasets import load_boston
boston = load_boston()
bos = pd.DataFrame(boston.data)
bos.columns = boston.feature_names
bos['PRICE'] = boston.target
bos.head()


# # Check for any missing values
# 

# In[3]:


bos.isnull().sum()


# # Data Exploration

# In[4]:


X=bos.drop('PRICE',axis=1)
y=bos['PRICE']


# In[5]:


X


# In[6]:


#  Minimum price of the data
minimum_price = np.array(y).min()

#  Maximum price of the data
maximum_price = np.array(y).max()

#  Mean price of the data
mean_price = np.array(y).mean()

# Median price of the data
median_price  =np.median(y)

#  Standard deviation of prices of the data
std_price = np.array(y).std()

# Show the calculated statistics
print("Statistics for Boston housing dataset:\n")
print("Minimum price: ${}".format(minimum_price)) 
print("Maximum price: ${}".format(maximum_price))
print("Mean price: ${}".format(mean_price))
print("Median price ${}".format(median_price))
print("Standard deviation of prices: ${}".format(std_price))


# In[7]:


plt.scatter(bos['CRIM'],y)


# In[8]:


plt.scatter(bos['LSTAT'],y)


# # Splitting our data into training and testing 

# In[9]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, make_scorer
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Import 'make_scorer', 'DecisionTreeRegressor', and 'GridSearchCV'

# In[10]:


from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
def fit_model(X, y):
    
    cv_sets=ShuffleSplit(n_splits=10, random_state=0, test_size=0.2, train_size=None)
    
    regressor = DecisionTreeRegressor(random_state=0)

    
    params = {'max_depth':[1,2,3,4,5,6,7,8,9,10],'min_samples_leaf':[1,2,3,4,5,6.5,7,8,9,10], 'min_samples_split':[1,2,3,4,5,6,7,8,9,10]} 
    scoring_fnc = make_scorer(r2_score)
    grid = GridSearchCV(estimator=regressor, param_grid= params, scoring=scoring_fnc,cv=cv_sets)
    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)
    # Return the optimal model after fitting the data
    return grid.best_estimator_


# In[11]:


# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)
y_pr = reg.predict(X_test)
y_pred = reg.predict(X_train)


# In[12]:


# Produce the value for 'max_depth'
print("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']))
print("Parameter 'min_samples_leaf' is {} for the optimal model.".format(reg.get_params()['min_samples_leaf']))
print("Parameter 'min_samples_split' is {} for the optimal model.".format(reg.get_params()['min_samples_split']))


# In[13]:


R2_for_train = r2_score(y_train, y_pred)
R2_for_test = r2_score(y_test, y_pr)


# ## show R2_for_train and R2_for_test

# In[14]:


print('R2_for_train',R2_for_train)
print('R2_for_test',R2_for_test)


# ### I found that using decision tree is a weak model, so I used another model (adaboost)

# ## adaboost

# In[15]:


from sklearn.ensemble import AdaBoostRegressor
model = AdaBoostRegressor(learning_rate=0.1,n_estimators=500,
                                  base_estimator = DecisionTreeRegressor(max_depth=7,min_samples_leaf=7),
                                  random_state=0)
model.fit(X_train,y_train)


# In[16]:


y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
from sklearn.metrics import r2_score
train_accuracy =r2_score(y_train, y_train_pred)
test_accuracy = r2_score(y_test, y_test_pred)
print('The training r2 is', train_accuracy)
print('The test r2 is', test_accuracy)


# In[ ]:




