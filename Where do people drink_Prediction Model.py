#!/usr/bin/env python
# coding: utf-8

# In[62]:


import requests


# In[55]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# In[72]:


url = 'https://github.com/fivethirtyeight/data/blob/master/alcohol-consumption/drinks.csv?raw=true'
df = pd.read_csv(url,index_col=0)
df.head()


# In[81]:


print('Download Starting ...')
url = 'https://raw.githubusercontent.com/dbouquin/IS_608/master/NanosatDB_munging/Countries-Continents.csv'
r = requests.get(url)
filename = url.split('/')[-1]
 
with open(filename,'wb') as output_file:
    output_file.write(r.content)
 
print('Download Completed!!!')


# In[121]:


path = 'C:/Users/QXJ/IBM/drinks.csv'
df = pd.read_csv(path)
df.head()


# In[75]:


df.columns


# In[76]:


df.dtypes


# In[77]:


df.describe(include = 'all')


# In[78]:


df.info()


# In[122]:


url = 'https://raw.githubusercontent.com/dbouquin/IS_608/master/NanosatDB_munging/Countries-Continents.csv'
df_2 = pd.read_csv(url)
df_2.head()


# In[123]:


df_2.rename(columns={"Country":"country"},inplace = True)
df_2.head()


# In[125]:


df_2.dtypes


# In[128]:


# join continent to the drink list
drink = pd.merge(df, df_2, how = 'left', on= "country")


# In[129]:


drink.head()


# In[133]:


drink.tail()


# In[132]:


missing_value = drink.isnull()
missing_value.head()


# In[140]:


# check the number of wine servings per continent
drink_group = drink[['Continent','wine_servings']]
drink_group.head()


# In[150]:


drink_continent = drink_group.groupby(['Continent']).sum()
drink_continent


# In[151]:


drink_continent.sort_values(by=['wine_servings'], ascending = False)


# In[156]:


drink_continent_beer = drink[['Continent','beer_servings']]
drink_continent_beer.head()


# In[160]:


# Perform a statistical summary and analysis of beer servings for each continent
drink_continent_beer.groupby(["Continent"]).describe()


# In[193]:


sns.set(style='whitegrid')
sns.boxplot(x='Continent', y = 'beer_servings', data = drink_continent_beer).set(title = 'Beer_servings by continents')


# In[198]:


# If the number of wine servings is negatively or positively correlated with the number of beer servings.
sns.regplot(x = drink[['wine_servings']], y= drink[['beer_servings']]).set(title = 'Wine_servings vs Beer_servings')


# In[199]:


drink.corr()


# In[205]:


sns.heatmap(drink.corr(),cmap='YlGnBu',annot = True).set(title ="correlation between different drink vs total consumption of alcohol")


# ### Compare SLR and MLR models by R^2 and MSE

# In[242]:


#Fit a linear regression model to predict
import sklearn
from sklearn.linear_model import LinearRegression


# In[243]:


lm = LinearRegression()
lm


# In[287]:


x = drink[['wine_servings']]
y = drink['total_litres_of_pure_alcohol']
lm.fit(x,y)


# In[288]:


yhat_wine = lm.predict(x)
yhat_wine[0:5]


# In[289]:


lm.score(x,y)


# In[ ]:


from sklearn.metrics import mean_squared_error


# In[290]:


mse = mean_squared_error(y,yhat_wine)
mse


# In[291]:


ax1 = sns.kdeplot(drink['total_litres_of_pure_alcohol'], color = 'r', label = 'acutal values')
sns.kdeplot(yhat_wine, color = 'b', label = 'fitted values', ax = ax1)
ax1.set_title('actual vs. fitted values for total alcohol consumption')
plt.show()


# #### The linear regression model doesn't work 

# In[292]:


x = drink[['beer_servings','spirit_servings','wine_servings']]
y = drink['total_litres_of_pure_alcohol']
lm.fit(x,y)


# In[293]:


yhat = lm.predict(x)
yhat[0:5]


# In[294]:


lm.intercept_


# In[295]:


lm.coef_


# In[258]:


lm.score(x,y)


# In[297]:


mse = mean_squared_error(y,yhat)
mse


# In[298]:


ax1 = sns.kdeplot(drink['total_litres_of_pure_alcohol'], color = 'r', label = 'acutal values')
sns.kdeplot(yhat, color = 'b', label = 'fitted values', ax = ax1)
ax1.set_title('actual vs. fitted values for total alcohol consumption')
plt.show()


# #### This multiple linear regression model is overfitting 

# In[264]:


# traing data 
x_data = drink[['beer_servings','spirit_servings','wine_servings']]
y_data = drink['total_litres_of_pure_alcohol']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_data, y_data, test_size = 0.1, random_state = 0)
print('number of test samples:', x_test.shape[0])
print('number of trating samples:',x_train.shape[0])


# In[265]:


lm_train = LinearRegression()


# In[266]:


lm_train.fit(x_train,y_train)


# In[268]:


lm_train.score(x_train,y_train)


# In[269]:


lm_train.score(x_test,y_test)


# ## Create pipeline

# In[271]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures


# In[272]:


Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]


# In[274]:


pipe = Pipeline(Input)
pipe


# In[278]:


z =  drink[['beer_servings','spirit_servings','wine_servings']]
Z = z.astype(float)
pipe.fit(Z,y)


# In[279]:


ypipe = pipe.predict(Z)
ypipe[0:4]


# In[280]:


Input=[('scale',StandardScaler()), ('model',LinearRegression())]
pipe = Pipeline(Input)


# In[281]:


pipe.fit(Z,y)


# In[282]:


ypipe = pipe.predict(Z)
ypipe[0:4]


# ### Refine the model

# #### Use polynomial transformation to test the data

# In[336]:


from sklearn. preprocessing import PolynomialFeatures


# In[337]:


lre = LinearRegression()


# In[338]:


x_data = drink[['beer_servings','spirit_servings','wine_servings']]
y_data = drink['total_litres_of_pure_alcohol']
x_train,x_test,y_train,y_test = train_test_split(x_data, y_data, test_size = 0.1, random_state = 0)
print('number of test samples:', x_test.shape[0])
print('number of trating samples:',x_train.shape[0])


# In[341]:


Rsqu_test = []
order = [1,2,3,4,5]
for n in order:
    pr = PolynomialFeatures(degree = n)
    x_train_pr = pr.fit_transform(x_train)
    x_test_pr = pr.fit_transform(x_test)
    lre.fit(x_train_pr,y_train)
    Rsqu_test.append(lre.score(x_test_pr,y_test))
    
plt.plot(order,Rsqu_test)
plt.show()


# **R^2 drops drastically after degree = 4**

# In[342]:


pr_2 = PolynomialFeatures(degree = 2)
x_train_pr2 = pr_2.fit_transform(x_train)
x_test_pr2 = pr_2.fit_transform(x_test)
pr_2


# In[343]:


poly_2 = LinearRegression()
poly_2.fit(x_train_pr2,y_train)


# In[344]:


poly_2.score(x_train_pr2,y_train)


# In[345]:


pr_4 = PolynomialFeatures(degree = 5)
x_train_pr4 = pr_4.fit_transform(x_train)
x_test_pr4 = pr_4.fit_transform(x_test)
pr_4


# In[346]:


poly_4 = LinearRegression()
poly_4.fit(x_train_pr4,y_train)
poly_4.score(x_train_pr4,y_train)


# In[353]:


yhat_poly4 = poly_4.predict(x_train_pr4)
yhat_poly4[0:4]


# In[354]:


ax1 = sns.kdeplot(y_data, color = 'r', label = 'acutal values')
sns.kdeplot(yhat_poly4, color = 'b', label = 'fitted values', ax = ax1)
ax1.set_title('actual vs. fitted values for total alcohol consumption')
plt.show()


# In[355]:


yhat_poly4_2 = poly_4.predict(x_test_pr4)
yhat_poly4_2[0:4]


# In[356]:


ax1 = sns.kdeplot(y_data, color = 'r', label = 'acutal values')
sns.kdeplot(yhat_poly4_2, color = 'b', label = 'fitted values', ax = ax1)
ax1.set_title('actual vs. fitted values for total alcohol consumption')
plt.show()


# **degree = 4 overfitting**

# In[350]:


poly_4.score(x_test_pr4,y_test)


# In[351]:


poly_2.score(x_test_pr2,y_test)


# In[357]:


yhat_poly2 = poly_2.predict(x_train_pr2)
yhat_poly4[0:4]


# In[358]:


ax1 = sns.kdeplot(y_data, color = 'r', label = 'acutal values')
sns.kdeplot(yhat_poly2, color = 'b', label = 'fitted values', ax = ax1)
ax1.set_title('actual vs. fitted values for total alcohol consumption')
plt.show()


# In[359]:


yhat_poly2_2 = poly_2.predict(x_test_pr2)
yhat_poly2_2[0:4]


# In[360]:


ax1 = sns.kdeplot(y_data, color = 'r', label = 'acutal values')
sns.kdeplot(yhat_poly2_2, color = 'b', label = 'fitted values', ax = ax1)
ax1.set_title('actual vs. fitted values for total alcohol consumption')
plt.show()


# **Both Polynomial regression seems overfitting, expecailly when degree = 4**

# ### Ridge regression

# In[361]:


from sklearn.linear_model import Ridge


# In[362]:


RigeModel = Ridge(alpha = 1)


# In[363]:


RigeModel.fit(x_train_pr2,y_train)


# In[364]:


RigeModel.score(x_test_pr2,y_test)


# In[366]:


yhat_new = RigeModel.predict(x_test_pr2)
yhat_new[0:4]


# In[369]:


print('predicted: ',yhat_new[0:4])
print('actual: ',y_test[0:4].values)


# In[370]:


from tqdm import tqdm


# In[381]:


Rsqu_test = []
Rsqu_train = []
dummy_1 = []
Alpha = 10 * np.array(range(0,10000))
pbar = tqdm(Alpha)

for alpha in pbar:
    RigeModel = Ridge(alpha = 1)
    RigeModel.fit(x_train_pr2,y_train)
    test_score, train_score = RigeModel.score(x_test_pr2,y_test), RigeModel.score(x_train_pr2,y_train)
    pbar.set_postfix({'Test score': test_score, 'Train score' : train_score})
    Rsqu_test.append(test_score)
    Rsqu_train.append(train_score)


# In[382]:


plt.plot(Alpha, Rsqu_test,label = 'Validation data')
plt.plot(Alpha,Rsqu_train,'r',label='Training data')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.legend()
plt.show()


# **The change of alpha has no effect on R^2**

# ### Conclusion: The best-fit model is Polynomial regression model with degree of 2 

# In[ ]:




