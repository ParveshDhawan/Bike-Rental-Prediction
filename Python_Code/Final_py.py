
# coding: utf-8

# # Bike Rental Total Count Prediction

# ### Libraries & Data Set Import

# In[ ]:


# Importing Libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression,Ridge
from sklearn import metrics
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.svm import SVR
import os        
import numpy as np   
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


os.getcwd()


# In[ ]:


original = pd.read_csv('../Data/day.csv')
df = original.copy()
df.head()


# In[ ]:


#Info Of data (dtypes // Shape)
df.info()


# ### Exploratory Data Analysis

# In[ ]:


#Convert to proper Date type
df.dteday = pd.to_datetime(df.dteday)

#Extracting only day Sequence
df['dteday'] = df.dteday.apply(lambda x: x.day)

#Converting to proper dtype
cat_var = ['dteday','season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday','weathersit']
num_var = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered']

# #function for converting cat to num codes
for i in cat_var:
    df[i] = df[i].astype('object')


# In[ ]:


df.describe()


# In[ ]:


df.head()


# In[ ]:


#calculating all the unique values for all df columns
for i in df.columns:
    print(i,' --------> ',len(df[i].value_counts()))


# In[ ]:


# #function for converting cat to num codes
# for i in cat_var:
#     df[i] = df[i].astype('object')
    
# df = df.replace({'season':{1 : 'springer', 2 : 'summer',3 : 'fall' , 4 : 'winter'}})
# df = df.replace({'yr':{0 : '2011', 1 : '2012'}})
# df = df.replace({'holiday' : {0 : 'no', 1 : 'yes'}})
# df = df.replace({'workingday' : {0 : 'no', 1 : 'yes'}})
# df = df.replace({'weathersit' : {1:"clear",2:"mist",3:"light snow"}})

# #df[['holiday','workingday']]
# #df.loc[(df['holiday'] == 'no') & (df['workingday'] == 'no'),:]

# df.to_csv('Labeled.csv')


# ### Missing Value Analysis

# In[ ]:


df.isnull().sum()


# No Missing Values Find

# ### Data Visualization

# ###### ♦ we know variable   'cnt = casual + registered'

# In[ ]:


plt.figure(figsize=(24,16))
plt.scatter(df['instant'], df['cnt'])
plt.xlabel('Days from January,1,2011 to December,31,2012', fontsize = 20)
plt.ylabel('Count', fontsize =20)
#plt.savefig('RentCount.png')


# In[ ]:


#removing instant
df.drop('instant',axis=1,inplace=True)


# In[ ]:


#Checking distribution of data via pandas visualization
df[num_var].hist(figsize=(20,20),color='g',alpha = 0.7)
#plt.savefig('distribution.png')
plt.show()


# In[ ]:


# Total count by season & holiday
fig = plt.figure(figsize=(10,7))
fig = sns.boxplot(x='season', y='cnt',hue='holiday', data=df)
plt.xlabel('Season',fontsize = 14)
plt.ylabel('cnt',fontsize = 14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('Distribution of Total Count in particular season with effect of holiday',fontsize=15)
#plt.savefig('dist_plot.png')
plt.show()


# In[ ]:


#Bar Plot Bivariate analysis
def _barplot(x,y,df):
    ss = df.groupby([x]).sum().transpose()
    ss = round(ss)
    ax = ss.loc[y].plot(kind='bar', figsize=(15,7))
    for i in ax.patches:
        ax.annotate(str(round(i.get_height())), (i.get_x()+.1, i.get_height()/1.5),fontsize=14)
        #ax.text(i.get_x()/1.5, i.get_height()/1.5,str(round((i.get_height()))), fontsize=14)
    plt.xlabel(x,fontsize= 15)
    plt.ylabel(y,fontsize= 15)
    plt.xticks(fontsize=12,rotation = 90)
    plt.yticks(fontsize=12)
    plt.title("'{X}' wise sum of total '{Y}'".format(X=x,Y=y),fontsize = 17)
    #plt.savefig("{X}_Vs_{Y}.png".format(X=x,Y=y))
    plt.show()


# In[ ]:


#Bar Plot of CNT w.r.t categorical variable
for i in ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday','weathersit']:
    _barplot(i,'cnt',df)


# In[ ]:


# #Each weekday wheather weather or not checking distribution of total count
# for i in num_var:
#     g = sns.FacetGrid(df, col='weekday', row='weathersit', margin_titles=True)
#     ax = g.map(plt.scatter, i,"cnt")
#     #ax.savefig("{}_weekday_weathersituation_.png".format(i))
#     plt.show()


# In[ ]:


# #Each weekday wheather holiday or not checking distribution of casual count
# for i in num_var:
#     g = sns.FacetGrid(df, col='weekday', row='weathersit', margin_titles=True)
#     ax = g.map(plt.scatter, i,"casual")
#     #ax.savefig("{}_weekday_weathersituation_casual.png".format(i))
#     plt.show()


# In[ ]:


# #Each weekday weather situdtion checking distribution of registered count
# for i in ['temp', 'atemp', 'hum', 'windspeed']:
#     g = sns.FacetGrid(df, col='weekday', row='weathersit',palette="Set1",hue="holiday")
#     ax = g.map(plt.scatter, i,"registered").add_legend()
#     #ax.savefig("{}_weekday_weathersituation_registered.png".format(i))
#     plt.show()


# In[ ]:


#Joint plot of all numeric column
for i in num_var:
    fig = plt.figure(figsize=(10,7))
    fig = sns.jointplot(x=i, y="cnt", data=df,color='g')
    fig.set_axis_labels(xlabel=i,ylabel='cnt',fontsize=14)
    plt.suptitle("'{X}' and '{Y}' Scatter Plot".format(X=i,Y='cnt'),y = 1.02,fontsize=15)
    #fig.savefig("{X}_and_{Y}_Scatter_Plot.png".format(X=i,Y='cnt'))
    plt.show()


# In[ ]:


#Total count by weather situation in particular season
fig = plt.figure()
fig = sns.countplot(x="weathersit", hue="season",data=df)
#plt.savefig('figg.png')


# In[ ]:


#atemp vs temp scatter plot
fig = plt.figure()
fig = sns.jointplot(x="temp", y="atemp", data=df)
#plt.savefig('scatt.png')


# In[ ]:


#hum vs windspeed
fig = plt.figure()
sns.jointplot(x="windspeed", y="hum", data=df)
#plt.savefig('scatt_hum_windspeed.png')


# In[ ]:


# fig = plt.figure()
# fig = sns.pairplot(df,size=2.5)
# plt.show()
# # fig.savefig('pairplot.png')


# Till Now we have analyse our data very breifly 
# 
# ♦ Now Proceeding for Outliers

#  

# ### Outlier Analysis

# Data spread According to total count.
# Scatter ploot od data will give us some instution about out lier as the must farthest point or data point from entire data.
# We will consider that as an Outlier and will treat it

# In[ ]:


#Scatter plot function
def diff_scattr(x,y):
    fig = plt.figure()
    fig = sns.lmplot(x,y, data=df,fit_reg=False)
    plt.xlabel(x,fontsize= 14)
    plt.ylabel(y,fontsize= 14)
    plt.xticks(fontsize=10, rotation=90)
    plt.yticks(fontsize=10)
    plt.title("{X} and {Y} Scatter Plot".format(X=x,Y=y),fontsize = 16)
    #fig.savefig("{X}_and_{Y}_Scatter_Plot..png".format(X=x,Y=y))
    plt.show()


# In[ ]:


for i in num_var:
    diff_scattr(x=i,y='cnt')


# In[ ]:


for i in ['temp','atemp','hum','windspeed']:
    diff_scattr(x=i,y='casual')


# In[ ]:


for i in ['temp','atemp','hum','windspeed']:
    diff_scattr(x=i,y='registered')


# In[ ]:


for i in cat_var:
    diff_scattr(x=i,y='cnt')


# In[ ]:


for i in cat_var:
    diff_scattr(x=i,y='casual')


# In[ ]:


for i in cat_var:
    diff_scattr(x=i,y='registered')


# ##### ♦ There are Few outliers in our data

# In[ ]:


# #Plotting Box Plot
for i in ['temp','atemp','hum','windspeed']:
    plt.figure()
    plt.clf() #clearing the figure
    sns.boxplot(df[i],palette="Set2")
    plt.title(i)
    #plt.savefig('{}_.png'.format(i))
    plt.show()


# In[ ]:


# #Plotting Box Plot
for i in cat_var:
    plt.figure()
    plt.clf() #clearing the figure
    sns.boxplot(x=i, y="cnt", data=df)
    plt.title(('outlier w.r.t cnt & {}').format(i))
    #plt.savefig('{}_cat_box_.png'.format(i))
    plt.show()


# In[ ]:


#Treating Out Liers and Converting them to nan
for i in ['temp','atemp','hum','windspeed']:
    #print(i)
    q75, q25 = np.percentile(df.loc[:,i], [75 ,25])
    iqr = q75 - q25
    minn = q25 - (iqr*1.5)
    maxx = q75 + (iqr*1.5)
#Converting to nan
    df.loc[df.loc[:,i] < minn,i] = np.nan
    df.loc[df.loc[:,i] > maxx,i] = np.nan
    print('{var} --------- :- {X}   Missing'.format(var = i, X = (df.loc[:,i].isnull().sum())))


# In[ ]:


df[df['windspeed'].isnull() | df['hum'].isnull()]
#null value indexed = [44, 49, 68, 93, 94, 292, 382, 407, 420, 432, 433, 450, 666, 721]


# In[ ]:


df[['hum','windspeed']].describe().transpose()


# In[ ]:


#Imputing values as mean
df.windspeed = df.windspeed.fillna(df.windspeed.mean())
df.hum = df.hum.fillna(df.hum.mean())


# #### Creating Weekend Column

# In[ ]:


# end = []

# for i in df.weekday:
#     if i == 0:
#         end.append(1)
#     elif i == 6:
#         end.append(1)
#     else:
#         end.append(0)
        
# df['weekend'] = end
# df['weekend'] = df['weekend'].astype('object')
# df = df[['dteday', 'season', 'yr', 'mnth', 'holiday', 'weekday','workingday','weekend','weathersit', 'temp',\
#          'atemp', 'hum', 'windspeed','casual', 'registered', 'cnt']]


# ### Correlation Check

# In[ ]:


#Setting up the pane or matrix size
f, ax = plt.subplots(figsize=(10,8))  #Width,height

#Generating Corelation Matrix
corr = df[['temp','atemp','hum','windspeed','casual', 'registered','cnt']].corr()

#corr = df[['instant', 'dteday', 'season', 'yr', 'mnth', 'holiday', 'weekday',\
#       'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed',\
#       'casual', 'registered', 'weekend','cnt']].corr()

#Plot using Seaborn library
sns.heatmap(corr,mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220,10, as_cmap=True),            square=True, ax=ax,annot=True,linewidths=1 , linecolor= 'black',vmin = -1, vmax = 1)

plt.show()
#f.savefig('heatmap.png')


# Variable 'atemp' & 'temp' are highly correlated

# ### Chi-Square Test Among different Independent Variable

# In[ ]:


#H1 = Variables are not independent
#H0 = Variable are independent
#If p-value is less than 0.05 we will reject null hyothesis by saying alternate hypothesis is true

#from scipy.stats import chi2_contingency

#Chi Function
def chi_check(df):
    #getting all the column name as object or category
    cat_names = df.select_dtypes(exclude=np.number).columns.tolist()
    cat_pair = [(i,j) for i in cat_names for j in cat_names] #creating pairs of column
    
    p_values =[]
    for i in cat_pair:
        #print(i[0],i[1])
        if i[0] != i[1]:
            chi_result = chi2_contingency(pd.crosstab(df.loc[:,i[0]], df.loc[:,i[1]]))
            p_values.append(chi_result[1])
        else:
            p_values.append(0)
            
    chi_mat = np.array(p_values).reshape(len(cat_names),len(cat_names))
    chi_mat = pd.DataFrame(chi_mat, index = cat_names, columns = cat_names)
    return chi_mat    


# In[ ]:


chi_check(df)


# As ['holiday','workingday','weekend'] are not so independent and might cause problem so removing them
# 
# Season and mnth column are also highly related to each other

# In[ ]:


chi_check(df[['dteday','season','yr','mnth','weekday','weathersit']])


# ### Anova Test

# In[ ]:


# import statsmodels.api as sm
# from statsmodels.formula.api import ols

def one_way_anova(df,target):
    predictor_list = df.select_dtypes(exclude=np.number).columns.tolist()
    for i in predictor_list:
        mod = ols(formula=('{} ~ {}').format(target, i),data=df).fit()
        rs = sm.stats.anova_lm(mod, typ=1)
        print(('Anova p- value b/w {} and {} ----->   {}').format(target,i,rs.iloc[0][4]))


# In[ ]:


print('------------------------------------------------------------target var = CNT')
one_way_anova(df,'cnt')
print()
print('------------------------------------------------------------target var = Casual')
one_way_anova(df,'casual')
print()
print('------------------------------------------------------------target var = REGISTERED')
one_way_anova(df,'registered')


# After Anova analysation if we go with cnt as our target variable the we have to remove the variable
# 'weekday' & 'workingday' as both have pvalues more than 0.05.But if we mark casual and register as our target var then we dnt need to remove it
# ##### Also cnt = casual + registered

# ### Multicollenierity Check
# $ V.I.F. = 1 / (1 - R^2). $

# In[ ]:


from patsy import dmatrices
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# get y and X dataframes based on CNT this regression:
#y, X = dmatrices('cnt ~ + season + yr + mnth + weathersit + temp + hum + windspeed',\
#                 df, return_type='dataframe')

y, X = dmatrices('cnt ~ + dteday + season + yr + mnth + workingday + weathersit + temp + hum + windspeed',df, return_type='dataframe')

# For each X, calculate VIF and save in dataframe
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)


# In[ ]:


df.columns


# In[ ]:


#Removing Variables
#df = df.drop(['atemp','holiday','weekday'],axis=1)
df = df.drop(['atemp','holiday','workingday'],axis=1)


#  

# #### Creating Dummies of Categorical variables

# In[ ]:


# def dummy_func(df):
#     #Extracting all object var
#     cat_names = df.select_dtypes(exclude=np.number).columns.tolist()
    
#     ##Creating Dummies
#     for i in cat_names:
#         dummies = pd.get_dummies(df[i], prefix= i, dummy_na=False)
#         df = df.drop(i, 1)
#         df = pd.concat([df, dummies],axis = 1)
        
#     #Converting back to object
#     for i in df.columns:
#         if df[i].dtypes == 'uint8':
#             df[i] = df[i].astype('object')
#     return df


# In[ ]:


# #Creating Dummies
# df_dummy = dummy_func(df)
# df_dummy.shape, df.shape


# In[ ]:


# df_dummy = df_dummy[['temp','hum','windspeed','dteday_1','dteday_2','dteday_3','dteday_4','dteday_5','dteday_6','dteday_7',\
#                      'dteday_8','dteday_9','dteday_10','dteday_11','dteday_12','dteday_13','dteday_14','dteday_15',\
#                      'dteday_16','dteday_17','dteday_18','dteday_19','dteday_20','dteday_21','dteday_22','dteday_23',\
#                      'dteday_24','dteday_25','dteday_26','dteday_27','dteday_28','dteday_29','dteday_30','dteday_31',\
#                      'season_1','season_2','season_3','season_4','yr_0','yr_1','mnth_1','mnth_2','mnth_3','mnth_4',\
#                      'mnth_5','mnth_6','mnth_7','mnth_8','mnth_9','mnth_10','mnth_11','mnth_12','workingday_0',\
#                      'workingday_1','weathersit_1','weathersit_2','weathersit_3','casual','registered','cnt']]
# df_dummy.shape


# ### Feature Scaling

# In[ ]:


# df[['cnt','casual','registered']].describe().transpose()
# #cnt_min = 22.0 // cnt_max = 8714


# In[ ]:


# # #Normalization of cnt
# df['total_cnt'] = (df['cnt'] - min(df['cnt'])) / (max(df['cnt']) - min(df['cnt']))

# #Checking Normalised
# df[['cnt','total_cnt']].describe().transpose()


# ### Sampling

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,:-3],
                                                    df.iloc[:,-1],
                                                    test_size=0.20,
                                                    random_state=101)


# ### Modeling
# #####  Base Models
#      ♦ Linear Regresion
#      ♦ Decision Tree
#      ♦ Random Forest
#      ♦ Ridge

# In[ ]:


#from sklearn import metrics

# Regression
# ‘explained_variance’	metrics.explained_variance_score
# ‘neg_mean_absolute_error’	metrics.mean_absolute_error 
# ‘neg_mean_squared_error’	metrics.mean_squared_error 
# ‘neg_mean_squared_log_error’	metrics.mean_squared_log_error
# ‘neg_median_absolute_error’	metrics.median_absolute_error
# ‘r2’	metrics.r2_score

def results(y_test,y_pred):
    print('R2 score ==>  ', round(metrics.r2_score(y_test, y_pred), 2))
    print(('Mean absolute percentage error ==>  {} % ').format(round(np.mean(np.abs((y_test - y_pred) / y_test))*100, 2)))
    #print('Mean Squared Error ==>  ', round(metrics.mean_squared_error(y_test, y_pred), 2))
    print('Root Mean Squared Error ==> ', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 2))

def cross_val(model):
    acc = cross_val_score(model, X_train, y_train, cv=10,scoring='r2',n_jobs=-1,)
    print('Mean Score of Cross validation = ',round(acc.mean(),2))
    print('Standard Deviation of CV = ',round(acc.std(),2))    
    
def test_scores(model):
    print('<<<------------------- Training Data Score --------------------->')
    print()
    #Predicting result on Training data
    y_pred = model.predict(X_train)
    results(y_train,y_pred)
    print()
    print('<<<------------------- Test Data Score --------------------->')
    print()
    # Evaluating on Test Set
    y_pred = model.predict(X_test)
    results(y_test,y_pred)
    


# #### Linear Regression

# In[ ]:


#from sklearn.linear_model import LinearRegression

linear_model = LinearRegression().fit(X_train,y_train)
test_scores(linear_model)

#cross_val(linear_model)
# # Mean Score of Cross validation =  0.77
# # Standard Deviation of CV =  0.05


# In[ ]:


# # Grid Search on LR model for best Parameters
# _model = LinearRegression(normalize=True)
# pdict = [{'copy_X':[True, False],
#           'fit_intercept':[True,False]}]
# g_srch_lm = GridSearchCV(_model, param_grid = pdict, scoring='r2' , cv =10, n_jobs =-1).fit(X_train,y_train)

# #Best Score
# print('Best Score ===> ',g_srch_lm.best_score_)
# print('Best Param ===> ',g_srch_lm.best_params_)

# test_scores(g_srch_lm)


# #### Decision Tree

# In[ ]:


#from sklearn.tree import DecisionTreeRegressor

tree_model = DecisionTreeRegressor(random_state=101).fit(X_train,y_train)
test_scores(tree_model)

#cross_val(tree_model)
# # Mean Score of Cross validation =  0.73
# # Standard Deviation of CV =  0.06


# In[ ]:


# # Grid Search on Decision Tree model for best Parameters
# _model = DecisionTreeRegressor(random_state=101)
# pdict = [{'max_depth':[2,4,6,8,10,12,15],
#           'max_features':['auto','sqrt'],
#           'min_samples_leaf':[2,4,6,8,10]}]
# g_srch = GridSearchCV(_model, param_grid = pdict, scoring='r2' , cv =10, n_jobs =-1).fit(X_train,y_train)

# #Best Score
# print('Best Score ===> ',g_srch.best_score_)
# print('Best Param ===> ',g_srch.best_params_)
# test_scores(g_srch)


# #### Random Forest

# In[ ]:


#from sklearn.ensemble import RandomForestRegressor

forest_model = RandomForestRegressor(n_estimators=500,random_state=101).fit(X_train,y_train)
test_scores(forest_model)

#cross_val(forest_model)
# # Mean Score of Cross validation =  0.87
# # Standard Deviation of CV =  0.03


# In[ ]:


# # Grid Search for finding Random forest best Parameters
# _model = RandomForestRegressor(random_state=101, n_jobs=-1)
# pdict = [{'max_depth':[2,4,6,8,10],
#           'max_features':['auto','sqrt'],
#           'n_estimators': [200,300,400,500,600,700,800,1000]}]

# g_srch = GridSearchCV(_model, param_grid = pdict, cv =10, n_jobs =-1).fit(X_train,y_train)

# #Best Score
# print('Best Score ===> ',g_srch.best_score_)
# print('Best Param ===> ',g_srch.best_params_)
# test_scores(g_srch)

# #'max_depth': 10, 'max_features': 'sqrt', 'n_estimators': 500
# #{'max_depth': 15, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'n_estimators': 300}
# # Best Score ===>  0.871143807987822
# # Best Param ===>  {'max_depth': 10, 'max_features': 'sqrt', 'n_estimators': 300}


# ### SVR

# In[ ]:


#from sklearn.svm import SVR
svr_model = SVR(kernel='poly').fit(X_train,y_train)
test_scores(svr_model)

#cross_val(svr_model)
# Mean Score of Cross validation =  0.6
# Standard Deviation of CV =  0.05


# Random forest model has out perfomed out of all models used.  

# ### Final Model with Optimized Parameters :- RANDOM FOREST

# In[ ]:


#from sklearn.ensemble import RandomForestRegressor

forest_model = RandomForestRegressor(max_depth= 15, max_features = 'sqrt',n_estimators = 500,random_state=101).fit(X_train,y_train)
test_scores(forest_model)

#cross_val(forest_model)
# # Mean Score of Cross validation =  0.88
# # Standard Deviation of CV =  0.03

# #max_depth= 10, max_features = 'sqrt',n_estimators = 500
# #max_depth = 15, max_features = 'sqrt', min_samples_leaf = 2, n_estimators = 500
# #max_depth = 10, max_features = 'sqrt', n_estimators = 300


# ##### Feature Importance

# In[ ]:


#Calculating feature importances
importances = forest_model.feature_importances_

# Sort feature importances in descending order
indices = np.argsort(importances)[::1]

# Rearrange feature names so they match the sorted feature importances
names = [df.columns[i] for i in indices]

# Creating plot
fig = plt.figure(figsize=(20,20))
plt.title("Feature Importance")

# Add horizontal bars
plt.barh(range(X_train.shape[1]),importances[indices],align = 'center')
plt.yticks(range(X_train.shape[1]), names)
plt.show()
#fig.savefig('feature_importance.png')


# ### Saving Output

# In[ ]:


#Predicting Output On entire Data
pred_rf = forest_model.predict(df.iloc[:,:-3])
df['predict'] = pred_rf

#Standard result with original
entire_data = pd.concat([original,df['predict']], axis=1)


# In[ ]:


entire_data.head()


# In[ ]:


#Entire _ENV
entire_data.to_csv('../Data/output/Py_output/Entire_output.csv')
#Season
entire_data[['dteday','weathersit','season','cnt','predict']].to_csv('../Data/output/Py_output/Season_output.csv')

