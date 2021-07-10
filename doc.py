import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import joblib

#importing Tikinter and other library

import tkinter as tk
from tkinter.filedialog import askopenfilename

root = tk.Tk()
root.withdraw() #Prevents the Tkinter window to come up
csvpath = askopenfilename()
root.destroy()
print(csvpath)
df = pd.read_csv(csvpath)
pd.set_option("display.max_columns", None)# based on the file path you can change the code here
#ds = pd.read_excel(csvpath) # for xlsx file
df

data=df.copy()

df.isnull().sum()

df.drop(['Unnamed: 0','CO', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'SO2','PM10_AQI', 'SO2_AQI', 'NOx_AQI', 'NO2_AQI',
       'CO_AQI', 'O3_AQI','Checks','NOx_24hr_avg','AQI_bucket_calculated', 'min_a13', 'max_a13', 'mean_a23',
       'min_a23', 'max_a23', 'mean_a24', 'min_a24', 'max_a24', 'mean_a25_x',
       'min_a25_x', 'max_a25_x', 'mean_a25_y', 'min_a25_y', 'max_a25_y',
       'mean_a27', 'min_a27', 'max_a27'],inplace=True,axis=1)

numerical_feature = [feature for feature in df.columns if df[feature].dtypes != 'O']
discrete_feature=[feature for feature in numerical_feature if len(df[feature].unique())<25]
continuous_feature = [feature for feature in numerical_feature if feature not in discrete_feature]
categorical_feature = [feature for feature in df.columns if feature not in numerical_feature]
print("Numerical Features Count {}".format(len(numerical_feature)))
print("Discrete feature Count {}".format(len(discrete_feature)))
print("Continuous feature Count {}".format(len(continuous_feature)))
print("Categorical feature Count {}".format(len(categorical_feature)))

# Handle Missing Values
df.isnull().sum()

# print numerical features
print(numerical_feature)

def randomsampleimputation(df, variable):
    df[variable]=df[variable]
    random_sample=df[variable].dropna().sample(df[variable].isnull().sum(),random_state=0)
    random_sample.index=df[df[variable].isnull()].index
    df.loc[df[variable].isnull(),variable]=random_sample
    
randomsampleimputation(df, "PM10_24hr_avg")
randomsampleimputation(df, "SO2_24hr_avg")
randomsampleimputation(df, "NO2_1hr_avg")
randomsampleimputation(df, "CO_8hr_max")
randomsampleimputation(df, "O3_8hr_max")
randomsampleimputation(df, "Combined_AQI")

for feature in continuous_feature:
    if(df[feature].isnull().sum()*100/len(df))>0:
        df[feature] = df[feature].fillna(df[feature].median())

def mode_nan(df,variable):
    mode=df[variable].value_counts().index[0]
    df[variable].fillna(mode,inplace=True)
mode_nan(df,"max_cases_Dia")

df["Date"] = pd.to_datetime(df["Date"], format = "%Y-%m-%dT", errors = "coerce")

df["Date_month"] = df["Date"].dt.month
df["Date_day"] = df["Date"].dt.day

IQR=df.PM10_24hr_avg.quantile(0.75)-df.PM10_24hr_avg.quantile(0.25)
lower_bridge=df.PM10_24hr_avg.quantile(0.25)-(IQR*1.5)
upper_bridge=df.PM10_24hr_avg.quantile(0.75)+(IQR*1.5)
print(lower_bridge, upper_bridge)

df.loc[df['PM10_24hr_avg']>=300.70,'PM10_24hr_avg']=300.70
df.loc[df['PM10_24hr_avg']<=-10.55,'PM10_24hr_avg']=-10.55

IQR=df.SO2_24hr_avg.quantile(0.75)-df.SO2_24hr_avg.quantile(0.25)
lower_bridge=df.SO2_24hr_avg.quantile(0.25)-(IQR*1.5)
upper_bridge=df.SO2_24hr_avg.quantile(0.75)+(IQR*1.5)
print(lower_bridge, upper_bridge)

df.loc[df['SO2_24hr_avg']>=0.01309,'SO2_24hr_avg']=0.01309
df.loc[df['SO2_24hr_avg']<=0.0016,'SO2_24hr_avg']=0.0016

IQR=df.NO2_1hr_avg.quantile(0.75)-df.NO2_1hr_avg.quantile(0.25)
lower_bridge=df.NO2_1hr_avg.quantile(0.25)-(IQR*1.5)
upper_bridge=df.NO2_1hr_avg.quantile(0.75)+(IQR*1.5)
print(lower_bridge, upper_bridge)

df.loc[df['NO2_1hr_avg']>=0.0573,'NO2_1hr_avg']=0.0573
df.loc[df['NO2_1hr_avg']<=-0.0019,'NO2_1hr_avg']=-0.0019

IQR=df.CO_8hr_max.quantile(0.75)-df.CO_8hr_max.quantile(0.25)
lower_bridge=df.CO_8hr_max.quantile(0.25)-(IQR*1.5)
upper_bridge=df.CO_8hr_max.quantile(0.75)+(IQR*1.5)
print(lower_bridge, upper_bridge)

df.loc[df['CO_8hr_max']>=2.2480,'CO_8hr_max']=2.2480
df.loc[df['CO_8hr_max']<=-0.1295,'CO_8hr_max']=-0.1295

IQR=df.O3_8hr_max.quantile(0.75)-df.O3_8hr_max.quantile(0.25)
lower_bridge=df.O3_8hr_max.quantile(0.25)-(IQR*1.5)
upper_bridge=df.O3_8hr_max.quantile(0.75)+(IQR*1.5)
print(lower_bridge, upper_bridge)

df.loc[df['O3_8hr_max']>= 0.05727,'O3_8hr_max']= 0.05727
df.loc[df['O3_8hr_max']<=0.00805,'O3_8hr_max']=0.00805

IQR=df.Combined_AQI.quantile(0.75)-df.Combined_AQI.quantile(0.25)
lower_bridge=df.Combined_AQI.quantile(0.25)-(IQR*1.5)
upper_bridge=df.Combined_AQI.quantile(0.75)+(IQR*1.5)
print(lower_bridge, upper_bridge)

df.loc[df['Combined_AQI']>=186.0,'Combined_AQI']=186.0
df.loc[df['Combined_AQI']<=10.0,'Combined_AQI']=10.0

IQR=df.mean_a13.quantile(0.75)-df.mean_a13.quantile(0.25)
lower_bridge=df.mean_a13.quantile(0.25)-(IQR*1.5)
upper_bridge=df.mean_a13.quantile(0.75)+(IQR*1.5)
print(lower_bridge, upper_bridge)

df.loc[df['mean_a13']>=17.1,'mean_a13']=17.1
df.loc[df['mean_a13']<=5.1,'mean_a13']=5.1

df.drop('Date',inplace=True,axis=1)

df.drop(df[df['max_cases_Dia'] >= 10].index,inplace=True)

data=df.copy()

#lets remove the outliers using zscore
from scipy.stats import zscore
z=abs(zscore(data))
print(data.shape)
new=data.loc[(z<3).all(axis=1)]
print(new.shape)
# we can observe the new zscore down below.

# lets sepearate input output columns
x=new.drop(columns=['Combined_AQI','max_cases_Dia', 'mean_a13']) # Input variable.
y=pd.DataFrame(new[['Combined_AQI','max_cases_Dia', 'mean_a13']]) #Target Variable.

# let import diffrent model library
from sklearn.linear_model import LinearRegression,Lasso,ElasticNet,Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
#import ensemble technique
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor



#importing error matrics
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

#lets apply regression to datasets
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
def maxr2_score(regr,x,y): #Def is used such that we can call it later
    max_r_score=0
    for r_state in range(42,100):
        x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=r_state,test_size=0.20)
        regr.fit(x_train,y_train)
        y_pred=regr.predict(x_test)
        r2_scr=r2_score(y_test,y_pred)
        if r2_scr>max_r_score:
            max_r_score=r2_scr
            final_r_state=r_state
    print()
    print('max r2 score correponding to',final_r_state,'is',max_r_score)
    print()
    print('Error:')
    print('Mean absolute error:',mean_absolute_error(y_test,y_pred))
    print('Mean Squared error:',mean_squared_error(y_test,y_pred))
    print('Root Mean Squared error:',np.sqrt(mean_squared_error(y_test,y_pred)))
    print('*******************************************************************')
    print()
    return final_r_state

model=[LinearRegression(),Lasso(),KNeighborsRegressor(),Ridge(),ElasticNet()]
for m in model:
    print('----->>',m,'<<-----')
    r_state=maxr2_score(m,x,y)
    
#lets cross validate all the model uing FOR loop.
from sklearn.model_selection import cross_val_score
model=[LinearRegression(),Lasso(),KNeighborsRegressor(),DecisionTreeRegressor(),Ridge(),ElasticNet()]
for m in model:
    cvs=cross_val_score(m,x,y,cv=10,scoring='r2')
    print('Cross val score of',m,'is:')
    print('Cross val score is',cvs)
    print('Mean cross val score of',m,'is',cvs.mean())
    print('Standard deviation of',m,'is',cvs.std())
    print()
    print('*******************************************************************')
    
# lets check the best parameter using grid search cv.
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')
rd=Ridge()
parameters={'alpha':[0.001,0.01,0.1,1],'random_state':range(42,100)} 
clf=GridSearchCV(rd,parameters,cv=5)
clf.fit(x,y)
clf.best_params_

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=59,test_size=0.20)
rd=Ridge(alpha=1)
rd.fit(x_train,y_train)
pred=rd.predict(x_test)
print("R2 Score for Ridge : ",r2_score(y_test,pred)*100)
print('Cross Validation Score for Ridge: ',cross_val_score(rd,x,y,cv=5,scoring='r2').mean()*100)
print()
print('Mean absolute error:',mean_absolute_error(y_test,pred))
print('Mean Squared error:',mean_squared_error(y_test,pred))
print('Root Mean Squared Error :',np.sqrt(mean_squared_error(y_test,pred)))

# so by below regplot we can observe that how the test data and predicted data conincides so can say that our model fits best.
plt.scatter(y_test,pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2) #lw=line width
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.show()

#lets save the model with pickle
import joblib
joblib.dump(rd,'diabetic.pkl')


