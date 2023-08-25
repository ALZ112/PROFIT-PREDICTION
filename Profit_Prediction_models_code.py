
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as pt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet # Elastic net regression
from sklearn.svm import SVR
import xgboost as xgb
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Data reading

data=pd.read_csv('C:\\Users\\Amogh Prabhu\\Desktop\\ML\\PROFIT PREDICTION\\50_startups.csv')
# Data preprocessing and Visulization

print(data)
print(data.info())
print(data.columns)
# correlation matrix
co_matrix=data[['R&D Spend', 'Administration', 'Marketing Spend', 'Profit']].corr()
sb.heatmap(co_matrix,annot=True,cmap='coolwarm')
pt.title('Correlation Matrix')
pt.show()

#pair plot
sb.pairplot(data,x_vars=['R&D Spend', 'Administration', 'Marketing Spend'],y_vars='Profit',kind='scatter')
pt.show()


# regression plots
sb.lmplot(x = 'R&D Spend',y='Profit',data=data)
sb.lmplot(x = 'Administration',y='Profit',data=data)
sb.lmplot(x = 'Marketing Spend',y='Profit',data=data)
pt.show()


# Distribution plots
sb.distplot(data['Profit'],color='maroon')
pt.show()



# Histogram plot
sb.histplot(data['Profit'],color='seagreen')


# models training ,testing and evalvation


y=data['Profit']
X=data[['R&D Spend', 'Administration', 'Marketing Spend']]


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=42)


# Linear regression
linear_reg = LinearRegression() # instance of class 
linear_reg.fit(X_train,y_train)
linear_reg_pred = linear_reg.predict(X_test)
linear_reg_r2 = r2_score(y_test,linear_reg_pred) # coefficient of determination
linear_reg_mse=mean_squared_error(y_test,linear_reg_pred) #MSE
linear_reg_mae=mean_absolute_error(y_test,linear_reg_pred) #MAE


# random forest regression

randfor_reg = RandomForestRegressor()
randfor_reg.fit(X_train,y_train)
randfor_reg_pred=randfor_reg.predict(X_test)
randfor_reg_r2=r2_score(y_test,randfor_reg_pred)
randfor_reg_mse=mean_squared_error(y_test,randfor_reg_pred)
randfor_reg_mae=mean_absolute_error(y_test,randfor_reg_pred)

# Decision Tree
dectree_reg = DecisionTreeRegressor()
dectree_reg.fit(X_train,y_train)
dectree_reg_pred = dectree_reg.predict(X_test)
dectree_reg_r2=r2_score(y_test,dectree_reg_pred)
dectree_reg_mse=mean_squared_error(y_test,dectree_reg_pred)
dectree_reg_mae = mean_absolute_error(y_test,dectree_reg_pred)

# Lasso regression model
# Error Term E=square residulals + alpha(abs(slope1)+abs(slope2)+abs(slope3)+..............)
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X_train,y_train)
lasso_reg_pred=lasso_reg.predict(X_test)
lasso_reg_r2=r2_score(y_test,lasso_reg_pred)
lasso_reg_mse=mean_squared_error(y_test,lasso_reg_pred)
lasso_reg_mae=mean_absolute_error(y_test,lasso_reg_pred)

# Ridge regression model

# Error Term E=square residulals + alpha(abs(slope1)^2+abs(slope2)^2+abs(slope3)^2+..............)
ridge_reg=Ridge(alpha = 0.1)
ridge_reg.fit(X_train,y_train)
ridge_reg_pred=ridge_reg.predict(X_test)
ridge_reg_r2=r2_score(y_test,ridge_reg_pred)
ridge_reg_mse=mean_squared_error(y_test,ridge_reg_pred)
ridge_reg_mae=mean_absolute_error(y_test,ridge_reg_pred)

# Elastic Net regression

# Error Term E=square residulals + alpha((l1_ratio(abs(slope1)+abs(slope2)+abs(slope3)+..............) + (1-l1_ratio)(abs(slope1)^2+abs(slope2)^2+abs(slope3)^2+..............)
elasticnet_reg=ElasticNet(alpha=0.1,l1_ratio=0.5) # lasso+ ridge
elasticnet_reg.fit(X_train,y_train)
elasticnet_reg_pred=elasticnet_reg.predict(X_test)
elasticnet_reg_r2=r2_score(y_test,elasticnet_reg_pred)
elasticnet_reg_mse=mean_squared_error(y_test,elasticnet_reg_pred)
elasticnet_reg_mae=mean_absolute_error(y_test,elasticnet_reg_pred)

# support vector regression
svr=SVR(kernel = 'linear')
svr.fit(X_train,y_train)
svr_pred=svr.predict(X_test)
svr_r2=r2_score(y_test,svr_pred)
svr_mse=mean_squared_error(y_test,svr_pred)
svr_mae=mean_absolute_error(y_test,svr_pred)

#XGBoost

xgb_reg=xgb.XGBRegressor()
xgb_reg.fit(X_train,y_train)
xgb_reg_pred=xgb_reg.predict(X_test)
xgb_reg_r2=r2_score(y_test,xgb_reg_pred)
xgb_reg_mse=mean_squared_error(y_test,xgb_reg_pred)
xgb_reg_mae=mean_absolute_error(y_test,xgb_reg_pred)

#Feedforward Neural Networks
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Feedforward Neural Network model

fnn_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32,activation='relu',input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dense(1)
])
fnn_model.compile(optimizer= 'adam' , loss = 'mean_squared_error')
fnn_model.fit(X_train_scaled,y_train,epochs = 100,batch_size = 32)
fnn_predict = fnn_model.predict(X_test_scaled)
fnn_r2 = r2_score(y_test,fnn_predict)
fnn_mse= mean_squared_error(y_test,fnn_predict)
fnn_mae = mean_absolute_error(y_test,fnn_predict)

# finding BEST MODEL
result = pd.DataFrame({
    'Model' : ['Linear' ,'Lasso' , 'Ridge', 'Elasticnet', 'Random Forest', 'Decision Tree' , 'SVR','XGBoost', 'FNN'],
    'R2 Score' : [linear_reg_r2,lasso_reg_r2,ridge_reg_r2,elasticnet_reg_r2,randfor_reg_r2,dectree_reg_r2,svr_r2,xgb_reg_r2,fnn_r2],
     'MSE' : [linear_reg_mse,lasso_reg_mse,ridge_reg_mse,elasticnet_reg_mse,randfor_reg_mse,dectree_reg_mse,svr_mse,xgb_reg_mse,fnn_mse],   
     'MAE' : [linear_reg_mae,lasso_reg_mae,ridge_reg_mae,elasticnet_reg_mae,randfor_reg_mae,dectree_reg_mae,svr_mae,xgb_reg_mae,fnn_mae],  
})

print(result)

result['MSE Rank'] = result['MSE'].rank(ascending=True , method = 'min')
result['MAE Rank'] = result['MAE'].rank(ascending=True , method = 'min')
result['R2 Rank']  = result['R2 Score'].rank(ascending=False, method = 'min')
result['Total Rank'] = result['MSE Rank']+result['MAE Rank']+result['R2 Rank']
result_sorted_rank = result.sort_values('Total Rank')
best_model_rank=result_sorted_rank.iloc[0]['Model']
print("BEST model Based on Total Rank: " , best_model_rank)


# sample prediction

rd_spend = float(input("Enter R & D spend : "))
admin = float(input("Enter Administration Cost: "))
market= float(input("Enter Markenting Expenditure :"))
# we will select based on best model
print("The predicted value of startup is:",float(lasso_reg.predict([[rd_spend,admin,market]])))
print("The predicted value of startup is:",float(randfor_reg.predict([[rd_spend,admin,market]])))
