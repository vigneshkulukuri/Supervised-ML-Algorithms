import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
df=pd.read_csv("D:\\Sem 1 to 8 all\\B.Tech\\Sem 5\\INT234 -PREDICTIVE ANALYTICS\\data Sets\\50_Startups.csv")
print(df.head(5))
x=df[['R&D Spend','Administration','Marketing Spend']].iloc[:,[0,1,2]].values
y=df[['Profit']].iloc[:,-1].values
print(x)
print(y)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


model=LinearRegression()
model.fit(x_train,y_train)
y_predict=model.predict(x_test)
print(y_predict)


from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
r2=r2_score(y_test,y_predict)
print(r2)

mae=mean_absolute_error(y_test,y_predict)
print(mae)

mse=mean_squared_error(y_test,y_predict)
print(mse)

rmse=np.sqrt(mse)
print(rmse)
