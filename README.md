# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the needed packages.

2.Assigning hours to x and scores to y.

3.Plot the scatter plot.

4.Use MSE, RMSE, MAE formula to find the values.

## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: DHANUSHKUMAR SIVAKUMAR
RegisterNumber:  212224040067

```
```

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df = pd.read_csv('student_scores.csv')
df.head()
```
```
df.tail()
```
```
X = df.iloc[:,:-1].values
X
```
```
Y = df.iloc[:,1].values
Y
```
```
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred = regressor.predict(X_test)
Y_pred
```
```
Y_test
```
```
plt.scatter(X_train,Y_train,color='orange')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
```
plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
```
mse=mean_squared_error(Y_test,Y_pred)
print("MSE = ",mse)

mae = mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)

rmse = np.sqrt(mse)
print("RMSE = ",rmse)
```

## Output:


Head Values


![Screenshot 2025-05-14 085626](https://github.com/user-attachments/assets/3a33ceef-a8d7-4e07-9891-d7d5b6b7986c)

Tail Values

![Screenshot 2025-05-14 084753](https://github.com/user-attachments/assets/755263bb-5dd1-4fc2-af0a-0687f069139f)

X values

![Screenshot 2025-05-14 084803](https://github.com/user-attachments/assets/cb41225f-752c-42d9-913a-b0238e22ef5e)

Actual Y Values

![Screenshot 2025-05-14 084817](https://github.com/user-attachments/assets/5f7306d5-8a51-41c9-a494-f2e1842a7793)

Predicted Y values

![Screenshot 2025-05-14 084827](https://github.com/user-attachments/assets/383d88d7-f4c8-41e4-ab24-bf2a93221f96)

Tested Y values

![Screenshot 2025-05-14 084837](https://github.com/user-attachments/assets/f7bf6083-ddff-4b77-995e-f0c01fb17d7e)

Training Data Graph

![Screenshot 2025-05-14 084849](https://github.com/user-attachments/assets/3bb383b2-bbb1-4454-a845-1bfbbf6c4bf4)

Test Data Graph

![Screenshot 2025-05-14 084859](https://github.com/user-attachments/assets/81191dd8-4b35-4bc8-b940-b26b5e406533)

Regression Performace Metrics

![Screenshot 2025-05-14 084909](https://github.com/user-attachments/assets/8b648ba6-57bf-447a-8694-8dfca797092d)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
