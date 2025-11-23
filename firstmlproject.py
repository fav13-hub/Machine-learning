import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error

#create a dataset manually
data=pd.DataFrame(
    {
        "Hours":[1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0],
        "Scores":[35,40,45,48,52,60,65,70,72,75,78,82,88,92]
    }
)

#split features and labels
x=data[["Hours"]] #feature
y=data["Scores"] #target label

#Train -test split
x_train, x_test, y_train, y_test = train_test_split(
    x,y,test_size=0.2, random_state=42
)

#Build and train the model
model = LinearRegression()
model.fit(x_train, y_train)

#Make predictions
pred = model.predict(x_test)
print("Predicted Scores:",pred)

#Evaluate the model
mae=mean_absolute_error(y_test,pred)
mse=mean_squared_error(y_test,pred)
rmse=mse**0.5

print("MAE:",mae)
print("MSE:",mse)
print("RMSE:",rmse)

#Predict a student's score
hours=[[5]]
prediction=model.predict(hours)
print(f"Predicted score for studying 5 hours:{prediction[0]}")

import matplotlib.pyplot as plt

plt.scatter(x, y, color="blue")
plt.plot(x, model.predict(x), color="red")
plt.title("Hours vs Scores")
plt.xlabel("Hours Studied")
plt.ylabel("Scores")
plt.show()

