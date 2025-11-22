🧠 Student Score Prediction Using Linear Regression

This project demonstrates a simple machine learning model that predicts a student’s exam score based on the number of hours they study. It uses Simple Linear Regression, one of the most fundamental algorithms for understanding relationships between variables.

📌 Project Overview

The goal of this project is to explore how study time affects academic performance.
The model is trained on a small dataset containing:

Hours Studied → Independent Variable

Exam Score → Dependent Variable

After training, the model can predict a student’s score based on the number of hours they have studied.

🔧 Technologies Used

Python 3.x

Pandas – data manipulation

Scikit-learn – machine learning

VS Code / Jupyter Notebook (optional)

📊 Dataset

The dataset contains manually created values representing hours studied and exam scores.

Hours	Score
2.5	48
5.0	72
7.5	92

The full dataset is included in the project script.

🧪 How the Model Works

Load and prepare the dataset

Split data into training and testing sets

Train the Linear Regression model

Evaluate performance using:

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)

Predict a new student’s score based on study hours

🚀 Code Example
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train, y_train)

hours = [[5]]
prediction = model.predict(hours)
print(f"Predicted score for studying 5 hours: {prediction[0]}")

📈 Model Evaluation

The following performance metrics are used to measure accuracy:

MAE – average prediction error

MSE – squared prediction error

RMSE – standard deviation of errors

These metrics give insight into how well the model performs on unseen data.

🎯 Results

The model provides a reasonable prediction of exam performance.
Example output:

Studying 5 hours → Predicted Score ≈ 75–80

(Exact values depend on the dataset split.)

📂 Project Structure
├── score_prediction.py
├── README.md
└── requirements.txt (optional)

🛠️ How to Run the Project

Clone the repository:

git clone https://github.com/fav13-hub/Machine-learning.git


Install dependencies:

pip install -r requirements.txt


Run the script:

python score_prediction.py

🤝 Contribution

Contributions are welcome!
You can fork the repository, make improvements, and submit a pull request.

📜 License

This project is licensed under the MIT License.
