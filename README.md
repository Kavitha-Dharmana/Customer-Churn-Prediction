# Customer Churn Prediction Using Machine Learning

**A Machine Learning Web Application to Predict Customer Churn in the Telecommunications Industry**

## About the Project

Customer Churn Prediction is a Machine Learning project that predicts whether a telecom customer is likely to continue using a service or leave it.

Customer churn is an important challenge for telecommunication companies because losing customers can affect revenue and business growth. This project analyzes customer demographics, service subscriptions, contract details, and billing information to predict potential churn.

The application uses a trained Machine Learning model and a Flask-based web interface. Users can enter customer details and receive a prediction along with an estimated churn probability.

## Project Objectives

- Predict whether a customer is likely to stay or leave.
- Analyze customer information and service usage patterns.
- Estimate the probability of customer churn.
- Apply Machine Learning to a real-world business problem.
- Demonstrate how predictive analytics can support customer retention strategies.

## Key Features

- **Customer Information Input:** Enter customer demographics, service details, contract information, and billing data.
- **Churn Prediction:** Predict whether a customer may stay or leave.
- **Churn Probability:** Display the model's estimated probability of churn.
- **Interactive Web Application:** Submit customer details through a Flask-powered interface.
- **Responsive User Interface:** Use a clean web form built with HTML and CSS.
- **Machine Learning Integration:** Load the trained model to generate predictions from user input.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Application development and model implementation |
| Pandas | Data cleaning and preprocessing |
| Scikit-learn | Model training, evaluation, and prediction |
| Joblib | Saving and loading the trained model |
| Flask | Building the web application |
| HTML | Structuring the web page |
| CSS | Styling the user interface |
| Jupyter Notebook | Data exploration and model experimentation |

## Machine Learning Algorithms

The project explores two classification algorithms:

1. **Logistic Regression** — A classification algorithm used to predict customer churn.
2. **Random Forest Classifier** — An ensemble learning algorithm used to generate predictions in the current web application.

The models are evaluated using classification accuracy to compare their performance.

## Dataset

The project uses the **Telco Customer Churn dataset**, which contains information about telecommunications customers, their subscribed services, contract types, and billing details.

Important features include:

- Customer demographics, including gender and senior citizen status
- Partner and dependent information
- Tenure and phone services
- Internet services and additional service subscriptions
- Contract type and payment method
- Monthly charges and total charges
- Customer churn status

The customer identifier (`customerID`) is excluded from model training because it identifies individual records rather than representing customer behavior.

## Project Workflow

1. **Data Collection:** Load the Telco Customer Churn dataset.
2. **Data Cleaning:** Convert `TotalCharges` into numeric values and handle missing values.
3. **Data Preprocessing:** Separate input features and the target variable, then encode categorical variables.
4. **Model Training:** Train Logistic Regression and Random Forest classification models.
5. **Model Evaluation:** Compare model performance using classification accuracy.
6. **Model Serialization:** Save the trained model and feature-column information using Joblib.
7. **Web Application Development:** Build a Flask application with an HTML form and CSS styling.
8. **Prediction:** Process user input, align features with the training columns, and generate the churn prediction and estimated probability.

## Model Performance

The following are the approximate accuracy results from the current experiments:

| Machine Learning Model | Accuracy |
|---|---:|
| Logistic Regression | 80.31% |
| Random Forest Classifier | 79.03% |

Logistic Regression achieved slightly higher accuracy in these experiments. The current web application uses the Random Forest model.

*Note: Accuracy alone does not fully measure churn prediction quality. Precision, recall, F1-score, and ROC-AUC can provide additional insights, especially when evaluating customer churn.*

## Project Structure

```text
Customer-Churn-Prediction/
│
├── Customer_Churn_Prediction.ipynb
├── Customer_Churn_Prediction.py
├── app.py
├── churn_model.pkl
├── model_columns.pkl
├── requirements.txt
├── README.md
│
├── templates/
│   └── index.html
│
└── static/
    └── style.css
```

*The structure above represents the intended project layout. The model files and `requirements.txt` must be present for the application to run as described.*

## Installation and Setup

### Prerequisites

- Python installed on your computer
- Git installed if cloning the repository
- The required Python libraries
- The trained model files

### Step 1: Clone the Repository

```bash
git clone https://github.com/Kavitha-Dharmana/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### Step 2: Install Dependencies

If `requirements.txt` is available, run:

```bash
pip install -r requirements.txt
```

### Step 3: Verify the Model Files

Ensure the following files exist in the project root directory:

- `churn_model.pkl`
- `model_columns.pkl`

These files are required by the Flask application.

### Step 4: Run the Application

```bash
python app.py
```

### Step 5: Open the Web Application

Open your browser and visit:

http://127.0.0.1:5000

Enter the customer details in the form and submit them to view the prediction.

## Example Output

Depending on the entered customer information, the application displays:

- **Prediction:** Customer may stay or Customer may leave
- **Churn Probability:** An estimated percentage generated by the model

The prediction is an estimate and does not guarantee that a customer will actually stay or leave.

## Future Enhancements

- Improve model performance through hyperparameter tuning.
- Evaluate models using precision, recall, F1-score, and ROC-AUC.
- Add visualizations to explore churn patterns and customer behavior.
- Provide customer retention recommendations.
- Deploy the application online so users can access it through a public URL.
- Improve input validation and application error handling.

## Learning Outcomes

Through this project, I gained practical experience in:

- Data cleaning and preprocessing
- Exploratory data analysis and feature engineering
- Classification model training and evaluation
- Saving and loading Machine Learning models
- Integrating a trained model with Flask
- Developing a web interface using HTML and CSS
- Applying Machine Learning to a real-world business problem

## Author

**Kavitha Dharmana**

B.Tech — Artificial Intelligence and Machine Learning

GitHub: [Kavitha-Dharmana](https://github.com/Kavitha-Dharmana)

## Disclaimer

This project is developed for educational and demonstration purposes. Predictions are based on the available dataset and trained model and should not be treated as guaranteed outcomes or as the sole basis for business decisions.

---

If you find this project interesting, feel free to explore the repository and share your feedback.
