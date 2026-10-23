#!/usr/bin/env python
# coding: utf-8

# In[1]:
# Install required libraries using: pip install pandas numpy matplotlib seaborn scikit-learn



# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df.head()


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.head()


# In[7]:


df.tail()


# In[8]:


df.isnull().sum()


# In[9]:


df.duplicated().sum()


# In[10]:


df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df["TotalCharges"].isnull().sum()


# In[11]:


df = df.dropna(subset=["TotalCharges"])


# In[12]:


df["TotalCharges"].isnull().sum()


# In[13]:


df["Churn"].value_counts()


# In[14]:


df["Churn"].value_counts(normalize=True) * 100


# In[15]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(data=df, x="Churn")

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.show()


# In[16]:


sns.countplot(data=df, x="Contract", hue="Churn")

plt.title("Customer Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.xticks(rotation=15)

plt.show()


# In[17]:


sns.boxplot(data=df, x="Churn", y="MonthlyCharges")

plt.title("Monthly Charges vs Churn")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")

plt.show()


# In[18]:


sns.boxplot(data=df, x="Churn", y="tenure")

plt.title("Customer Tenure vs Churn")
plt.xlabel("Churn")
plt.ylabel("Tenure (Months)")

plt.show()


# In[19]:


df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

df["Churn"].value_counts()


# In[20]:


X = df.drop(columns=["Churn", "customerID"])
y = df["Churn"]

print("Input shape:", X.shape)
print("Target shape:", y.shape)


# In[21]:


X = pd.get_dummies(X, drop_first=True, dtype=int)

X.head()


# In[22]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# In[23]:


from sklearn.linear_model import LogisticRegression


# In[24]:


model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)


# In[25]:


y_pred = model.predict(X_test)

print(y_pred[:10])


# In[26]:


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("Model Accuracy (%):", accuracy * 100)


# In[27]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


# In[28]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Stayed", "Left"]
)

display.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()


# In[29]:


from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print(classification_report(y_test, rf_pred))


# In[30]:


import pandas as pd
import matplotlib.pyplot as plt

importance = pd.Series(
    rf_model.feature_importances_,
    index=X_train.columns
)

importance = importance.sort_values(ascending=False)

print(importance.head(10))


# In[31]:


importance.head(10).sort_values().plot(
    kind="barh",
    figsize=(8, 5)
)

plt.title("Top 10 Features for Customer Churn Prediction")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()


# In[32]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

results = []

for name, predictions in [
    ("Logistic Regression", y_pred),
    ("Random Forest", rf_pred)
]:
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1 Score": f1_score(y_test, predictions, zero_division=0)
    })

results_df = pd.DataFrame(results)
results_df


# In[33]:


sample_customer = X_test.iloc[[0]]

prediction = rf_model.predict(sample_customer)[0]
probability = rf_model.predict_proba(sample_customer)[0][1]

if prediction == 1:
    print("Prediction: Customer may leave")
else:
    print("Prediction: Customer may stay")

print("Predicted churn probability:", round(probability * 100, 2), "%")


# In[ ]:




