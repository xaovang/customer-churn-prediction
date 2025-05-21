import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load the data
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Show the first 5 rows
print(df.head())


# Remove missing or bad data
df = df[df['TotalCharges'] != ' ']
df['TotalCharges'] = df['TotalCharges'].astype(float)

# Convert Churn to numbers (Yes=1, No=0)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Drop customerID (not useful for training)
df = df.drop('customerID', axis=1)

# Convert all Yes/No text columns to 1/0
yes_no_columns = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in yes_no_columns:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# One-hot encode other categorical columns (like gender, contract type, etc.)
df = pd.get_dummies(df, drop_first=True)

# Check the cleaned data
print(df.head())



# Split into features (X) and target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Train/test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# redict on the test set
predictions = model.predict(X_test)

# rint results
print("✅ Accuracy:", accuracy_score(y_test, predictions))
print("\n📊 Classification Report:")
print(classification_report(y_test, predictions))



# Get feature importance
importances = model.feature_importances_
features = X.columns

# Put into a DataFrame
feat_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feat_df = feat_df.sort_values(by='Importance', ascending=False).head(10)

# Plot it
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_df)
plt.title('Top 10 Features Influencing Churn')
plt.show()
