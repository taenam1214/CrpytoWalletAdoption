import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load the dataset
df = pd.read_csv('crypto_wallet_adoption_extended_data.csv')

# Display the dataset
print(df)

# Feature Engineering
df['Tech_Economic_Index'] = df['Tech_Savviness'] * df['GDP_Per_Capita']
df['Education_Urban_Index'] = df['Education_Level'] * df['Urbanization']

# Exclude non-numeric columns for correlation matrix
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Correlation matrix
corr_matrix = numeric_df.corr()
print(corr_matrix)

# Heatmap of the correlation matrix
plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix for Crypto Wallet Adoption Factors')
plt.show()

# Define features and target variable
X = df[['Internet_Penetration', 'Smartphone_Penetration', 'GDP_Per_Capita', 'Education_Level', 'Urbanization', 'Regulatory_Environment', 'Tech_Savviness', 'Financial_Inclusion', 'Tech_Economic_Index', 'Education_Urban_Index']]
y = df['Crypto_Wallet_Adoption']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Random forest model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
linear_scores = cross_val_score(linear_model, X, y, cv=kf, scoring='r2')
rf_scores = cross_val_score(rf_model, X, y, cv=kf, scoring='r2')

print(f"Linear Regression Cross-Validation R2 Scores: {linear_scores}")
print(f"Random Forest Cross-Validation R2 Scores: {rf_scores}")

# Model evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, mse, r2

linear_mae, linear_mse, linear_r2 = evaluate_model(linear_model, X_test, y_test)
rf_mae, rf_mse, rf_r2 = evaluate_model(rf_model, X_test, y_test)

print(f"Linear Regression - MAE: {linear_mae}, MSE: {linear_mse}, R2: {linear_r2}")
print(f"Random Forest - MAE: {rf_mae}, MSE: {rf_mse}, R2: {rf_r2}")

# Predictions and strategy formulation
df['Predicted_Adoption_Linear'] = linear_model.predict(X)
df['Predicted_Adoption_RF'] = rf_model.predict(X)

print(df)

# Visualizations
plt.figure(figsize=(10, 6))
plt.scatter(df['GDP_Per_Capita'], df['Crypto_Wallet_Adoption'], color='blue', label='Actual Adoption')
plt.scatter(df['GDP_Per_Capita'], df['Predicted_Adoption_Linear'], color='red', label='Predicted Adoption (Linear)')
plt.scatter(df['GDP_Per_Capita'], df['Predicted_Adoption_RF'], color='green', label='Predicted Adoption (RF)')
plt.xlabel('GDP Per Capita')
plt.ylabel('Crypto Wallet Adoption (%)')
plt.title('Actual vs Predicted Crypto Wallet Adoption')
plt.legend()
plt.show()

# Insights and Strategies
strategies = """
1. **Internet and Smartphone Penetration**: Higher internet and smartphone penetration rates are positively correlated with crypto wallet adoption. Focus on regions with growing internet and smartphone usage.
2. **Economic Factors**: GDP per capita shows a positive correlation. Tailor marketing strategies to target economically growing regions.
3. **Education Level**: Higher average years of schooling is positively correlated with adoption. Educational campaigns about cryptocurrency can be beneficial.
4. **Urbanization**: Urban areas have higher adoption rates. Focus on urban centers for initial adoption and expansion.
5. **Regulatory Environment**: More crypto-friendly regulations correlate with higher adoption. Advocate for favorable regulatory environments.
6. **Tech Savviness**: Populations with higher tech savviness show higher adoption. Promote tech literacy and cryptocurrency knowledge.
7. **Financial Inclusion**: Higher financial inclusion correlates with higher adoption. Work with financial institutions to promote crypto wallets as a part of financial inclusion efforts.
8. **Tech-Economic Index**: Higher tech savviness combined with higher GDP per capita correlates with higher adoption. Target economically strong and tech-savvy regions.
9. **Education-Urban Index**: Higher education levels combined with higher urbanization rates correlate with higher adoption. Focus on urban areas with high education levels for initial adoption campaigns.
"""

print(strategies)
