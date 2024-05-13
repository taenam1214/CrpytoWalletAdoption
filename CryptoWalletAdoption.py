import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('crypto_wallet_adoption_extended_data.csv')

# Display the dataset
print(df)

# Exclude non-numeric columns for correlation matrix
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Correlation matrix
corr_matrix = numeric_df.corr()
print(corr_matrix)

# Heatmap of the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix for Crypto Wallet Adoption Factors')
plt.show()

# Define features and target variable
X = df[['Internet_Penetration', 'Smartphone_Penetration', 'GDP_Per_Capita', 'Education_Level', 'Urbanization', 'Regulatory_Environment', 'Tech_Savviness', 'Financial_Inclusion']]
y = df['Crypto_Wallet_Adoption']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Coefficients
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)

# Predictions
y_pred = model.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Predictions and strategy formulation
df['Predicted_Adoption'] = model.predict(X)

print(df)

# Insights and Strategies
strategies = """
1. **Internet and Smartphone Penetration**: Higher internet and smartphone penetration rates are positively correlated with crypto wallet adoption. Focus on regions with growing internet and smartphone usage.
2. **Economic Factors**: GDP per capita shows a positive correlation. Tailor marketing strategies to target economically growing regions.
3. **Education Level**: Higher average years of schooling is positively correlated with adoption. Educational campaigns about cryptocurrency can be beneficial.
4. **Urbanization**: Urban areas have higher adoption rates. Focus on urban centers for initial adoption and expansion.
5. **Regulatory Environment**: More crypto-friendly regulations correlate with higher adoption. Advocate for favorable regulatory environments.
6. **Tech Savviness**: Populations with higher tech savviness show higher adoption. Promote tech literacy and cryptocurrency knowledge.
7. **Financial Inclusion**: Higher financial inclusion correlates with higher adoption. Work with financial institutions to promote crypto wallets as a part of financial inclusion efforts.
"""

print(strategies)
