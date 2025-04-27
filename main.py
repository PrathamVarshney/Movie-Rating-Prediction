import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('IMDb Movies India.csv',encoding='latin1')
print(df.head())

print("\nColumn Names:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSummary Statistics:")
print(df.describe())

print("\nSample Unique Genres:", df['Genre'].dropna().unique()[:5])
print("\nSample Unique Directors:", df['Director'].dropna().unique()[:5])

df['Year'] = df['Year'].str.extract('(\d{4})') 
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')
df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')

df['Year'] = df['Year'].fillna(df['Year'].median())
df['Duration'] = df['Duration'].fillna(df['Duration'].median())
df['Rating'] = df['Rating'].fillna(df['Rating'].median())
df['Votes'] = df['Votes'].fillna(df['Votes'].median())

categorical_cols = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
for col in categorical_cols:
    df[col] = df[col].fillna('Unknown')

director_success = df.groupby('Director')['Rating'].mean()
df['Director_Success'] = df['Director'].map(director_success)
print("\nSample Director_Success feature:")
print(df[['Director', 'Director_Success']].head())


genre_avg_rating = df.groupby('Genre')['Rating'].mean()
df['Genre_Avg_Rating'] = df['Genre'].map(genre_avg_rating)
print("\nSample Genre_Avg_Rating feature:")
print(df[['Genre', 'Genre_Avg_Rating']].head())

print("\nAll Columns in Dataset:")
print(df.columns.tolist())

le = LabelEncoder()

categorical_cols = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']

for col in categorical_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))
    else:
        print(f"Warning: Column '{col}' not found in dataset.")

X = df.drop(['Name', 'Rating'], axis=1) 
y = df['Rating']  

print("Features (X) shape:", X.shape)
print("Target (y) shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

rf_y_pred = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_y_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_y_pred))
rf_r2 = r2_score(y_test, rf_y_pred)

print(f"Random Forest Mean Absolute Error (MAE): {rf_mae:.2f}")
print(f"Random Forest Root Mean Squared Error (RMSE): {rf_rmse:.2f}")
print(f"Random Forest R² Score: {rf_r2:.2f}")