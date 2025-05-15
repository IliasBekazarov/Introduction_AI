import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv("housing.csv")

df['total_bedrooms'].fillna(df['tota l_bedrooms'].mean(), inplace=True)

features = ['total_rooms', 'total_bedrooms'] 
X = df[features]
y = df['median_house_value']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Среднеквадратичная ошибка: {mse:.2f}')
print(f'Коэффициент объясненной дисперсии (R²): {r2:.2f}')

joblib.dump(model, 'house_price_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(features, 'features.pkl')

print("Модель, масштабировщик и список признаков сохранены в файлы: 'house_price_model.pkl', 'scaler.pkl', 'features.pkl'")
