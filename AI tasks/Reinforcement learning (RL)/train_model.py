import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Загрузка данных
data = pd.read_csv('student-mat.csv', sep=',')

# Выбор признаков и целевой переменной
features = ['studytime', 'failures', 'Medu', 'Fedu', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']
target = 'G3'

X = data[features]
y = data[target]

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
model = LinearRegression()
model.fit(X_train, y_train)

# Сохранение модели
joblib.dump(model, 'student_performance_model.pkl')

# Оценка модели
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Train R^2: {train_score:.3f}, Test R^2: {test_score:.3f}")