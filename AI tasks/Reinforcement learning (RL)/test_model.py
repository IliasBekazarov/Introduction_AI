import pandas as pd
import joblib

# Загрузка модели
model = joblib.load('student_performance_model.pkl')

print("=== Ввод данных для предсказания оценки ===")
print("Введите значения следующих параметров:")

# Собираем данные через простые инпуты
studytime = input("Время учебы (1-4): ")
failures = input("Количество провалов: ")
Medu = input("Образование матери (0-4): ")
Fedu = input("Образование отца (0-4): ")
goout = input("Время с друзьями (1-5): ")
Dalc = input("Алкоголь в будни (1-5): ")
Walc = input("Алкоголь в выходные (1-5): ")
health = input("Здоровье (1-5): ")
absences = input("Пропуски занятий: ")
G1 = input("Оценка за 1 период (0-20): ")
G2 = input("Оценка за 2 период (0-20): ")

# Создаем DataFrame из введенных данных
test_data = {
    'studytime': [float(studytime)],
    'failures': [float(failures)],
    'Medu': [float(Medu)],
    'Fedu': [float(Fedu)],
    'goout': [float(goout)],
    'Dalc': [float(Dalc)],
    'Walc': [float(Walc)],
    'health': [float(health)],
    'absences': [float(absences)],
    'G1': [float(G1)],
    'G2': [float(G2)]
}

test_df = pd.DataFrame(test_data)

# Делаем предсказание
prediction = model.predict(test_df)
print(f"\nПредсказанная итоговая оценка (G3): {prediction[0]:.1f}")