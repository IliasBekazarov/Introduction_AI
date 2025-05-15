import numpy as np
import joblib

model = joblib.load('house_price_model.pkl')
scaler = joblib.load('scaler.pkl')
features = joblib.load('features.pkl')

def predict_house_price(model):
    print("\n=== Тест-панель для предсказания цены жилья ===")
    
    while True:
        
        input_data = []
        
        for feature in features:
            value = input(f"{feature}: ")
            if value == "":  
                print("Выход из тест-панели.")
                return
            try:
                input_data.append(float(value))
            except ValueError:
                print(f"Ошибка: введите числовое значение для {feature}")
                break
        
        input_array = np.array([input_data])
        input_scaled = scaler.transform(input_array)

        prediction = model.predict(input_scaled)[0]
        print(f"\nПредсказанная стоимость жилья: ${prediction:,.2f}\n")


predict_house_price(model)