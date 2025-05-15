import joblib

# 1. Загрузка модели и названий категорий
model = joblib.load('model.joblib.pkl')
target_names = joblib.load('target_names.pkl')

# 2. Примеры для теста
examples = [
    "NASA is launching a new satellite to explore the galaxy.",
    "The baseball team scored a home run in the final inning.",
    "New 3D graphics card improves rendering performance.",
    "One of them is not good for this job. The idea was too simple. Discovering a new place to eat."
]

# 3. Классификация
for text in examples:
    prediction = model.predict([text])[0]
    print(f"\n📄 Text: {text}")
    print(f"🧠 Predicted Category: {target_names[prediction]}")
