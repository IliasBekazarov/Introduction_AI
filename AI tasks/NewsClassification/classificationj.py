from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib

# 1. Загружаем данные (выбираем 4 категории)
categories = ['sci.space', 'rec.sport.baseball', 'comp.graphics', 'talk.politics.misc']
data = fetch_20newsgroups(subset='train', categories=categories)

# 2. Создаём модель: TF-IDF + Наивный Байес
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# 3. Делим данные
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=42)

# 4. Обучаем модель
model.fit(X_train, y_train)

# 5. Проверка точности
predicted = model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, predicted)
print(f"✅ Accuracy: {accuracy:.2f}")

# 6. Сохраняем модель и названия категорий
joblib.dump(model, 'model.joblib.pkl')
joblib.dump(data.target_names, 'target_names.pkl')
