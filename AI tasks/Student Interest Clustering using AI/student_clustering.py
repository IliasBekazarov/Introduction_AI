import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. Маалыматты жүктөө
df = pd.read_csv("data.csv")

# 2. TF-IDF менен текстти санга айлантуу
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Interest'])

# 3. Кластерлөө (мисалы, 3 кластер)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# 4. PCA аркылуу 2D координаттарга кыскартуу
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(X.toarray())

# 5. Визуализация
plt.figure(figsize=(10, 6))
colors = ['red', 'green', 'blue']
for i in range(3):
    points = reduced_data[df['Cluster'] == i]
    plt.scatter(points[:, 0], points[:, 1], c=colors[i], label=f'Cluster {i}')

plt.title('🧠 TF-IDF Кластерлөө Визуализациясы')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.grid(True)
plt.show()

# 6. Терминалда кластер жыйынтыгын көрсөтүү
print("\n🔍 Жалпы Кластерлөө Жыйынтыгы:")
print(df[['Name', 'Interest', 'Cluster']].to_string(index=False))

# 7. Ар бир кластер боюнча топторду өзүнчө чыгаруу
print("\n📊 Кластерлер боюнча бөлүнгөн студенттер:")

for i in range(3):
    print(f"\n🔹 Cluster {i}:")
    cluster_df = df[df['Cluster'] == i]
    for _, row in cluster_df.iterrows():
        print(f"👤 {row['Name']} — {row['Interest']}")

