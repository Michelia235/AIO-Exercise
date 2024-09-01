# Import library
import numpy as np
from datasets import load_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

# Load IMDB dataset
imdb = load_dataset("imdb")
imdb_train, imdb_test = imdb['train'], imdb['test']

# Convert text to vector using BoW
vectorizer = CountVectorizer(max_features=1000)
X_train = vectorizer.fit_transform(imdb_train['text']).toarray()  # Chuyển đổi văn bản huấn luyện thành vector
X_test = vectorizer.transform(imdb_test['text']).toarray()  # Chuyển đổi văn bản kiểm tra thành vector
y_train = np.array(imdb_train['label'])  # Chuyển đổi nhãn huấn luyện thành mảng NumPy
y_test = np.array(imdb_test['label'])  # Chuyển đổi nhãn kiểm tra thành mảng NumPy

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Chuẩn hóa đặc trưng của tập huấn luyện
X_test = scaler.transform(X_test)  # Chuẩn hóa đặc trưng của tập kiểm tra

# Build KNN Classifier
knn_classifier = KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree')
knn_classifier.fit(X_train, y_train)  # Huấn luyện mô hình phân loại KNN

# Predict test set and evaluate
y_pred = knn_classifier.predict(X_test)  # Dự đoán nhãn cho tập kiểm tra
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')  # Đánh giá độ chính xác của mô hình
