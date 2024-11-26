import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from preprocess import trainData, validateData, testData

# Set model name
model_name = 'svm'

X_train, y_train = trainData(model_name)
X_val, y_val = validateData(model_name)
X_test, y_test = testData(model_name)

# Reduce dimensionality using PCA
pca = PCA(n_components=0.95, whiten=True, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)
X_test_pca = pca.transform(X_test)

# Train the SVM model with a polynomial kernel
svm_model = SVC(kernel='rbf',  class_weight='balanced')
svm_model.fit(X_train_pca, y_train)

# predict
y_train_predict = svm_model.predict(X_train_pca)
y_val_predict = svm_model.predict(X_val_pca)
y_test_predict = svm_model.predict(X_test_pca)

# Training Accuracy
train_accuracy = accuracy_score(y_train, y_train_predict)
print(f"SVM Training Accuracy: {train_accuracy * 100:.2f}%")

# Validation Accuracy
val_accuracy = accuracy_score(y_val, y_val_predict)
print(f"SVM Validation Accuracy: {val_accuracy * 100:.2f}%")

# Test Accuracy
test_accuracy = accuracy_score(y_test, y_test_predict)
print(f"SVM Test Accuracy: {test_accuracy * 100:.2f}%")

# Labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Test Confusion matrix
val_confusion_matrix = confusion_matrix(y_val, y_val_predict)
print('Confusion matrix for validation:')
print(val_confusion_matrix)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(val_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=emotion_labels, yticklabels=emotion_labels)
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Classification report for test data
print(classification_report(y_test, y_test_predict, zero_division=0))

# Save model
model_data = {
    'svm_model': svm_model,
    'pca': pca,
    'y_test': y_test
}
model_file = os.path.join('model', 'svm.pkl')
joblib.dump(model_data, model_file)
