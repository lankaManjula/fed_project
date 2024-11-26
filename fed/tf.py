import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    GlobalAveragePooling2D,
    Dense,
    Dropout
)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
from preprocess import trainData, validateData, testData

# Set model name
model_name = "tf"

# Load the data
X_train, y_train = trainData(model_name)
X_val, y_val = validateData(model_name)
X_test, y_test = testData(model_name)

# One-hot encode for y values
y_train_encoded = to_categorical(y_train)
y_val_encoded = to_categorical(y_val)
y_test_encoded = to_categorical(y_test)

# Print shapes
print(
    X_train.shape,
    y_train_encoded.shape,
    X_val.shape,
    y_val_encoded.shape,
    X_test.shape,
    y_test_encoded.shape
)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# Create Model
def createModel(num_classes=7):
    base_model = MobileNetV2(input_shape=(64, 64, 3), include_top=False, weights="imagenet")
    model = Sequential(
        [
            Input(shape=(64, 64, 3)),
            Conv2D(3, kernel_size=(3, 3), padding="same"),  # Convert grayscale to RGB
            base_model,
            GlobalAveragePooling2D(),
            Dropout(0.3),
            Dense(256, activation="relu"),
            Dropout(0.3),
            Dense(num_classes, activation="softmax")
        ]
    )
    return model

# Initialize model
model = createModel()

lr_schedule = CosineDecay(initial_learning_rate=0.0001, decay_steps=50000)
optimizer = Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# Early stopping and reduce LR
early_stopping = EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)

# Train the model
history = model.fit(
    datagen.flow(X_train, y_train_encoded, batch_size=32),
    epochs=20,
    validation_data=(X_val, y_val_encoded),
    callbacks=[early_stopping, reduce_lr]
)

# Training and Validation Accuracy
train_accuracy = history.history["accuracy"][-1]
val_accuracy = history.history["val_accuracy"][-1]
print(f"MobileNetV2 Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"MobileNetV2 Validation Accuracy: {val_accuracy * 100:.2f}%")

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test_encoded)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Predictions and Metrics for Validation Data
y_val_predictions = np.argmax(model.predict(X_val), axis=-1)
val_confusion_matrix = confusion_matrix(np.argmax(y_val_encoded, axis=-1), y_val_predictions)

# Predictions and Metrics for Test Data
y_test_predictions = np.argmax(model.predict(X_test), axis=-1)
test_confusion_matrix = confusion_matrix(np.argmax(y_test_encoded, axis=-1), y_test_predictions)

# Emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Classification Report for Validation Data
val_report = classification_report(np.argmax(y_val_encoded, axis=-1), y_val_predictions, target_names=emotion_labels)
print("Classification Report for MobileNetV2 (Validation Data):")
print(val_report)

# Classification Report for Test Data
test_report = classification_report(np.argmax(y_test_encoded, axis=-1), y_test_predictions, target_names=emotion_labels)
print("Classification Report for MobileNetV2 (Test Data):")
print(test_report)

# Plot confusion matrix for validation data
plt.figure(figsize=(8, 6))
sns.heatmap(
    val_confusion_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=emotion_labels,
    yticklabels=emotion_labels,
)
plt.title("MobileNetV2 Confusion Matrix (Validation Data)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Plot confusion matrix for test data
plt.figure(figsize=(8, 6))
sns.heatmap(
    test_confusion_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=emotion_labels,
    yticklabels=emotion_labels,
)
plt.title("MobileNetV2 Confusion Matrix (Test Data)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Plot training history
if history.history:
    pd.DataFrame(history.history).plot(figsize=(10, 5))
    plt.grid(True)
    plt.show()

    pd.DataFrame(history.history)[["accuracy", "val_accuracy"]].plot()
    plt.title("Model Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["Train", "Validation"])
    plt.grid()
    plt.show()

    pd.DataFrame(history.history)[["loss", "val_loss"]].plot()
    plt.title("Model Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Train", "Validation"])
    plt.grid()
    plt.show()

# Save the model
model.save("model/tf.h5")
