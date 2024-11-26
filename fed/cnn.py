import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
from preprocess import trainData, validateData, testData

# Model name
model_name = "cnn"

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
    y_test_encoded.shape,
)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)
datagen.fit(X_train)

# Model
def createModel(input_shape=(64, 64, 1), num_classes=7):
    model = Sequential()
    model.add(
        Conv2D(
            32,
            kernel_size=(3, 3),
            activation="relu",
            padding="SAME",
            input_shape=input_shape
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, kernel_size=(3, 3), padding="SAME", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, kernel_size=(3, 3), padding="SAME", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(256, kernel_size=(3, 3), padding="SAME", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))
    return model

# Create model
model = createModel()

# Print summary
model.summary()

# Learning Rate
lr_schedule = ExponentialDecay(0.001, decay_steps=100000, decay_rate=0.96)

optimizer = Adam(learning_rate=lr_schedule)
model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
)

# Callbacks
early_stopping = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)
lr_scheduler = ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=3, min_lr=0.001
)

# Train the model
history = model.fit(
    datagen.flow(X_train, y_train_encoded, batch_size=32),
    validation_data=(X_val, y_val_encoded),
    epochs=50,
    callbacks=[early_stopping, lr_scheduler]
)

# Training and Validation Accuracy
train_accuracy = history.history["accuracy"][-1]  # Accuracy from last epoch
val_accuracy = history.history["val_accuracy"][-1]  # Validation Accuracy from last epoch

# Print training and validation accuracy
print(f"CNN Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"CNN Validation Accuracy: {val_accuracy * 100:.2f}%")

# Evaluate the model, Test Accuracy
test_loss, test_accuracy = model.evaluate(X_test, y_test_encoded)
print(f"CNN Test accuracy: {test_accuracy * 100:.2f}%")

# Predictions and Metrics for Validation
y_val_predictions = np.argmax(model.predict(X_val), axis=-1)
cnn_cm_val = confusion_matrix(np.argmax(y_val_encoded, axis=-1), y_val_predictions)

# Predictions and Metrics for Test
y_test_predictions = np.argmax(model.predict(X_test), axis=-1)
cnn_cm_test = confusion_matrix(np.argmax(y_test_encoded, axis=-1), y_test_predictions)

# Emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Classification Report for Validation Data
val_report = classification_report(
    np.argmax(y_val_encoded, axis=-1), y_val_predictions, target_names=emotion_labels
)
print("Classification Report for CNN (Validation Data):")
print(val_report)

# Classification Report for Test Data
test_report = classification_report(
    np.argmax(y_test_encoded, axis=-1), y_test_predictions, target_names=emotion_labels
)
print("Classification Report for CNN (Test Data):")
print(test_report)

# Plot confusion matrix for validation data
plt.figure(figsize=(8, 6))
sns.heatmap(
    cnn_cm_val,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=emotion_labels,
    yticklabels=emotion_labels
)
plt.title("CNN Confusion Matrix (Validation Data)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Plot confusion matrix for test data
plt.figure(figsize=(8, 6))
sns.heatmap(
    cnn_cm_test,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=emotion_labels,
    yticklabels=emotion_labels,
)
plt.title("CNN Confusion Matrix (Test Data)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Plot training history
if history.history:
    pd.DataFrame(history.history).plot(figsize=(10, 5))
    plt.grid(True)
    plt.show()

    # Plot training history
    pd.DataFrame(history.history)[["accuracy", "val_accuracy"]].plot()
    plt.title("Model accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["Train", "Validation"])
    plt.grid(True)
    plt.show()

    # Plot training loss
    pd.DataFrame(history.history)[["loss", "val_loss"]].plot()
    plt.title("Model loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Train", "Validation"])
    plt.grid(True)
    plt.show()

# Save the model
model.save("model/cnn.h5")
