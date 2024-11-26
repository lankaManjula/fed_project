import pandas as pd
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # type: ignore

# Read the dataset
df = pd.read_csv('data/facial-expression-recognitionferchallenge/fer2013/fer2013/fer2013.csv')

# Load the Haar Cascade face detector to detect faces in an image
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Converts the image to grayscale to reduce the complexity
def resize_image(image, size=(64, 64)):
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

# Applying Gaussian blur to remove noise
def apply_gaussian_blur(image):
    return cv2.GaussianBlur(image, (3, 3), 0)

# Normalize the images by scaling the pixel values to a range of 0-1
def normalize_image(image):
    return image / 255.0

# Detect facce and crop the image. 
def crop_image(image):
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        image = image[y:y+h, x:x+w]
    return image

# svm
def preprocess_image_svm(img_str, size=(64, 64)):
    img = np.fromstring(img_str, sep=' ', dtype=np.uint8).reshape(48, 48)
    img = crop_image(img)
    img = resize_image(img, size,)
    img = apply_gaussian_blur(img)
    img = normalize_image(img)
    img = img.flatten()  # 1D array
    return img

# cnn
def preprocess_image_cnn(img_str, size=(64, 64)):
    img = np.fromstring(img_str, sep=' ', dtype=np.uint8).reshape(48, 48)
    img = crop_image(img)
    img = resize_image(img, size)
    img = apply_gaussian_blur(img)
    img = normalize_image(img)
    img = np.expand_dims(img, axis=-1) # 1 channel
    return img

# transform
def preprocess_image_tf(img_str, size=(64, 64)):
    img = np.fromstring(img_str, sep=' ', dtype=np.uint8).reshape(48, 48)
    img = crop_image(img)
    img = resize_image(img, size)
    img = apply_gaussian_blur(img)
    img_rgb = np.stack((img,) * 3, axis=-1)  # Shape: (64, 64, 3)
    img_rgb = preprocess_input(img_rgb)  # Normalizes to [-1, 1]
    return img_rgb

# Preprocess dataset
def preprocessDataset(df, model_name='cnn'):
    X = []
    y = df['emotion'].values
    for pixels in df['pixels']:
        if model_name=='svm':
            X.append(preprocess_image_svm(pixels, size=(64, 64)))
        if model_name=='cnn':
            X.append(preprocess_image_cnn(pixels, size=(64, 64)))
        if model_name=='tf':
            X.append(preprocess_image_tf(pixels, size=(64, 64)))
    X = np.array(X)
    # Print shape of preprocessed data
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    return np.array(X), np.array(y)

# Train dataset 80%
def trainData(model_name):
    train_df = df[df['Usage'] == 'Training']
    return preprocessDataset(train_df, model_name)

# Validation dataset 10%
def validateData(model_name):
     validate_df = df[df['Usage'] == 'PublicTest']
     return preprocessDataset(validate_df, model_name)

# Test dataset 10%
def testData(model_name):
    test_df = df[df['Usage'] == 'PrivateTest']
    return preprocessDataset(test_df,model_name)
