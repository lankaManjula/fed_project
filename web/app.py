import os
import cv2
import numpy as np
import joblib
from flask import Flask, request, Response, render_template, jsonify, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model # type: ignore

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'web/static/upload/'
app.config['RESULT_FOLDER'] = 'web/static/'

# Load pre-trained models
cnn_model = load_model('model/cnn.h5')  # CNN Model
tf_model = load_model('model/tf.h5')
svm_pca_model = joblib.load('model/svm.pkl')  # SVM with PCA
svm_model = svm_pca_model['svm_model']
pca = svm_pca_model['pca']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

# Detect face and crop the image. 
def crop_image(image):
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        image = image[y:y+h, x:x+w]
    return image

def preprocess_image_svm(img):
    img = crop_image(img)
    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    img_flat = img.flatten() # Flatten to 1D
    img_pca = pca.transform([img_flat]) # Apply PCA transformation
    return img_pca

def preprocess_image_cnn(img):
    img = crop_image(img)
    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    img = np.expand_dims(img, axis=(0, -1)) # Shape (1, 64, 64, 1)
    return img

def preprocess_image_tf(img):
    img = crop_image(img)
    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    img = np.stack((img,) * 3, axis=-1) # Shape: (64, 64, 3)
    img = np.expand_dims(img, axis=0) # Shape: (1, 64, 64, 3)
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file_name, file_extension = os.path.splitext(filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Check if file is an image
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({'error': 'Invalid file type. Please upload an image.'})

    # Load and process the image
    img = cv2.imread(filepath)

    # file path
    result_filename = f'result{file_extension}'
    result_path_svm = os.path.join(app.config['RESULT_FOLDER'], 'svm_'+ result_filename)
    result_path_cnn = os.path.join(app.config['RESULT_FOLDER'], 'cnn_'+ result_filename)
    result_path_tf = os.path.join(app.config['RESULT_FOLDER'], 'tf_'+ result_filename)
  
    # Generate image url
    result_url_svm = url_for('static', filename=f'/svm_{result_filename}', _external=True)
    result_url_cnn = url_for('static', filename=f'/cnn_{result_filename}', _external=True)
    result_url_tf = url_for('static', filename=f'/tf_{result_filename}', _external=True)

    cv2.imwrite(result_path_svm, img)
    cv2.imwrite(result_path_cnn, img)
    cv2.imwrite(result_path_tf, img)

    # Process for SVM
    try:
        svm_input = preprocess_image_svm(img)
        svm_prediction = svm_model.predict(svm_input)[0]  # SVM predicts a class label
        svm_emotion = emotion_dict[int(svm_prediction)]
        svm_img = cv2.imread(result_path_svm)
        cv2.putText(svm_img, svm_emotion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imwrite(result_path_svm, svm_img)
    except Exception as e:
        svm_prediction = str(e)

    # Process for CNN
    try:
        cnn_input = preprocess_image_cnn(img)
        cnn_prediction = cnn_model.predict(cnn_input)
        cnn_label = np.argmax(cnn_prediction)
        cnn_accuracy = np.max(cnn_prediction) * 100
        cnn_emotion = emotion_dict[int(cnn_label)]
        cnn_img = cv2.imread(result_path_cnn)
        cv2.putText(cnn_img, cnn_emotion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite(result_path_cnn, cnn_img)
    except Exception as e:
        cnn_label, cnn_accuracy = "Error", str(e)

    # Process for Transfer lering Model
    try:
        transform_input = preprocess_image_tf(img)
        transform_prediction = tf_model.predict(transform_input)
        transform_label = np.argmax(transform_prediction)
        transform_accuracy = np.max(transform_prediction) * 100
        transform_emotion = emotion_dict[int(transform_label)]
        transform_img = cv2.imread(result_path_tf)
        cv2.putText(transform_img,  transform_emotion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imwrite(result_path_tf, transform_img)
    except Exception as e:
        transform_label, transform_accuracy = "Error", str(e)

    # Return predictions, accuracies, and the image URL
    return jsonify({
        'svm_prediction': {
            'accuracy': float(svm_prediction),
            'label': int(svm_prediction),
            'emo': svm_emotion,
            'result_image_url': result_url_svm
        },
        'cnn_prediction': {
            'label': int(cnn_label),
            'accuracy': float(cnn_accuracy),
            'emo': cnn_emotion,
            'result_image_url': result_url_cnn
        },
        'tf_prediction': {
            'label': int(transform_label),
            'accuracy': float(transform_accuracy),
            'emo': transform_emotion,
            'result_image_url': result_url_tf
        }
    })

#  Video capture object
cap = cv2.VideoCapture(0)

def generate_frames(mode_name = 'cnn'):
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (400, 400))

        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        # Process each detected face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y - 10), (x + w, y + h + 30), (0, 255, 0), 2)
            roi_gray = gray_frame[y:y + h, x:x + w] # Region of Interes
            if mode_name == 'cnn':
                model = cnn_model
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (64, 64)), -1), 0)
            if mode_name == 'tf':
                model = tf_model
                cropped_img = cv2.resize(roi_gray, (64, 64)) # Resize to (64, 64)
                cropped_img = np.stack((cropped_img,) * 3, axis=-1) # Convert to RGB (64, 64, 3)
                cropped_img = np.expand_dims(cropped_img, axis=0) 

            # Predict emotion
            emotion_prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            emotion_label = emotion_dict[maxindex]

            print(maxindex, emotion_label)

            # Display emotion on frame
            cv2.putText(frame, emotion_label, (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame as part of the response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames('cnn'), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
