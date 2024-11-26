# Face Emotion Detection Project

Detect and classify emotions from facial expressions in images or video.

---

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
- [Usage](#usage)
  - [Download Dataset](#download-dataset)
  - [Data Analysis](#data-analysis)
  - [Preprocessing](#preprocessing)
  - [Model Training](#model-training)
  - [Running the Web Server](#running-the-web-server)
- [Models](#models)
---

## Introduction

This project aims to detect and classify emotions from facial expressions using different machine learning and deep learning techniques. It includes implementations for Support Vector Machines (SVM), Convolutional Neural Networks (CNN), and transfer learning approaches. A web server is also provided to showcase real-time emotion detection.

---

## Setup

Follow these steps to set up the project on your local machine.

### 1. Clone the Repository

Clone the repository using the following command:
```
git clone git@github.com:lankaManjula/fed_project.git
```

Navigate to the project directory:
```
cd fed-project
```

### 2. Create a Virtual Environment and Activate

**Windows:**
```
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install the required packages:
```
pip install -r requirements.txt
```

---

## Usage

### Download Dataset

Download the required datasets from Kaggle by providing your username and password:
```
python fed\download.py
```

### Data Analysis

Analyze the dataset:
```
python fed\analysis.py
```

### Preprocessing

Preprocess the data:
```
python fed\preprocess.py
```

### Model Training

#### Train SVM Model:
```
python fed\svm.py
```

#### Train CNN Model:
```
python fed\cnn.py
```

#### Train Transfer Learning Model:
```
python fed\tf.py
```

---

### Running the Web Server

Launch the web server to interact with the emotion detection system:
```
python web\app.py
```

---

## Models

This project supports the following models:

- **Support Vector Machine (SVM):** A classic machine learning approach for emotion classification.
- **Convolutional Neural Networks (CNN):** A deep learning approach optimized for image data.
- **Transfer Learning:** Uses MobileNet for improved accuracy.
---
