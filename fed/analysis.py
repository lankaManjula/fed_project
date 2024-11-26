import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv_path = 'data/facial-expression-recognitionferchallenge/fer2013/fer2013/fer2013.csv'
df = pd.read_csv(csv_path)

# Print first 5 rows
print('First 5 rows:', df.head())

# Print total data
print('Shape:', df.shape)

# Ensure data types
print('Types:', df.dtypes)

# Ensure null values
print('Info:', df.info())
print('Null values:', df.isnull().sum())

# Ensure usage type
print('Unique Usage:', df['Usage'].unique())

# Count of each dataset usage type
usage_counts = df['Usage'].value_counts()
print('Count Usage: ', usage_counts)

# Ensure length of the image 
print('Length of the image:', math.sqrt(len(df.iloc[0].pixels.split(' '))))

# Create image size
Image_size = (48,48)

# Ensure color Images
rgb_images = df[df['pixels'].apply(lambda p: len(p.split()) != 48 * 48)]
print('Color Images(RGB):', len(rgb_images))

#Ensure all are gray images and should match with df.shape
gray_images = df[df['pixels'].apply(lambda p: len(p.split()) == 48 * 48)]
print('Gray Images:', len(gray_images))

# Ensure emotions
unique_emo = df['emotion'].unique()
print('Unique emotions:', unique_emo.sort())

# Count Emotons
print(df['emotion'].value_counts())

emo_df = df[df['emotion'].isin(range(0, len(unique_emo)))].groupby('emotion').head(1)
emo_df = emo_df.sort_values(by='emotion')

plt.figure(figsize=(15, 7))

for i, (pixels, emotion) in enumerate(zip(emo_df['pixels'], emo_df['emotion'])):
    img = np.array(list(map(int, pixels.split())), dtype=np.uint8)  
    img = img.reshape(Image_size)    
    plt.subplot(1, len(emo_df), i + 1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(f"Emotion: {emotion}")

plt.tight_layout()
plt.show()

# Create emotion lables
emotion_labels = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}

# plot emotions
fig = plt.figure(figsize =(10, 7))
emotion_counts = df['emotion'].value_counts()
emotion_names = [emotion_labels[idx] for idx in emotion_counts.index]
plt.bar(emotion_names, emotion_counts.values)
plt.xlabel('Emotion')
plt.ylabel('Emotion Count')
plt.title('Emotion Dataset')
plt.tight_layout()
plt.show()

# plot 20 images
def plotImos(df, num_images = 20):
    print(df.shape)
    df = df[:num_images]
    plt.figure(figsize=(15, 5))

    for i, (pixels, emotion) in enumerate(zip(df['pixels'], df['emotion'])):
        if i >= num_images:
                break
        img = np.array(list(map(int, pixels.split())), dtype=np.uint8)  
        img = img.reshape(Image_size)
        emotion_lbl = emotion_labels[int(emotion)]    
        plt.subplot(2, int(num_images/2), i + 1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.title(emotion_lbl)

    plt.tight_layout()
    plt.show()

df_train = df[df['Usage'] == 'Training']
df_val = df[df['Usage'] == 'PublicTest']
df_test= df[df['Usage'] == 'PrivateTest']

plotImos(df_train)
