import tensorflow as tf
import numpy as np
import librosa

# Modelo CNN preentrenado
cnn_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', pooling='avg')

def extract_image_features(image_paths):
    features = []
    for img_path in image_paths:
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
        feature = cnn_model.predict(img_array)
        features.append(feature.flatten())
    return np.array(features)

def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled
