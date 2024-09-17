import cv2
import numpy as np
from feature_extraction import extract_image_features, extract_audio_features
from model import build_model

# Cargar el modelo entrenado
model = build_model(sequence_length, image_feature_dim, audio_feature_dim)
model.load_weights('modelo_entrenado.h5')

def detect_fouls_in_video(video_path, audio_path):
    # Extracción de fotogramas y audio
    frames = extract_frames_from_video(video_path)
    audio_features = extract_audio_features(audio_path)
    
    # Procesamiento por secuencias
    for i in range(0, len(frames) - sequence_length):
        frame_sequence = frames[i:i+sequence_length]
        img_features = extract_image_features(frame_sequence)
        input_images = np.expand_dims(img_features, axis=0)
        input_audio = np.expand_dims(audio_features, axis=0)
        
        prediction = model.predict([input_images, input_audio])
        if prediction > 0.5:
            print(f"Falta detectada en el segundo {i / fps}")
            # Marcar en el video o generar reporte
