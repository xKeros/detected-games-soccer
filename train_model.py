from model import build_model
from feature_extraction import extract_image_features, extract_audio_features
import numpy as np

# Supongamos que ya tienes tus secuencias y etiquetas
sequences = ...  # Lista de secuencias de imágenes
labels = ...     # Lista de etiquetas (1: falta, 0: no falta)

# Extracción de características
image_features = []
audio_features = []
for seq in sequences:
    img_feats = extract_image_features(seq['images'])
    image_features.append(img_feats)
    aud_feat = extract_audio_features(seq['audio'])
    audio_features.append(aud_feat)

image_features = np.array(image_features)
audio_features = np.array(audio_features)
labels = np.array(labels)

# Construir y entrenar el modelo
model = build_model(sequence_length=image_features.shape[1],
                    image_feature_dim=image_features.shape[2],
                    audio_feature_dim=audio_features.shape[1])

model.fit([image_features, audio_features], labels, epochs=10, batch_size=8, validation_split=0.2)
