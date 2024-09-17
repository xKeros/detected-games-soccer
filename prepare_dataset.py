import os
import numpy as np
import json
from keras.preprocessing.sequence import pad_sequences

def load_annotations(annotation_path):
    with open(annotation_path, 'r') as f:
        data = json.load(f)
    return data['faltas']

def create_sequences(video_frames_folder, audio_features_folder, annotations, sequence_length=30):
    sequences = []
    labels = []
    frame_files = sorted(os.listdir(video_frames_folder))
    num_frames = len(frame_files)
    # Convertir tiempos de faltas a índices de frames
    faltas_frames = []
    for falta in annotations:
        inicio_frame = int(falta['inicio'] * fps)
        fin_frame = int(falta['fin'] * fps)
        faltas_frames.extend(range(inicio_frame, fin_frame+1))
    for i in range(0, num_frames - sequence_length):
        frame_sequence = frame_files[i:i+sequence_length]
        # Cargar imágenes y audio aquí (omito por brevedad)
        label = 1 if any(f in faltas_frames for f in range(i, i+sequence_length)) else 0
        sequences.append(frame_sequence)
        labels.append(label)
    return sequences, labels
