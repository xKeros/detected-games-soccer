import tensorflow as tf

def build_model(sequence_length, image_feature_dim, audio_feature_dim):
    # Entrada de imágenes
    input_images = tf.keras.Input(shape=(sequence_length, image_feature_dim))
    x = tf.keras.layers.LSTM(128)(input_images)
    
    # Entrada de audio
    input_audio = tf.keras.Input(shape=(audio_feature_dim,))
    y = tf.keras.layers.Dense(64, activation='relu')(input_audio)
    y = tf.keras.layers.Dense(32, activation='relu')(y)
    
    # Combinación
    combined = tf.keras.layers.concatenate([x, y])
    output = tf.keras.layers.Dense(1, activation='sigmoid')(combined)
    
    model = tf.keras.Model(inputs=[input_images, input_audio], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
