(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=vocab_size)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalMaxPooling1D, Conv1D, MaxPooling1D, Dropout
from tensorflow.keras.datasets import reuters
import numpy as np
import matplotlib.pyplot as plt

# Function to build models
def build_model(num_classes, output_dim,
                embedding_matrix, input_lenght=256):
    """
    Builds and compiles a convolutional neural network model.

    Parameters:
    use_pretrained (bool): If True, use the pretrained embedding matrix; otherwise, 
        initialize embeddings randomly.
    output_dim (int): Dimension of the embedding vectors.
    num_classes (int): Number of output classes.
    embedding_matrix (np.array): Pretrained embedding matrix.

    Returns:
    tf.keras.Model: Compiled Keras model.
    """
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=output_dim, 
                  weights=None, trainable=False, input_length=input_length),
        Conv1D(128, 5, activation='relu'),
        MaxPooling1D(5),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Load reuters dataset
vocab_size = 20000 # Limit vocabulary size to 20,000 most frequent words
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=vocab_size)
