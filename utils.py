import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.manifold import TSNE
from tqdm import tqdm


def load_dataset(file_path: str) -> list[str]:
    with open(file_path, "r") as f:
        text_data = [line.strip() for line in f if line.strip()]
    return text_data


def build_vectorizer(
    text_data: list[str],
    vocab_size: int = 10000,
    sequence_length=None,
) -> keras.layers.TextVectorization:
    vectorizer = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=sequence_length,
    )

    vectorizer.adapt(text_data)
    return vectorizer


def build_vocab_mappings(
    vectorizer: keras.layers.TextVectorization,
) -> tuple[list[str], dict[str, int], dict[int, str]]:
    vocab = vectorizer.get_vocabulary()
    word_to_index = {w: i for i, w in enumerate(vocab)}
    index_to_word = {i: w for i, w in enumerate(vocab)}
    return vocab, word_to_index, index_to_word


def build_context_target_pairs(
    sequences: np.ndarray,
    window_size: int = 2,
    pad_token: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    context_size = window_size * 2
    X_train = []
    y_train = []

    for seq in tqdm(sequences):
        # Remove padding tokens
        seq = [token for token in seq if token != pad_token]

        if len(seq) < context_size + 1:
            continue

        for i in range(window_size, len(seq) - window_size):
            target = seq[i]
            left_context = seq[i - window_size : i]
            right_context = seq[i + 1 : i + 1 + window_size]
            context = left_context + right_context
            if len(context) != context_size:
                continue
            X_train.append(context)
            y_train.append(target)

    X_train = np.array(X_train, dtype="int32")
    y_train = np.array(y_train, dtype="int32")
    return X_train, y_train

def build_word_emb_model(
    vocab_size: int,
    embedding_dim: int,
    context_size: int,
    hidden_units: int | None = None,
) -> keras.Model:
    inputs = keras.Input(shape=(context_size,), dtype="int32", name="context_words")
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        name="word_embeddings"
    )(inputs)
    x = layers.GlobalAveragePooling1D()(x)

    if hidden_units:
        x = layers.Dense(hidden_units, activation="relu")(x)

    outputs = layers.Dense(vocab_size, activation="softmax", name="target_probs")(x)

    model = keras.Model(inputs, outputs, name=f"word_emb_d{embedding_dim}_c{context_size}")
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def get_callbacks(model_path: str) -> list[keras.callbacks.Callback]:
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    )

    path = keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )
    return [early_stopping, path]


def get_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate the cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)


def get_top_k_similar_words(target_word, embeddings, word_index, vocab, k=10):
    """Get the top k most similar words to the given word based on cosine similarity."""
    if target_word not in word_index:
        print(f"Word '{target_word}' not found in vocabulary.")
        return []

    target_idx = word_index[target_word]
    target_vector = embeddings[target_idx]

    similarities = []
    for word, idx in word_index.items():
        if word == target_word or word == "":  # Skip self and padding
            continue
        sim = get_cosine_similarity(target_vector, embeddings[idx])
        similarities.append((word, sim))

    # Sort by descending similarity and return top K
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]


def visualize_tsne_embeddings(
    words, embeddings, word_index, title="t-SNE Embeddings", filename=None
):
    """
    Visualizes t-SNE embeddings of selected words.

    Args:
        words (list): List of words to visualize.
        embeddings (numpy.ndarray): Array containing word embeddings.
        word_index (dict): Mapping of words to their indices in the embeddings array.
        title (str): Title for the plot.
        filename (str, optional): File to save the visualization. If None, plot is displayed.

    Returns:
        None
    """
    # Filter the embeddings for the selected words
    indices = [word_index[word] for word in words]
    selected_embeddings = embeddings[indices]

    # Set perplexity for t-SNE, it's recommended to use a value less than the number of selected words
    perplexity = min(5, len(words) - 1)

    # Use t-SNE to reduce dimensionality
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0)
    reduced_embeddings = tsne.fit_transform(selected_embeddings)

    # Plotting
    plt.figure(figsize=(10, 10))
    plt.title(title)
    for i, word in enumerate(words):
        plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1])
        plt.annotate(
            word,
            xy=(reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
            xytext=(5, 2),
            textcoords="offset points",
            ha="right",
            va="bottom",
        )

    # Save or display the plot
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def visualize_all_tsne_embeddings(
    embeddings, word_index, words_to_plot, words_to_label=None, filename=None
):
    """
    Visualizes t-SNE embeddings of selected words with optional labeling.

    Args:
        embeddings (numpy.ndarray): Array containing word embeddings.
        word_index (dict): Mapping of words to their indices in the embeddings array.
        words_to_plot (list): List of words to plot.
        words_to_label (list, optional): List of words to label. Defaults to None.
        filename (str, optional): File to save the visualization. If None, plot is displayed.

    Returns:
        None
    """
    # Create a reverse mapping from index to word
    index_word = {index: word for word, index in word_index.items()}

    # Ensure words_to_label is a subset of words_to_plot
    if words_to_label is None:
        words_to_label = words_to_plot
    words_to_label = set(words_to_label).intersection(words_to_plot)

    # Filter the embeddings for the words to plot
    indices_to_plot = [word_index[word] for word in words_to_plot if word in word_index]
    selected_embeddings = embeddings[indices_to_plot]

    # Set perplexity for t-SNE, it's recommended to use a value less than the number of selected words
    perplexity = min(5, len(words_to_plot) - 1)

    # Use t-SNE to reduce dimensionality
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0)
    reduced_embeddings = tsne.fit_transform(selected_embeddings)

    # Plotting
    plt.figure(figsize=(12, 12))
    for i, index in enumerate(indices_to_plot):
        plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1], alpha=0.5)
        if index_word[index] in words_to_label:  # Annotate only selected words
            plt.annotate(
                index_word[index],
                xy=(reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
                xytext=(5, 2),
                textcoords="offset points",
                ha="right",
                va="bottom",
            )
