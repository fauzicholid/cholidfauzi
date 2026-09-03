"""CNN architecture for patch-based LULC (land use / land cover) classification.

Each training/inference sample is a small square patch of satellite bands
(shape: patch_size x patch_size x n_bands) centered on one pixel; the model
predicts the LULC class of that center pixel. Using 'same' padding and a
global-average-pooling head (instead of flattening) keeps the architecture
valid for any --patch-size used in prepare_training_patches.py.
"""
from tensorflow import keras
from tensorflow.keras import layers


def build_cnn(patch_size: int, n_bands: int, n_classes: int) -> keras.Model:
    inputs = keras.Input(shape=(patch_size, patch_size, n_bands), name="patch")

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    if patch_size >= 6:
        x = layers.MaxPooling2D(2, padding="same")(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="lulc_class")(x)

    model = keras.Model(inputs, outputs, name="lulc_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
