import json
import os
from tensorflow import keras
from tensorflow.keras import layers

def build_model(input_shape, num_classes, config_name="config_1"):
    """Fully-connected neural network (MLP) with configurable structure."""
    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Flatten()(x)

    if config_name == "config_1":
        # โครงสร้าง 1: 1 Hidden Layer
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        
    elif config_name == "config_2":
        # โครงสร้าง 2: 3 Hidden Layers
        x = layers.Dense(256, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(
        1 if num_classes == 2 else num_classes,
        activation="sigmoid" if num_classes == 2 else "softmax"
    )(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy" if num_classes == 2 else "sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def train_model(X_train, y_train, X_val, y_val, num_classes,
                output_dir=None, epochs=30, batch_size=32, config_name="config_1"):
    """Build, train and save the model. Returns (model, history)."""
    model = build_model(X_train.shape[1:], num_classes, config_name)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
    ]

    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        model_name = f"nn_model_{config_name}_{epochs}ep.keras"
        model.save(os.path.join(output_dir, model_name))
        
        with open(os.path.join(output_dir, f"history_{config_name}_{epochs}ep.json"), "w") as f:
            json.dump({k: [float(v) for v in vs] for k, vs in history.history.items()}, f)

        print(f"Saved: {os.path.join(output_dir, model_name)}")

    return model, history

def predict_model(model, X_test):
    probabilities = model.predict(X_test, verbose=0)
    if probabilities.shape[-1] == 1:
        return (probabilities.ravel() > 0.5).astype(int)
    return probabilities.argmax(axis=1)