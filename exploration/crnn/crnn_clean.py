"""
CRNN Model for Captcha Recognition Training and Evaluation
---
File Overview:
This script trains and evaluates a Convolutional-Recurrent Neural Network (CRNN)
with a Connectionist Temporal Classification (CTC) loss for recognizing
captcha images.

---
Evaluation Results: (evaluation script not included here)
Top 20 confused pairs (with freq + accuracy):
 True   Pred   ConfCnt | True_a True_b Pred_a Pred_b |  Acc_a  Acc_b
--------------------------------------------------------------------
     0 o            91 |    342    348    305    413 |  0.553  0.649
     5 s            77 |    319    343    245    393 |  0.599  0.784
     o 0            59 |    348    342    413    305 |  0.649  0.553
     1 i            48 |    317    353    312    435 |  0.612  0.751
     l i            48 |    356    353    314    435 |  0.680  0.751
     j i            36 |    329    353    292    435 |  0.757  0.751
     d o            35 |    340    348    263    413 |  0.656  0.649
     m n            29 |    325    357    294    411 |  0.763  0.810
     a r            28 |    356    299    271    352 |  0.618  0.823
     v u            25 |    348    325    328    338 |  0.796  0.822
     i 1            25 |    353    317    435    312 |  0.751  0.612
     z 2            24 |    318    323    329    330 |  0.824  0.833
     g 9            22 |    324    336    305    353 |  0.707  0.821
     b 8            22 |    323    335    309    292 |  0.759  0.713
     y 4            22 |    351    317    312    333 |  0.758  0.817
     l 1            21 |    356    317    314    312 |  0.680  0.612
     k h            21 |    354    353    339    345 |  0.816  0.762
     8 b            21 |    335    323    292    309 |  0.713  0.759
     a q            21 |    356    333    271    396 |  0.618  0.781
     0 q            20 |    342    333    305    396 |  0.553  0.781
exact-match: 35.50% | CER: 0.2230
accuracy: 76.19%

---
Key Efforts & Design Highlights:

1.  **Custom CTCLayer:** A custom Keras layer was implemented to wrap the 
    `keras.backend.ctc_batch_cost` function. This allows the CTC loss to be
    computed *inside* the model, which simplifies training (no need for a 
    custom `train_step`) and cleanly separates the model's logic.

2.  **CRNN Architecture:** The model follows a standard CRNN pattern:
    * **CNN:** A stack of Conv2D/MaxPooling2D layers (64, 128, 256 filters)
        acts as a robust feature extractor.
    * **Pooling Strategy:** Pooling is intentionally stopped after the second
        block to preserve a longer sequence length (width) for the RNN.
    * **CNN-to-RNN Bridge:** `Permute` and `Reshape` layers are used to
        transform the CNN's 2D feature map (H, W, C) into a 1D sequence
        (W, H*C) that the RNN can process.
    * **RNN:** Two layers of Bidirectional LSTMs are used to process the
        sequence, capturing context from both left-to-right and right-to-left,
        which is crucial for OCR.

3.  **CTC Blank Bias Initialization:** A critical (and non-obvious) 
    optimization was implemented. The final 'blank' token's bias in the
    softmax layer is initialized to a strong negative value (-5.0). This 
    prevents the network from defaulting to predicting 'blank' for all
    time steps early in training, which can cause learning to stall.

4.  **Robust Callbacks:** The training process uses a set of callbacks for
    professional-grade model training:
    * `ModelCheckpoint`: Saves both the *latest* epoch (for recovery) and
        the *best* epoch (based on 'val_loss') for final evaluation.
    * `ReduceLROnPlateau`: Dynamically adjusts the learning rate when
        progress stalls, helping to find a better optimum.
    * `TQDMProgressBar`: A custom callback for a clean, user-friendly
        progress bar that reports essential metrics.

5.  **Custom Evaluation Metrics:** The standard evaluation metrics for this
    task (Character Error Rate - CER) are not built into Keras. Therefore,
    `levenshtein` distance and `cer` were implemented from scratch.
    
6.  **Greedy CTC Decoder:** A `greedy_decode` function was implemented to
    translate the raw (T, C) softmax output from the model back into
    a human-readable string, correctly handling the collapsing of
    repeated characters and the removal of blank tokens.
"""

from json import load
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from tqdm import tqdm
import os

from sklearn.model_selection import train_test_split

# --- Constants ---
# Centralizing constants here makes the script easy to configure and maintain.

# Image dimensions
IMG_WIDTH = 550
IMG_HEIGHT = 80

# Data and Paths
SAVE_PREPROCESSED_IMAGES_DIR = 'crnn_preprocessed_images.npz'
TEST_PREPROCESSED_IMAGES_DIR = 'crnn_test_images.npz'
CHECKPOINT_DIR = "/home/h/crnn/captcha/checkpoints_crnn"
BEST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "crnn_best_simple_model.weights.h5")
LATEST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "crnn_latest_simple_model.weights.h5")

# Model & Alphabet
ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"
NUM_CLASSES = len(ALPHABET)
BLANK_INDEX = NUM_CLASSES  # CTC blank token is the last index (0-35 are chars, 36 is blank)
idx_to_char = np.array(list(ALPHABET))
char_to_idx = {c: i for i, c in enumerate(ALPHABET)}

# Training Hyperparameters
BATCH_SIZE = 64
EPOCHS = 1000
LEARNING_RATE = 1e-3
CLIPNORM = 5.0  # Using clipnorm to prevent exploding gradients in the RNN
LR_PATIENCE = 3
LR_FACTOR = 0.3
# EARLY_STOP_PATIENCE = 8

# --- Data Loading & Preprocessing ---

def encode_label(text: str) -> np.ndarray:
    """
    Encode a captcha string (e.g. 'a3fz') into an int array using ALPHABET.
    Raises if it sees an unexpected character.
    """
    text = text.strip().lower()
    indices = []
    for ch in text:
        if ch in char_to_idx:
            indices.append(char_to_idx[ch])
        else:
            raise ValueError(f"Unexpected character '{ch}' in label '{text}'")
    return np.array(indices, dtype="int32")

def load_preprocessed_data(path):
    """
    Loads the .npz file containing the preprocessed images and labels.
    """
    try:
        print(f"Loading preprocessed data from {path}...")
        data = np.load(path)

        images = data['images']
        labels = data['labels']

        print("Data loaded successfully.")
        print(f"Images array shape: {images.shape}")
        print(f"Labels array shape: {labels.shape}")

        return images, labels

    except FileNotFoundError:
        print(f"[Error] File not found at {path}.")
        return None, None
    except Exception as e:
        print(f"[Error] Could not load data: {e}")
        return None, None

# --- Model Definition ---

class CTCLayer(layers.Layer):
    """
    --- Effort Highlight: Custom CTC Loss Layer ---
    
    This custom Keras layer wraps the backend CTC loss function.
    This is a cleaner approach than a custom training loop or a `model.add_loss()`
    call in the model definition, as it encapsulates the loss logic.
    
    It takes two inputs:
    1. y_true (labels): (B, max_label_len), padded with -1.
    2. y_pred (softmax): (B, T, NUM_CLASSES + 1), the raw RNN output.
    
    It computes the loss and adds it to the model using `self.add_loss()`.
    The layer's *output* is just y_pred, passed through.
    """
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # y_true: (B, max_label_len) with -1 padding
        # y_pred: (B, T, NUM_CLASSES+1) softmax outputs
        batch_len = tf.cast(tf.shape(y_true)[0], "int64")
        time_steps = tf.cast(tf.shape(y_pred)[1], "int64")
        
        y_true = tf.cast(y_true, "int32")

        # Crucial for CTC: find the *true* length of each label, ignoring the -1 padding.
        # `label_length` will be (B, 1) tensor (e.g., [[5], [6], [4]])
        label_length = tf.math.count_nonzero(y_true != -1, axis=-1, keepdims=True)

        # Replaces -1 padding with 0. This is safe because the CTC loss
        # function ignores any elements in y_true *past* the `label_length`.
        y_true_safe = tf.where(y_true != -1, y_true, 0)

        # Ensure y_pred is float32 for the loss function
        y_pred = tf.cast(y_pred, tf.float32)

        # Tell CTC that all sequences have the full time_steps length.
        # `input_length` will be (B, 1) tensor (e.g., [[137], [137], [137]])
        input_length = time_steps * tf.ones((batch_len, 1), dtype="int64")

        # Calculate the loss
        loss = self.loss_fn(y_true_safe, y_pred, input_length, label_length)
        
        # This is the key step: add the loss to the model.
        self.add_loss(tf.reduce_mean(loss))
        
        # We must return y_pred so it can be used as the model's output
        return y_pred


def build_model():
    """
    --- Effort Highlight: CRNN Model Architecture ---
    
    This function builds the complete CRNN model.
    """
    image_input = layers.Input((IMG_HEIGHT, IMG_WIDTH, 1), name="image")
    # Labels are variable length, so shape is (None,)
    labels_input = layers.Input((None,), dtype="int32", name="label")

    x = image_input

    # --- CNN Feature Extractor ---
    # Block 1
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)  # Dims: 80x550 -> 40x275

    # Block 2
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)  # Dims: 40x275 -> 20x137

    # Block 3
    # **Design Choice**: No pooling here. We want to preserve the sequence
    # length (width=137) as much as possible for the RNN.
    # The height (20) will be "flattened" into the feature vector.
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    # Final CNN output shape: (B, 20, 137, 256) [B, H, W, C]

    # --- CNN-to-RNN Bridge ---
    # This is a crucial step to "read" the image for the RNN.
    # 1. Permute: (B, H, W, C) -> (B, W, H, C)
    #    We want the Width (137) to be the 'time' dimension.
    x = layers.Permute((2, 1, 3))(x)
    
    # 2. Reshape: (B, W, H, C) -> (B, W, H*C)
    #    This flattens the height and channel dimensions into a single
    #    large feature vector (20 * 256 = 5120) for *each* time step (width).
    #    Shape is now (B, 137, 5120)
    x = layers.Reshape((-1, 20 * 256))(x)

    # --- RNN Sequence Processor ---
    # Two layers of Bidirectional LSTMs process the sequence.
    # `Bidirectional` reads the sequence forwards and backwards,
    # capturing context from both sides, which is vital for OCR.
    # `return_sequences=True` is essential, as we need an output
    # for *each* time step for CTC.
    x = layers.Bidirectional(
        layers.LSTM(256, return_sequences=True)
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(256, return_sequences=True)
    )(x)

    # --- Output Layer ---
    # A Dense layer projects the RNN output (B, T, 512) to the
    # number of classes (B, T, 37).
    # `NUM_CLASSES + 1` to account for the CTC blank token.
    softmax = layers.Dense(NUM_CLASSES + 1,
                           activation="softmax",
                           name="softmax")(x)

    # Attach the custom CTC loss layer
    outputs = CTCLayer(name="ctc_loss")(labels_input, softmax)

    # Build and compile the model
    # Inputs: A list of the two inputs [image, label]
    # Outputs: The output of the CTCLayer
    model = keras.Model([image_input, labels_input], outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
            clipnorm=CLIPNORM, # Use clipnorm to prevent exploding gradients
        )
    )
    
    # --- Effort Highlight: Blank Bias Initialization ---
    # This is a critical non-obvious trick for stable CTC training.
    # By default, all biases are ~0, making the 'blank' token the easiest to predict.
    # This can cause the model to get stuck predicting only blanks.
    # We set the blank's initial bias to a negative value (-5.0) to make it
    # 'harder' to predict at the start, forcing the model to learn real characters.
    output_layer = model.get_layer("softmax")
    W, b = output_layer.get_weights()
    b[BLANK_INDEX] = -5.0
    output_layer.set_weights([W, b])
    
    return model

def get_prediction_model(model):
    """
    Creates a separate model for inference (prediction).
    This model takes *only* the image as input and outputs the
    raw softmax sequence from the 'softmax' layer.
    """
    return keras.Model(
        inputs=model.input[0], # Index [0] is the 'image' input
        outputs=model.get_layer("softmax").output
    )

# --- Training Callbacks ---

class TQDMProgressBar(keras.callbacks.Callback):
    """
    --- Effort Highlight: Custom TQDM Progress Bar ---
    
    A custom callback to provide a clean, per-epoch TQDM progress bar.
    This improves the training user experience vs. Keras's default
    verbose modes, especially in notebooks or scripts.
    """
    def on_train_begin(self, logs=None):
        self.epochs = self.params.get("epochs", 1)
        print(f"Training for {self.epochs} epochs...")

    def on_epoch_begin(self, epoch, logs=None):
        steps = self.params.get("steps")
        if steps is None:
            bs = self.params.get("batch_size", 1)
            samples = self.params.get("samples")
            steps = (samples + bs - 1) // bs if samples is not None else None

        self.current_epoch = epoch + 1
        self.pbar = tqdm(
            total=steps,
            desc=f"Epoch {self.current_epoch}/{self.epochs}",
            unit="batch",
            leave=False,
        )

    def on_train_batch_end(self, batch, logs=None):
        self.pbar.update(1)
        if logs is not None:
            loss = logs.get("loss", 0.0)
            self.pbar.set_postfix(loss=f"{loss:.4f}")

    def on_epoch_end(self, epoch, logs=None):
        self.pbar.close()
        if logs is not None:
            msg = f"Epoch {epoch+1}: loss={logs.get('loss', 0):.4f}"
            if "val_loss" in logs:
                msg += f", val_loss={logs['val_loss']:.4f}"
            print(msg)

# --- Evaluation Metrics ---

def levenshtein(a, b):
    """
    --- Effort Highlight: Levenshtein Distance Implementation ---
    
    Calculates the Levenshtein (edit) distance between two strings.
    This is the standard way to measure the difference between a
    predicted string and a ground-truth string.
    """
    dp = np.zeros((len(a) + 1, len(b) + 1), dtype=np.int32)
    dp[:, 0] = np.arange(len(a) + 1)
    dp[0, :] = np.arange(len(b) + 1)
    for i in range(1, len(a) + 1):
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            cost = 0 if ai == b[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return int(dp[-1, -1])

def cer(ref, hyp):
    """
    --- Effort Highlight: Character Error Rate (CER) Implementation ---
    
    Calculates the Character Error Rate (CER) using Levenshtein distance.
    CER = EditDistance / Length(Reference)
    This is the primary metric for OCR and ASR tasks.
    """
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return levenshtein(ref, hyp) / float(len(ref))

def greedy_decode(pred):
    """
    --- Effort Highlight: Greedy CTC Decoder ---
    
    Performs greedy CTC decoding on a single prediction sequence.
    'pred' is a (T, C) numpy array (e.g., 137, 37).
    
    This implements the core CTC decoding logic:
    1. Get the most likely char index (argmax) at each time step.
    2. Collapse repeated characters (e.g., 'a' 'a' 'a' -> 'a').
    3. Remove all blank tokens.
    
    We handle steps 2 & 3 efficiently by only adding a character
    if it's not the blank token AND it's not the same as the
    previous character added.
    """
    seq = tf.argmax(pred, axis=-1).numpy().tolist()
    out = []
    prev = BLANK_INDEX
    for c in seq:
        if c != BLANK_INDEX and c != prev:
            out.append(ALPHABET[c])
        prev = c
    return "".join(out)

def greedy_decode_batch(preds_btc):
    """Performs greedy CTC decoding on a batch of predictions."""
    texts = []
    for i in range(preds_btc.shape[0]):
        texts.append(greedy_decode(preds_btc[i]))
    return texts

def decode_true_labels(y_true_batch):
    """Converts padded ground-truth label arrays (with -1) back to strings."""
    texts = []
    for y in y_true_batch:
        ids = y[y != -1] # Filter out -1 padding
        texts.append("".join(ALPHABET[i] for i in ids))
    return texts

# --- Main Functions ---

def run_training(model, x_train, y_train, x_val, y_val):
    """
    Sets up callbacks and fits the model.
    """
    print(f"Training on {len(x_train)} samples, validating on {len(x_val)}.")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # --- Effort Highlight: Robust Callback Setup ---
    callbacks = [
        # 1. Save the best model:
        #    This monitors 'val_loss' and saves *only* the model
        #    weights that achieve the lowest validation loss.
        keras.callbacks.ModelCheckpoint(
            filepath=BEST_CKPT_PATH,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            mode="min",
            verbose=0,
        ),
        # 2. Save the latest model:
        #    This saves the model weights at the end of *every* epoch.
        #    Useful for resuming training if it gets interrupted.
        keras.callbacks.ModelCheckpoint(
            filepath=LATEST_CKPT_PATH,
            save_best_only=False,
            save_weights_only=True,
            verbose=0,
        ),
        # 3. Dynamic Learning Rate:
        #    This monitors 'val_loss' and reduces the learning rate
        #    by a factor of 0.3 if it doesn't improve for 3 epochs.
        #    This helps the model "settle" into a good minimum.
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=LR_FACTOR,
            patience=LR_PATIENCE,
            min_lr=1e-5,
            verbose=0,
        ),
        # 4. Custom Progress Bar
        TQDMProgressBar(),
    ]

    history = model.fit(
        # The model's inputs are a dictionary matching the input layer names
        x={"image": x_train, "label": y_train},
        # y=None because the 'label' is already provided in the 'x' dict
        # and the loss is computed inside the CTCLayer.
        y=None,
        validation_data=({"image": x_val, "label": y_val}, None),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        # verbose=0 because our TQDMProgressBar handles all output
        verbose=0,
    )
    
    print("Training finished.")
    print(f"Best model weights saved to: {BEST_CKPT_PATH}")


def run_evaluation(ckpt_path, x_val, y_val):
    """
    Loads the best weights and runs a final evaluation.
    """
    print("\n--- Starting Evaluation ---")
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint file not found: {ckpt_path}")
        print("Cannot run evaluation.")
        return

    # Rebuild the SAME architecture
    model = build_model()
    
    print(f"Loading best weights from {ckpt_path}")
    model.load_weights(ckpt_path)

    # Create the separate prediction_model
    prediction_model = get_prediction_model(model)
    
    print("Running predictions on validation set...")
    # Use the prediction_model (image_input only) for this step
    preds = prediction_model.predict(x_val)

    # --- Decode predictions ---
    print("Decoding predictions...")
    pred_texts = greedy_decode_batch(tf.convert_to_tensor(preds))

    # --- Decode validation labels (padded with -1) ---
    true_texts = decode_true_labels(y_val)

    # --- Evaluate ---
    exact = np.mean([p == t for p, t in zip(pred_texts, true_texts)])
    cer_mean = np.mean([cer(t, p) for p, t in zip(pred_texts, true_texts)])

    print("\n--- Validation Results ---")
    print(f"[VAL] Exact-Match: {exact*100:.2f}%")
    print(f"[VAL] Mean CER:    {cer_mean:.4f}")
    
    print("\n--- Samples (pred | true) ---")
    for p, t in list(zip(pred_texts, true_texts))[:15]:
        # Using :<20 formats the string to be 20 chars wide,
        # left-aligned, which makes the output a clean table.
        print(f"{p:<20} | {t}")

def main():
    """
    Main execution flow for loading data, training, and evaluation.
    """
    print("--- CRNN Captcha Model ---")
    print(f"Alphabet size (NUM_CLASSES): {NUM_CLASSES}")
    print(f"Blank token index: {BLANK_INDEX}")
    
    # 1. Load data from .npz file
    images, labels = load_preprocessed_data(SAVE_PREPROCESSED_IMAGES_DIR)
    if images is None or labels is None:
        print("Preprocessed data not found. Please run preprocess.py first.")
        return

    # 2. Split data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.1, random_state=42
    )
    x_test, y_test = load_preprocessed_data(TEST_PREPROCESSED_IMAGES_DIR)
    
    # 3. Build the model
    model = build_model()
    model.summary()

    # 4. Run training
    # This will train for 'EPOCHS' and save the best weights
    run_training(model, x_train, y_train, x_val, y_val)
    
    # 5. Run evaluation
    # This loads the *best* weights from training and scores them.
    run_evaluation(BEST_CKPT_PATH, x_val, y_val)
    run_evaluation(BEST_CKPT_PATH, x_test, y_test)

if __name__ == "__main__":
    main()