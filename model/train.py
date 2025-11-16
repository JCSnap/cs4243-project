import csv
import cv2
import glob
import itertools
import json
import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow

from keras import layers
from sklearn.cluster import KMeans
from typing import List, Dict, Tuple

Image = np.ndarray

class CNNModel:
    def __init__(self, img_height=80, img_width=80, charset="0123456789abcdefghijklmnopqrstuvwxyz"):
        self.img_height = img_height
        self.img_width = img_width
        self.charset = charset
        self.num_classes = len(charset)
        self.char_to_idx = {char: idx for idx, char in enumerate(charset)}
        self.idx_to_char = {idx: char for idx, char in enumerate(charset)}
        self.model = None

    def build_model(self):
        # Input layer, H=80 x W=80 x 1
        inputs = layers.Input(shape=(self.img_height, self.img_width, 1))

        # Normalize pixel values
        x = layers.Rescaling(scale=1.0 / 255)(inputs)

        # Block 1
        x = layers.Conv2D(32, (3, 3), activation='relu')(x)
        x = layers.Conv2D(32, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        # Block 2
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        # Block 3
        x = layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        # Flatten and dense layers
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)

        # Output layer: single character
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        # Create the model
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        return self.model
    
    def compile_model(self, model, learning_rate=0.001):
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=['categorical_accuracy']
        )
        return model
    
    def encode_labels(self, labels):
        encoded_labels = [self.char_to_idx[label.lower()] for label in labels]
        return keras.utils.to_categorical(encoded_labels, num_classes=self.num_classes)

    def decode_predictions(self, predictions):
        decoded = []
        pred_indices = np.argmax(predictions, axis=1)

        for idx in pred_indices:
            decoded.append(self.idx_to_char[idx])

        return decoded

    def get_confidence(self, predictions):
        return np.max(predictions, axis=1)

class HyperparameterValidator:
    def __init__(self, test_images_path: str, results_dir: str="./validation_results"):
        self.test_images_path = test_images_path
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        # Load test data
        self.test_chars = self._load_test_data()

    def _load_test_data(self):
        # Load and preprocess test data
        test_full_captchas: List[Image] = []
        test_full_labels: List[str] = []

        for filename in os.listdir(self.test_images_path):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            image_path = os.path.join(self.test_images_path, filename)
            img = cv2.imread(image_path)
            
            if img is None:
                continue
            
            base_name = os.path.splitext(filename)[0]
            label = base_name.split("-")[0].lower()
            
            test_full_captchas.append(img)
            test_full_labels.append(label)

        test_chars: Dict[str, List[Tuple[Image, str]]] = {}

        for i in range(len(test_full_captchas)):
            full_captcha_image = test_full_captchas[i]
            full_captcha_label = test_full_labels[i]

            test_chars[full_captcha_label] = []

            captcha_chars: List[Image] = segment_captcha_into_chars(full_captcha_image)

            for j, char in enumerate(captcha_chars):
                refined_char = prepare_character_image(char, target_size=80)
                test_chars[full_captcha_label].append((refined_char, full_captcha_label[j]))

        return test_chars

    def validate_model(self, model: keras.Model, cnn: 'CNNModel'):
        predicted_labels = {}
        for captcha_label, char_label_pairs in self.test_chars.items():
                captcha_prediction_vectors = model.predict([img for (img, _) in char_label_pairs])
                captcha_predictions = cnn.decode_predictions(captcha_prediction_vectors)
                captcha_prediction_str = "".join(captcha_predictions)
                predicted_labels[captcha_label] = captcha_prediction_str
        
        final_predicted_labels = get_levenshtein_distances(predicted_labels)

        return final_predicted_labels
    
    def grid_search(self, hyperparams: Dict):
        all_results = {}

        for category in ["./none", "./cleaned", "./enhanced", "./normalised", "./augmented"]:
            X_train, y_train = self._load_training_data(category)

            param_names = list(hyperparams.keys())
            param_values = [hyperparams[name] for name in param_names]
            combinations = list(itertools.product(*param_values))

            results = []

            print(f"Testing {len(combinations)} hyperparameter combinations for {category}")

            for i, combo in enumerate(combinations):
                params = dict(zip(param_names, combo))
                print(f"\nCombination {i+1}/{len(combinations)}: {params}")

                model, cnn = self._create_model_with_params(params)

                history = self._train_model(model, cnn, X_train, y_train, params)

                validation_results = self.validate_model(model, cnn)

                combined_results = {
                    'hyperparameters': params,
                    'training_history': {
                        'final_loss': float(history.history['loss'][-1]),
                        'best_loss': float(min(history.history['loss'])),
                        'epochs_trained': len(history.history['loss'])
                    },
                    'validation_results': validation_results
                }

                results.append(combined_results)

                self._save_results(results, os.path.join(category, "grid_search_results.json"))
        
            best_result = max(results, key=lambda x: x['validation_results']['Avg_Distance'])

            all_results[category] = (results, best_result)

        return all_results

    def _load_training_data(self, train_data_path: str):
        images_path = os.path.join(train_data_path, "images.npy")
        labels_path = os.path.join(train_data_path, "labels.csv")
        
        X_train = np.load(images_path, allow_pickle=True)
        
        with open(labels_path, "r") as csv_file:
            csv_reader = csv.reader(csv_file)
            y_train = next(csv_reader)
        
        return X_train, y_train

    def _create_model_with_params(self, params: Dict):
        cnn = CNNModel()

        inputs = keras.layers.Input(shape=(80, 80, 1))
        x = keras.layers.Rescaling(scale=1.0 / 255)(inputs)
        
        # Get dropout rates
        dropout1, dropout2, dropout3 = params.get('dropout_rates', (0.2, 0.2, 0.3))

        # Block 1
        x = keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
        x = keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(dropout1)(x)
        
        # Block 2
        x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(dropout2)(x)
        
        # Block 3
        x = keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(dropout3)(x)
        
        # Dense layers
        x = keras.layers.Flatten()(x)
        dense_units = params.get('dense_units', 256)
        x = keras.layers.Dense(dense_units, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.4)(x)
        
        outputs = keras.layers.Dense(cnn.num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)

        # Compile with learning rate
        learning_rate = params.get('learning_rate', 0.001)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=['categorical_accuracy']
        )

        return model, cnn
    
    def _train_model(self, model: keras.Model, cnn: 'CNNModel', X_train, y_train, params: Dict):
        y_train_encoded = cnn.encode_labels(y_train)

        batch_size = params.get('batch_size', 64)
        epochs = params.get('epochs', 200)
        
        history = model.fit(
            X_train,
            y_train_encoded,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-7)
            ],
            verbose=0
        )
        
        return history
    
    def _save_results(self, results: List[Dict], filename: str):
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {filepath}")

def create_model(img_height=80, img_width=80, charset="0123456789abcdefghijklmnopqrstuvwxyz", learning_rate=0.001):
    cnn = CNNModel(img_height, img_width, charset)
    model = cnn.build_model()
    model = cnn.compile_model(model, learning_rate=learning_rate)
    return model, cnn

def remove_black_lines(img, label, save=True, output_dir="./test/black_removed", suffix="_LINES_REMOVED") -> Image | None:
    # Define black pixel mask
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([8, 8, 8])
    mask = cv2.inRange(img, lower_black, upper_black)

    # Inpaint to reconstruct the colors
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)
    cleaned = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # Save to cleaned folder
    if save:
        save_path = os.path.join(output_dir, f"{label}{suffix}.png")
        cv2.imwrite(save_path, cleaned)
    return cleaned

def enhance_contrast_ycrcb(img, label, save=True, output_dir="./test/contrast_enhanced") -> Image | None:
    # Convert directly from BGR to YCrCb
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    # Equalize Y (luminance) channel
    y_eq = cv2.equalizeHist(y)

    # Merge and convert back to BGR
    ycrcb_eq = cv2.merge((y_eq, cr, cb))
    enhanced = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

    # Save result if needed
    if save:
        save_path = os.path.join(output_dir, f"{label}_LINES_REMOVED_ENHANCED.png")
        cv2.imwrite(save_path, enhanced)

    return enhanced

def prepare_character_image(img: Image, target_size: int = 80) -> Image:
    """
    Convert image to grayscale, upscale/pad to target_size while maintaining aspect ratio,
    and add channel dimension to make it target_size x target_size x 1.
    
    Args:
        img: Input image (can be BGR, RGB, or grayscale)
        target_size: Target size for both width and height (default: 80)
        
    Returns:
        Processed image as numpy array with shape (target_size, target_size, 1)
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Get current dimensions
    h, w = gray.shape[:2]
    
    # If already the target size, just add channel dimension
    if h == target_size and w == target_size:
        gray = np.expand_dims(gray, axis=-1)
        return gray
    
    # Upscale and pad while maintaining aspect ratio
    if h > w:
        # Height is larger, scale based on height
        new_h = target_size
        new_w = int(w * target_size / h)
        if new_w == 0: new_w += 1
        img_resized = cv2.resize(gray, (new_w, new_h))
        pad_total = target_size - new_w
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        img_padded = cv2.copyMakeBorder(img_resized, 0, 0, pad_left, pad_right,
                                       cv2.BORDER_CONSTANT, value=255)
    else:
        # Width is larger or equal, scale based on width
        new_w = target_size
        new_h = int(h * target_size / w)
        if new_h == 0: new_h += 1
        img_resized = cv2.resize(gray, (new_w, new_h))
        pad_total = target_size - new_h
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        img_padded = cv2.copyMakeBorder(img_resized, pad_top, pad_bottom, 0, 0,
                                       cv2.BORDER_CONSTANT, value=255)
    
    # Add channel dimension to make it (target_size, target_size, 1)
    img_padded = np.expand_dims(img_padded, axis=-1)
    
    return img_padded

# ============================================================
# HELPER FUNCTIONS FOR SEGMENTATION
# ============================================================

def _find_connected_components(img_rgb: np.ndarray, min_area: int = 50):
    """Return valid connected components (spatial separation)."""
    non_white_mask = ~((img_rgb[:,:,0] == 255) & 
                       (img_rgb[:,:,1] == 255) & 
                       (img_rgb[:,:,2] == 255))
    binary_img = non_white_mask.astype(np.uint8) * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    
    components = []
    for i in range(1, num_labels):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            mask = (labels == i)
            cropped = img_rgb.copy()
            cropped[~mask] = [255, 255, 255]
            components.append({
                'id': i,
                'mask': mask,
                'bbox': (x, y, w, h),
                'area': area,
                'centroid': centroids[i],
                'image': cropped[y:y+h, x:x+w]
            })
    return components, labels

def _cluster_by_color(img_rgb: np.ndarray, component: dict, min_pixels: int = 100):
    """Use color clustering within component to detect overlapping chars."""
    x, y, w, h = component['bbox']
    mask = component['mask']
    coords = np.where(mask)
    pixels = img_rgb[coords]
    if len(pixels) < min_pixels:
        return None
    best_sep = None
    best_score = 0
    for n_clusters in range(2, 6):
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(pixels)
            clusters = []
            for cid in range(n_clusters):
                mask_cid = cluster_labels == cid
                y_coords, x_coords = coords[0][mask_cid], coords[1][mask_cid]
                if len(y_coords) < 50:
                    continue
                mask_full = np.zeros(img_rgb.shape[:2], dtype=bool)
                mask_full[y_coords, x_coords] = True
                bbox = (x_coords.min(), y_coords.min(),
                        x_coords.max() - x_coords.min() + 1,
                        y_coords.max() - y_coords.min() + 1)
                clusters.append({
                    'mask': mask_full,
                    'bbox': bbox,
                    'area': len(y_coords),
                    'centroid': kmeans.cluster_centers_[cid]
                })
            if len(clusters) >= 2:
                sizes = [c['area'] for c in clusters]
                balance = min(sizes) / max(sizes)
                score = balance * len(clusters)
                if score > best_score:
                    best_score = score
                    best_sep = {'n_clusters': n_clusters, 'clusters': clusters, 'score': score}
        except:
            continue
    return best_sep

def _refine_with_morphology(cluster_info):
    """Apply morphological cleanup to separate characters."""
    refined = []
    for cluster in cluster_info:
        mask = cluster['mask']
        kernel = np.ones((3, 3), np.uint8)
        mask_uint8 = (mask * 255).astype(np.uint8)
        closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= 30:
                sub_mask = (labels == i)
                y_coords, x_coords = np.where(sub_mask)
                bbox = (x_coords.min(), y_coords.min(),
                        x_coords.max() - x_coords.min() + 1,
                        y_coords.max() - y_coords.min() + 1)
                refined.append({
                    'mask': sub_mask,
                    'bbox': bbox,
                    'area': area,
                    'centroid': centroids[i]
                })
    return refined

# ============================================================
# MAIN SEGMENTATION FUNCTION
# ============================================================

def segment_captcha_into_chars(img_path) -> List[Image]:
    """
    Segment a single captcha image into individual characters.
    
    Args:
        img_path: Path to the captcha image file (str) or image array (np.ndarray)
        
    Returns:
        List of character images (as numpy arrays)
    """
    # Handle both path string and image array
    if isinstance(img_path, str):
        # Read image from path
        img = cv2.imread(img_path)
        print(f"Reading {img_path}")
        if img is None:
            print(f"file {img_path} does not exist")
            return []
    else:
        # Assume it's already an image array
        img = img_path
        if img is None or not isinstance(img, np.ndarray):
            print("Not a path or Numpy array, error")
            return []
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Find connected components
    components, _ = _find_connected_components(img_rgb)
    components_sorted = sorted(components, key=lambda c: c['centroid'][0])
    
    detected_chars = []
    for comp in components_sorted:
        x, y, w, h = comp['bbox']
        area, aspect = comp['area'], w / (h + 1e-5)
        is_large = area > 400 or w > 40 or (area > 300 and aspect > 1.2)
        
        if is_large:
            # Try to separate overlapping characters using color clustering
            sep = _cluster_by_color(img_rgb, comp)
            if sep and len(sep['clusters']) > 1:
                refined = _refine_with_morphology(sep['clusters'])
                for r in sorted(refined, key=lambda c: c['bbox'][0]):
                    cx, cy, cw, ch = r['bbox']
                    char_mask = r['mask']
                    char_img = img_rgb.copy()
                    char_img[~char_mask] = [255, 255, 255]
                    char_crop = char_img[cy:cy+ch, cx:cx+cw]
                    detected_chars.append(char_crop)
                continue
        
        # Single character component
        detected_chars.append(comp['image'])
    
    print(f"Detected {len(detected_chars)} chars")
    return detected_chars

# def train_data_npy(images_path: str):
#     model, cnn = create_model(learning_rate=0.001)

#     #====================================================
#     # TODO: Read from the augmented folder
#     # TODO: Important, make sure each character formated into 80x80x1, NOT 80x80
#     #====================================================
#     images_path = os.path.join(images_path, "images.npy")
#     images = np.load(images_path, allow_pickle=True)
    
#     labels_path = os.path.join(images_path, "labels.csv")
#     with open(labels_path, "r") as csv_file:
#         csv_reader = csv.reader(csv_file)
#         y_train = next(csv_reader)  # Read the first row which contains all labels
    
#     X_train = []
#     for img in images:
#         processed_img = prepare_character_image(img, target_size=80)
#         X_train.append(processed_img)
    
#     X_train = np.array(X_train)
#     y_train_encoded = cnn.encode_labels(y_train)
#     y_train_encoded = np.array(y_train_encoded)

#     history = model.fit(
#         X_train, 
#         y_train_encoded, 
#         epochs=200, 
#         batch_size=64, 
#         callbacks=[
#             keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True), 
#             keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-7)
#         ]
#     )

#     # Save trained model
#     MODEL_SAVE_PATH = os.path.join(images_path, 'model_weights/cnn_character_model.keras')
#     model_dir = os.path.dirname(MODEL_SAVE_PATH)
#     os.makedirs(model_dir, exist_ok=True)
#     model.save(MODEL_SAVE_PATH)
#     print(f"Model saved to: {MODEL_SAVE_PATH}")

#     # Plot loss over epochs graph
#     plt.figure(figsize=(10, 6))
#     plt.plot(history.history['loss'], label='Training Loss')
#     plt.title('Model Training Loss Over Epochs')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

#     return model, cnn

# def test_data(test_images_path: str, model: keras.Model, cnn: CNNModel):
#     # Read test data
#     #====================================================
#     # TODO: Read from the test folder, hear assume images are the full OG NO PREPROC captcha image
#     # TODO: Important, make sure each character formated into 80x80x1, NOT 80x80
#     #====================================================
#     test_full_captchas: List[Image] = []
#     test_labels: List[str] = []
    
#     for filename in os.listdir(test_images_path):
#         if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#             continue
        
#         image_path = os.path.join(test_images_path, filename)
#         img = cv2.imread(image_path)
        
#         if img is None:
#             continue
        
#         # Extract label from filename (everything before the first "-")
#         base_name = os.path.splitext(filename)[0]
#         label = base_name.split("-")[0]
        
#         test_full_captchas.append(img)
#         test_labels.append(label)
    
#     predicted_labels = {}
#     for i in range(len(test_full_captchas)):
#         full_captcha_image = test_full_captchas[i]
#         full_captcha_label = test_labels[i]

#         # Remove black line
#         full_captcha_no_lines = remove_black_lines(full_captcha_image, full_captcha_label)

#         # Enhance contrast
#         full_captcha_high_contrast = enhance_contrast_ycrcb(full_captcha_no_lines, full_captcha_label)

#         # Segment captcha
#         captcha_chars = segment_captcha_into_chars(full_captcha_high_contrast)

#         refined_captcha_chars = []
#         for char in captcha_chars:
#             # Convert to grayscale, upscale/pad to 80x80, and add channel dimension
#             refined_captcha_char = prepare_character_image(char, target_size=80)
#             refined_captcha_chars.append(refined_captcha_char)

#         refined_captcha_chars = np.array(refined_captcha_chars)
#         captcha_prediction_vectors = model.predict(refined_captcha_chars)
#         captcha_predictions = cnn.decode_predictions(captcha_prediction_vectors)
#         captcha_prediction_str = "".join(captcha_predictions)

#         predicted_labels[full_captcha_label] = captcha_prediction_str

#     return predicted_labels

def get_levenshtein_distances(labels_to_pred_dict):
    results = {
        'Actual': [],
        'Predicted': [],
        'Distance': [],
        'Avg_Distance': -float('inf'),
        'Min_Distance': -float('inf'),
        'Max_Distance': -float('inf')
    }

    for actual, predicted in labels_to_pred_dict.items():
        m = len(actual)
        n = len(predicted)

        # Create a 2D array (matrix) to store the distances
        dp = [[0] * (n + 1) for _ in range(m+1)]

        # Initialize the first row and column
        for i in range(m + 1):
            dp[i][0] = i  # Cost of deleting all characters from actual to get an empty predicted
        for j in range(n + 1):
            dp[0][j] = j  # Cost of inserting all characters of predicted into an empty actual
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if actual[i - 1] == predicted[j - 1]:
                    cost = 0  # Characters are the same, no edit needed
                else:
                    cost = 1  # Characters are different, substitution needed

                dp[i][j] = min(
                    dp[i - 1][j] + 1,      # Deletion
                    dp[i][j - 1] + 1,      # Insertion
                    dp[i - 1][j - 1] + cost  # Substitution (cost is 0 if same, 1 if different)
                )

        distance = dp[m][n]

        results['Actual'].append(actual)
        results['Predicted'].append(predicted)
        results['Distance'].append(distance)

    distances_only = np.array(results['Distance'])
    results['Avg_Distance'] = np.mean(distances_only)
    results['Min_Distance'] = np.min(distances_only)
    results['Max_Distance'] = np.max(distances_only)

    return results

def images_to_npy_and_csv(images_path: str):
    full_captchas: List[Image] = []
    actual_labels: List[str] = []
    
    for filename in os.listdir(images_path):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        image_path = os.path.join(images_path, filename)
        img = cv2.imread(image_path)
        
        if img is None:
            continue
        
        # Extract label from filename (everything before the first "-")
        base_name = os.path.splitext(filename)[0]
        label = base_name.split("-")[0].lower()
        
        full_captchas.append(img)
        actual_labels.append(label)
    
    char_images = []
    char_labels = []
    for i in range(len(full_captchas)):
        full_captcha_image = full_captchas[i]
        full_captcha_label = actual_labels[i]

        # Segment captcha
        captcha_chars = segment_captcha_into_chars(full_captcha_image)

        for j in range(min(len(captcha_chars), len(full_captcha_label))):
            # Convert to grayscale, upscale/pad to 80x80, and add channel dimension
            refined_captcha_char = prepare_character_image(captcha_chars[j], target_size=80)
            char_images.append(refined_captcha_char)
            char_labels.append(full_captcha_label[j])

    char_images = np.array(char_images)
    char_labels = np.array(char_labels)
    np.save(os.path.join(images_path, "images.npy"), char_images)
    np.savetxt(os.path.join(images_path, "labels.csv"), char_labels, delimiter=",")

def main():
    np.random.seed(1)

    # Segmenting, already done
    # images_path = "./none"
    # images_to_npy_and_csv(images_path)

    # cleaned_images_path = "./cleaned"
    # images_to_npy_and_csv(cleaned_images_path)

    enhanced_images_path = "./enhanced"
    images_to_npy_and_csv(enhanced_images_path)

    validator = HyperparameterValidator(
        test_images_path="./test",
        results_dir="./validation_results"
    )

    hyperparams = {
        'learning_rate': [0.001, 0.0005, 0.0001],
        'batch_size': [32, 64, 128],
        'dropout_rates': [(0.2, 0.2, 0.3), (0.3, 0.3, 0.4), (0.2, 0.3, 0.5)],
        'dense_units': [128, 256, 512]
    }

    all_results = validator.grid_search(hyperparams=hyperparams)
    print(all_results)

if __name__ == "__main__":
    main()
