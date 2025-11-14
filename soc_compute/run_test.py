import os
import cv2
import numpy as np
import pandas as pd
import csv
from keras.models import load_model

from train import (
    CNNModel,
    prepare_character_image,
)

# If segmentation functions are in a separate file, import them instead
from train import (
    segment_captcha_into_chars,
    remove_black_lines,
    enhance_contrast_ycrcb,
)

# ============================================================
# EDIT DISTANCE
# ============================================================

def levenshtein(a, b):
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]

    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j

    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            )

    return dp[m][n]

# ============================================================
# TESTING PIPELINE
# ============================================================

def test_model(model_path, test_folder, output_csv="test_results.csv"):
    cnn = CNNModel()
    model = load_model(model_path)

    results = []

    for filename in os.listdir(test_folder):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(test_folder, filename)
        img = cv2.imread(img_path)

        if img is None:
            continue

        true_label = os.path.splitext(filename)[0].split("-")[0]

        no_lines = remove_black_lines(img, true_label, save=False)
        enhanced = enhance_contrast_ycrcb(no_lines, true_label, save=False)
        chars = segment_captcha_into_chars(enhanced)

        processed = [prepare_character_image(c) for c in chars]
        processed = np.array(processed)

        pred_vecs = model.predict(processed)
        pred_chars = cnn.decode_predictions(pred_vecs)
        pred_str = "".join(pred_chars)
        confs = cnn.get_confidences(pred_vecs)

        dist = levenshtein(true_label, pred_str)

        results.append({
            "filename": filename,
            "true_label": true_label,
            "predicted": pred_str,
            "edit_distance": dist,
            "num_segments": len(chars),
            "num_preds": len(pred_chars),
            "avg_confidence": float(np.mean(confs)),
            "min_confidence": float(np.min(confs)),
            "max_confidence": float(np.max(confs)),
        })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Test results saved to {output_csv}")

    return df


def main():
    model_path = "./model_weights/cnn_character_model.keras"
    test_path = "./test/enhanced_contrast"

    test_model(model_path, test_path)


if __name__ == "__main__":
    main()
