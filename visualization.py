import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from tqdm import tqdm

# Paths
model_path = r'C:\Users\gowri\PycharmProjects\Hematology\split_data\vgg_blood_model.h5'
dataset_path = r'C:\Users\gowri\PycharmProjects\Hematology\dataset'

# Load model
model = load_model(model_path)

# Grad-CAM setup
grad_model = Model(
    inputs=model.inputs,
    outputs=[model.get_layer('block5_conv3').output, model.output]
)

# Function to generate Grad-CAM heatmap
def get_gradcam_heatmap(img_array, predicted_class):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0)

    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    heatmap = cv2.resize(np.uint8(255 * heatmap), (224, 224))
    return heatmap

# Create output directory if not exists
os.makedirs(output_path, exist_ok=True)

# Loop through each class
for blood_class in os.listdir(dataset_path):
    class_folder = os.path.join(dataset_path, blood_class)
    if not os.path.isdir(class_folder):
        continue

    print(f"🔎 Processing class: {blood_class}")

    heatmaps = []
    for img_name in tqdm(os.listdir(class_folder)):
        try:
            img_path = os.path.join(class_folder, img_name)
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            preds = model.predict(img_array)
            pred_class = np.argmax(preds[0])

            heatmap = get_gradcam_heatmap(img_array, pred_class)
            heatmaps.append(heatmap)
        except Exception as e:
            print(f"⚠ Skipped {img_name} due to error: {e}")

    if len(heatmaps) > 0:
        avg_heatmap = np.mean(heatmaps, axis=0).astype(np.uint8)
        avg_heatmap_color = cv2.applyColorMap(avg_heatmap, cv2.COLORMAP_JET)

        save_path = os.path.join(output_path, f"{blood_class}_pattern.jpg")
        cv2.imwrite(save_path, avg_heatmap_color)
        print(f"✅ Saved pattern for {blood_class} at: {save_path}")
    else:
        print(f"❌ No valid heatmaps found for class: {blood_class}")