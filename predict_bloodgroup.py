import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# === 1. Load the model once ===
model_path = r'C:\Users\gowri\PycharmProjects\Hematology\split_data\vgg_blood_model.h5'
model = load_model(model_path)

# === 2. Prepare class labels from training data ===
train_dir = r'C:\Users\gowri\PycharmProjects\Hematology\split_data\train'
img_size = (224, 224)

datagen = ImageDataGenerator(rescale=1. / 255)
train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

class_indices = train_gen.class_indices
labels = {v: k for k, v in class_indices.items()}  # reverse mapping

# === 3. Define a prediction function ===
def predict_blood_group(test_image_path):
    if not os.path.exists(test_image_path):
        raise FileNotFoundError(f"Image not found: {test_image_path}")

    # Load and preprocess the image
    img = image.load_img(test_image_path, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class = labels[predicted_class_index]

    return predicted_class
