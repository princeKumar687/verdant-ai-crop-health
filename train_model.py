"""
Train a lightweight MobileNetV2 classifier on a folder-based PlantVillage dataset.
Expected structure:
dataset/
  Tomato___Early_blight/
    image1.jpg
  Tomato___healthy/
    image2.jpg
...one folder per class.
Use only a dataset you are licensed to use. This script saves models/crop_disease_model.keras
and models/class_names.json for app.py.
"""
import os, json, tensorflow as tf
from tensorflow.keras import layers

DATA_DIR = os.environ.get("CROP_DATA_DIR", "dataset")
OUT_DIR = "models"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = int(os.environ.get("EPOCHS", "15"))

if not os.path.isdir(DATA_DIR):
    raise SystemExit(f"Dataset folder not found: {DATA_DIR}")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training", seed=42,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation", seed=42,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE
)
class_names = train_ds.class_names
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(lambda x,y:(tf.cast(x,tf.float32)/255.0,y)).prefetch(AUTOTUNE)
val_ds = val_ds.map(lambda x,y:(tf.cast(x,tf.float32)/255.0,y)).prefetch(AUTOTUNE)

base = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3), include_top=False, weights="imagenet"
)
base.trainable = False
inputs = tf.keras.Input(shape=(224,224,3))
x = layers.RandomFlip("horizontal")(inputs)
x = layers.RandomRotation(0.08)(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
os.makedirs(OUT_DIR, exist_ok=True)
model.save(os.path.join(OUT_DIR, "crop_disease_model.keras"))
with open(os.path.join(OUT_DIR, "class_names.json"), "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)
print("Saved model and class labels to models/.")
