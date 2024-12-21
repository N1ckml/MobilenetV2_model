import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import psutil
import time
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Input
from tensorflow.keras.applications import MobileNetV3Large

# Montar Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Configuración de datos
image_size = 224
batch_size = 8

# Directorios de datos
train_dir = '/content/drive/My Drive/dataset_Paddy/train_images'
val_dir = '/content/drive/My Drive/dataset_Paddy/val_images'
test_dir = '/content/drive/My Drive/dataset_Paddy/test_images'

# Cargar datos
def load_dataset(directory):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=(image_size, image_size),
        batch_size=batch_size
    )

train_ds = load_dataset(train_dir)
val_ds = load_dataset(val_dir)
test_ds = load_dataset(test_dir)

class_names = train_ds.class_names

# Prefetch para optimizar el rendimiento
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# Configuración del modelo
input_tensor_shape = (224, 224, 3)
pre_model = MobileNetV3Large(include_top=False, pooling='avg', input_shape=input_tensor_shape, weights='imagenet')
for layer in pre_model.layers:
    layer.trainable = False

model = Sequential([
    pre_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dense(len(class_names), activation='softmax')
])

model.summary()

# Compilación del modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Entrenamiento del modelo
callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=5, verbose=1)
start_time = time.time()
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[callback]
)
training_time = time.time() - start_time

# Evaluación del modelo
results = model.evaluate(test_ds, batch_size=batch_size)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# Predicciones
true_labels = []
predicted_labels = []

for images, labels in test_ds:
    true_labels.extend(labels.numpy())
    predictions = model.predict(images, verbose=0)
    predicted_labels.extend(np.argmax(predictions, axis=1))

import pandas as pd
# Métricas de evaluación
true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

cm = confusion_matrix(true_labels, predicted_labels)
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax)
plt.xticks(rotation=90)
plt.show()

print("\nClassification Report:\n")
print(classification_report(true_labels, predicted_labels, target_names=class_names))

# Gráficas de precisión y pérdida
pd.DataFrame(history.history)[['accuracy', 'val_accuracy']].plot()
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

pd.DataFrame(history.history)[['loss', 'val_loss']].plot()
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# Recursos usados
gpu_memory = tf.config.experimental.get_memory_info('GPU:0') if tf.config.list_physical_devices('GPU') else None
ram_usage = psutil.virtual_memory().used / (1024 ** 3)  # Convertir a GB

print(f"Training Time: {training_time:.2f} seconds")
if gpu_memory:
    print(f"GPU Memory Used: {gpu_memory['peak'] / (1024 ** 3):.2f} GB")
print(f"RAM Used: {ram_usage:.2f} GB")

