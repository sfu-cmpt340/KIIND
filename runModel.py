import tensorflow as tf
import os
from tensorflow.keras.models import load_model


model_path = os.path.join('models', 'imageclassifier5.h5')
model = load_model(model_path)
model.predict()
