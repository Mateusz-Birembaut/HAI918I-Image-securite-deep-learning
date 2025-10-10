# Importation deep learning
import tensorflow as tf
from tensorflow import keras

# Importation data science
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Importation utilitaire
import sys, os
from importlib import reload

# Chargement des données
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Redimensionnement
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print("x_train : ", x_train.shape)
print("y_train : ", y_train.shape)
print("x_test : ", x_test.shape)
print("y_test : ", y_test.shape)

print("Avant la normalisation : min = {}, max = {}".format(x_train.min(), x_train.max()))

# Normalisation des pixels
x_max = x_train.max()
x_train = x_train / x_max
x_test = x_test / x_max

print("Après la normalisation : min = {}, max = {}".format(x_train.min(), x_train.max()))

# Instanciation du modèle
model = keras.models.Sequential()

# Ajout des différentes couches selon l'architecture
model.add(keras.layers.Input((28, 28, 1)))

model.add(keras.layers.Conv2D(32, (3, 3), activation = 'relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))

model.add(keras.layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(300, activation = 'relu'))

model.add(keras.layers.Dense(10, activation = 'softmax'))

model.summary()

# Compilation du modèle
model.compile(optimizer = 'adam',
       loss = 'sparse_categorical_crossentropy',
       metrics = ['accuracy'])
	  
# Paramètrres d'entraînement	  
batch_size = 512
epochs = 16

# Lancement de l'entraînement
history = model.fit(x_train, y_train,
          batch_size = batch_size,
          epochs = epochs,
          verbose = True,
          validation_data = (x_test, y_test))

# Test du modèle sur le jeu d'évaluation
score = model.evaluate(x_test, y_test, verbose = 0)

print(f"Test loss : {score[0]:4.4f}")
print(f"Test accuracy : {score[1]:4.4f}")

model.save_weights("mnist.weights.h5")

# Historique de la précision du modèle sur les jeux d'entraînement et de validation
model.metrics_names
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Model accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(['train', 'validation'], loc = 'upper left')
plt.show()

# Matrice de confusion
class Estimator:
  _estimator_type = ''
  _classes = []
  def __init__(self, model, classes):
    self.model = model
    self._estimator_type = 'classifier'
    self._classes = classes
  def predict(self, X):
    y_prob = self.model.predict(X)
    y_pred = y_prob.argmax(axis = 1)
    return y_pred
  
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

classifier = Estimator(model, class_names)

ConfusionMatrixDisplay.from_estimator(classifier, x_test, y_test)
