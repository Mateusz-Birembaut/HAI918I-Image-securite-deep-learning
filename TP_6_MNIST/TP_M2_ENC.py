# Importation deep learning
import tensorflow as tf
from tensorflow import keras

# Importation data science
import numpy as np
import matplotlib.pyplot as plt

# Chargement des données MNIST
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()

# Redimensionnement et normalisation
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

print("x_train :", x_train.shape)
print("x_test :", x_test.shape)
print("Avant normalisation : min =", x_train.min(), "max =", x_train.max())

# Architecture de l'autoencodeur
input_img = keras.layers.Input(shape=(28,28,1))

# Encodeur
x = keras.layers.Conv2D(8, (3,3), activation='relu', padding='same')(input_img)
x = keras.layers.MaxPooling2D((2,2), padding='same')(x)
x = keras.layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
encoded = keras.layers.MaxPooling2D((2,2), padding='same')(x)

# Décodeur
x = keras.layers.Conv2DTranspose(16, (3,3), strides=2, activation='relu', padding='same')(encoded)
x = keras.layers.Conv2DTranspose(8, (3,3), strides=2, activation='relu', padding='same')(x)
decoded = keras.layers.Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)

# Modèle autoencodeur
autoencoder = keras.models.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

# Paramètres d'entraînement
batch_size = 512
epochs = 1

# Entraînement
history = autoencoder.fit(
    x_train, x_train,
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    validation_data=(x_test, x_test)
)

# Sauvegarde du modèle
autoencoder.save("mnist_autoencoder.keras")

# Rechargement du modèle (exemple)
autoencoder = keras.models.load_model("mnist_autoencoder.keras")

# Évaluation sur le test
decoded_imgs = autoencoder.predict(x_test)

# Affichage de quelques images originales et reconstruites
n = 20
plt.figure(figsize=(20,4))
for i in range(n):
    # Images originales
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28,28), cmap='gray')
    plt.axis('off')
    
    # Images reconstruites
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28,28), cmap='gray')
    plt.axis('off')
plt.show()

# Historique d'entraînement
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Loss de reconstruction")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
