from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np

# Données d'entrée (exemple)
# Les données d'entrée doivent être un tableau 3D avec les dimensions (nombre d'échantillons, longueur de la séquence, nombre de caractéristiques)
X_train = np.random.rand(100, 10, 2)  # 100 échantillons, séquences de longueur 10, 2 caractéristiques

# Données de sortie (exemple)
# Les données de sortie doivent être un tableau 2D avec les dimensions (nombre d'échantillons, nombre de classes)
y_train = np.random.randint(0, 2, size=(100, 1))  # 100 échantillons, classes binaires

# Création du modèle
model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(10, 2)))  # Couche LSTM bidimensionnelle
model.add(Dropout(0.2))
model.add(LSTM(units=64))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))  # Couche de sortie avec activation sigmoïde pour la classification binaire

# Compilation du modèle
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entraînement du modèle
model.fit(X_train, y_train, epochs=10, batch_size=32)

# À ce stade, votre modèle est entraîné et prêt à être utilisé pour la prédiction.
