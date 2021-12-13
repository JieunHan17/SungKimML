import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

x_data = np.array([[73, 80, 75],
                   [93, 88, 93],
                   [89, 91, 90],
                   [96, 98, 100],
                   [73, 66, 70]])
y_data = np.array([[152],
                   [185],
                   [180],
                   [196],
                   [142]])

# Make the model
tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=3, activation='linear'))
tf.model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=1e-5))

# Summary
tf.model.summary()

# Train
history = tf.model.fit(x_data, y_data, epochs=100)

# Predict
x_test = np.array([[90, 88, 93], [70, 70, 70]])
print(tf.model.predict(x_test))
