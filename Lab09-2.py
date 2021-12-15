import tensorflow as tf
import numpy as np
import os
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

# Make the model
tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=2, input_dim=2, activation='sigmoid'))
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=2, activation='sigmoid'))
tf.model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=0.1), metrics=['accuracy'])

# Summary
tf.model.summary()

# Make callback
log_dir = os.path.join(".", "logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorflow_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train
history = tf.model.fit(x_data, y_data, epochs=10000, callbacks=[tensorflow_callback])

# Predict
print('Prediction: \n', tf.model.predict(x_data))
score = tf.model.evaluate(x_data, y_data)
print('Accuracy: ', score[1])