import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# X and Y data
x_train = [1, 2, 3]
y_train = [1, 2, 3]

# Make the model
tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=1)) ## units: 결과값의 개수, input_dim: 독립변수의 개수

# Minimize
sgd = tf.keras.optimizers.SGD(learning_rate=0.01)
tf.model.compile(loss='mse', optimizer=sgd)  ## mse = mean square error

# Train
tf.model.fit(x_train, y_train, epochs=200)  ## epoch: input layer부터 output layer까지 Forward, Backward를 진행하는것

# Predict
print(tf.model.predict([4]))

# Summary
tf.model.summary()
