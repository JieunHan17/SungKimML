import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

x_data = np.array(
    [[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4], [4, 1, 5, 5], [1, 7, 5, 5], [1, 2, 5, 6], [1, 6, 6, 6], [1, 7, 7, 7]])
y_data = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]])

# Make the model
tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=3, input_dim=4, activation='softmax', use_bias=True))
tf.model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=0.1), metrics=['accuracy'])

# Summary
tf.model.summary()

# Train
history = tf.model.fit(x_data, y_data, epochs=2000)

# Predict
a = tf.model.predict(np.array([[1, 11, 7, 9]]))
print(a) ## 확률 값이 나옴

# One-hot encoding(최강자 추출)
print(tf.keras.backend.eval(tf.argmax(a, axis=1)))  ## axis: 어떤 걸 축으로 할 것인가? 가장 안쪽을 축으로 하면 0부터 시작
