import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Hyper parameters
hidden_size = 5
input_dim = 5
batch_size = 1
sequence_length = 6

# Preprocess the data
idx2char = ['h', 'i', 'e', 'l', 'o']  ## h=0, i=1, e=2, l=3, o=4
x_data = [[0, 1, 0, 2, 3, 3]]
x_one_hot = np.array([[[1, 0, 0, 0, 0],  # h 0
                       [0, 1, 0, 0, 0],  # i 1
                       [1, 0, 0, 0, 0],  # h 0
                       [0, 0, 1, 0, 0],  # e 2
                       [0, 0, 0, 1, 0],  # l 3
                       [0, 0, 0, 1, 0]]],  # l 3
                     dtype=np.float32)
y_data = [[1, 0, 2, 3, 3, 4]]
y_one_hot = tf.keras.utils.to_categorical(y_data, num_classes=hidden_size)

# Make the model
tf.model = tf.keras.Sequential()
cell = tf.keras.layers.LSTMCell(units=hidden_size,
                                input_shape=(sequence_length, input_dim))  ## BasicRNNCell 또는 GRUCell을 써도 됨
tf.model.add(tf.keras.layers.RNN(cell=cell, return_sequences=True))
tf.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=hidden_size, activation="softmax")))
tf.model.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.Adam(lr=0.1), metrics=['accuracy'])

# Train
tf.model.fit(x_one_hot, y_one_hot, batch_size=batch_size, epochs=50)

# Summary
tf.model.summary()

# Predict
predictions = tf.model.predict(x_one_hot)
for prediction in predictions:
    print(prediction)
    result_str = [idx2char[c] for c in np.argmax(prediction, axis=1)]
    print("\tPrediction str: ", ''.join(result_str))
