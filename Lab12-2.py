import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sample = " if you want you"

# Preprocess the data
idx2char = list(set(sample))
char2idx = {c: i for i, c in enumerate(idx2char)} ## key: value
sample_idx = [char2idx[c] for c in sample]
x_data = [sample_idx[:-1]] ## X data sample (0 ~ n-1)
y_data = [sample_idx[1:]] ## Y label sample (1 ~ n)

# Hyper parameters
hidden_size = len(char2idx)
input_dim = len(char2idx)
batch_size = 1
sequence_length = len(sample) - 1 ## X data가 마지막 원소를 제외시키기 때문

x_one_hot = tf.keras.utils.to_categorical(x_data, num_classes=hidden_size)
y_one_hot = tf.keras.utils.to_categorical(y_data, num_classes=hidden_size)

# Make the model
tf.model = tf.keras.Sequential()
cell = tf.keras.layers.LSTMCell(units=hidden_size,
                                input_shape=(sequence_length, input_dim))
'''
- BasicRNNCell 또는 GRUCell을 써도 됨
- LSTM은 Long Short-Term Memory units이며, 긴 시퀀스 기억 가능
'''

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
