import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sentence = '''if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.'''
seq_length = 9

# Preprocess the data
idx2char = list(set(sentence))
char2idx = {c: i for i, c in enumerate(idx2char)}  ## key: value

x_data = []
y_data = []
for i in range(0, len(sentence) - seq_length):
    x_str = sentence[i:i + seq_length]
    y_str = sentence[i + 1:i + seq_length + 1]
    x_data.append([char2idx[c] for c in x_str])
    y_data.append([char2idx[c] for c in y_str])

# Hyper parameters
hidden_size = len(char2idx)
batch_size = len(x_data)  ## 중요!
x_one_hot = tf.keras.utils.to_categorical(x_data, num_classes=hidden_size)
y_one_hot = tf.keras.utils.to_categorical(y_data, num_classes=hidden_size)
input_dim = x_one_hot.shape[2]

# Make the model
tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.LSTM(units=hidden_size,
                                  input_shape=(seq_length, input_dim),
                                  return_sequences=True))
'''
LSTM 레이어를 여러 쌓아올릴 때는 return_sequences=True 필요
'''

tf.model.add(tf.keras.layers.LSTM(units=hidden_size, return_sequences=True))
tf.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=hidden_size, activation="softmax")))
tf.model.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.Adam(lr=0.1), metrics=['accuracy'])

# Train
tf.model.fit(x_one_hot, y_one_hot, batch_size=batch_size, epochs=100)

# Summary
tf.model.summary()

# Predict
predictions = tf.model.predict(x_one_hot)
for i, prediction in enumerate(predictions):
    index = np.argmax(prediction, axis=1)
    if i == 0:
        print(''.join([idx2char[t] for t in index]), end='')
    else:
        print(idx2char[index[-1]], end='')

# Evaluate
score = tf.model.evaluate(x_one_hot, y_one_hot)
print("Loss: ", score[0])
print("Accuracy:", score[1])