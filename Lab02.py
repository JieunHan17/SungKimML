import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print('## = ', tf.__version__)

# Build graph (tensors) using TensorFlow operations
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

print("node1: ", node1, "node2: ", node2)
print("node3: ", node3)

# Run graph and update variables in the graph(and return values)
tf.print(node1, node2, node3)

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)


@tf.function
def adder_node(a, b):
    return a + b


print(adder_node(a, b))
