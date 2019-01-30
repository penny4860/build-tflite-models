
import tensorflow as tf
print(tf.__version__)
interpreter = tf.contrib.lite.Interpreter(model_path=str("models/graph.lite"))
print(interpreter)


