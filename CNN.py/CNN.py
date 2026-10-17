import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
tf.keras.utils.set_random_seed(42)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
class_names = ("airplane", "automobile", "bird", "cat", "deer",
"dog", "frog", "horse", "ship", "truck")
y_train = y_train.squeeze()
y_test = y_test.squeeze()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
print("Train:", x_train.shape, "Test:", x_test.shape)
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
ax.imshow(x_train[i])
ax.set_title(class_names[int(y_train[i])], fontsize=9)
ax.axis("off")
plt.tight_layout()
plt.show()