import tensorflow as tf

model=tf.keras.Sequential([
    tf.keras.layers.Dense(24,activation="relu"),
    tf.keras.layers.Dense(24,activation="relu"),
    tf.keras.layers.Dense(2)
])

model.build(input_shape=(None,4))
model.summary()
