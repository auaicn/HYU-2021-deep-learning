import tensorflow as tf
import numpy as np
import random
import datetime

EPOCHS = 1000

m = 10000
n = 500
d = 2


def generateRandomMatrix(dim, num):
    print("generating Matrix with shape ({}, {})".format(dim, num))
    M = np.zeros((dim, num), np.float32)
    for i in range(num):
        for d in range(dim):
            M[d][i] = random.uniform(-1, 1)
    return M


def generateSamples(dim, num, message):
    # Make Training Sample, "Test"s
    print("Making {} {} Samples....".format(num, message))
    x = generateRandomMatrix(dim, num)
    y = np.zeros(num, np.float32)

    for i in range(num):
        if np.sum(x.T[i], axis=0) > 0:
            y[i] = 1

    print("Done\n")
    return x.T, y.T


# make samples
x_train, y_train = generateSamples(d, m, "Training")
x_test, y_test = generateSamples(d, n, "Test")

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=3, activation='sigmoid', input_shape=(d,)),
    tf.keras.layers.Dense(units=1, activation='sigmoid'),
])

model.summary()

# 1. Compare loss functions
# Using SGD optimizer (hyper parameter : 0.4)

# 1-1 Binary Cross-Entrophy
print("Binary Cross-Entropy")
model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.SGD(0.4),
              metrics=['accuracy'])

start = datetime.datetime.now()
model.fit(x_train, y_train, epochs=EPOCHS)
end = datetime.datetime.now()

print("\nTrain TIme:", end-start)
print("\nTrain Set")
model.evaluate(x_train,  y_train, verbose=1)
print("\nTest Set")
model.evaluate(x_test,  y_test, verbose=1)

# 1-1 Mean Square Error
model.compile(loss=tf.keras.losses.mean_squared_error,
              optimizer=tf.keras.optimizers.SGD(0.4),
              metrics=['accuracy'])

start = datetime.datetime.now()
model.fit(x_train, y_train, epochs=EPOCHS)
end = datetime.datetime.now()

print("\nTrain TIme:", end-start)
print("\nTrain Set")
model.evaluate(x_train,  y_train, verbose=1)
print("\nTest Set")
model.evaluate(x_test,  y_test, verbose=1)

# 2. Compare optimizers

# Using the binary cross-entropy loss

# 2 - 1 SGD
model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.SGD(0.4),
              metrics=['accuracy'])

start = datetime.datetime.now()
model.fit(x_train, y_train, epochs=EPOCHS)
end = datetime.datetime.now()

print("\nTrain TIme:", end-start)
print("\nTrain Set")
model.evaluate(x_train,  y_train, verbose=1)
print("\nTest Set")
model.evaluate(x_test,  y_test, verbose=1)

# 2 - 2 RMSProp
model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.RMSprop(0.01),
              metrics=['accuracy'])

start = datetime.datetime.now()
model.fit(x_train, y_train, epochs=EPOCHS)
end = datetime.datetime.now()

print("\nTrain TIme:", end-start)
print("\nTrain Set")
model.evaluate(x_train,  y_train, verbose=1)
print("\nTest Set")
model.evaluate(x_test,  y_test, verbose=1)

# 2 - 3 Adam
model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.02),
              metrics=['accuracy'])

start = datetime.datetime.now()
model.fit(x_train, y_train, epochs=EPOCHS)
end = datetime.datetime.now()

print("\nTrain TIme:", end-start)
print("\nTrain Set")
model.evaluate(x_train,  y_train, verbose=1)
print("\nTest Set")
model.evaluate(x_test,  y_test, verbose=1)

# 3. Compare mini-batches
# • Used ‘SGD’ optimizer
# • Used the binary cross-entropy loss

# 3 - 1 Mini-batch = 4
model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.SGD(0.4),
              metrics=['accuracy'])

start = datetime.datetime.now()
model.fit(x_train, y_train, batch_size=4, epochs=EPOCHS)
end = datetime.datetime.now()

print("\nTrain TIme:", end-start)
print("\nTrain Set")
model.evaluate(x_train,  y_train, verbose=1)
print("\nTest Set")
model.evaluate(x_test,  y_test, verbose=1)

# 3 - 2 Mini-batch = 32 (default)
model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.SGD(0.4),
              metrics=['accuracy'])

start = datetime.datetime.now()
# batch size same as default
model.fit(x_train, y_train, batch_size=32, epochs=EPOCHS)
end = datetime.datetime.now()

print("\nTrain TIme:", end-start)
print("\nTrain Set")
model.evaluate(x_train,  y_train, verbose=1)
print("\nTest Set")
model.evaluate(x_test,  y_test, verbose=1)

# 3 - 3 Mini-batch = 32 128
model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.SGD(0.4),
              metrics=['accuracy'])

start = datetime.datetime.now()
# batch size same as default
model.fit(x_train, y_train, batch_size=128, epochs=EPOCHS)
end = datetime.datetime.now()

print("\nTrain TIme:", end-start)
print("\nTrain Set")
model.evaluate(x_train,  y_train, verbose=1)
print("\nTest Set")
model.evaluate(x_test,  y_test, verbose=1)
