# Binary classification using TensorFlow

Due date : 2021-05-26 (Friday)

# Model

same with task3 for previous `[2] shallow network` assignment

`Input` 2-dim vector, 𝒙 = {𝑥1, 𝑥2}  
`Hidden` 3-dim  
`Output` 1-dim y ∈ {0,1}

![Model Image](https://github.com/auaicn/HYU-2021-deep-learning/blob/main/images/%5B3%5D%20TF%20Model.png?raw=true)

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=3, activation='sigmoid', input_shape=(d,)),
    tf.keras.layers.Dense(units=1, activation='sigmoid'),
])

model.summary()
```

following is Model Summary

```script
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 3)                 9
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 4
=================================================================
Total params: 13
Trainable params: 13
Non-trainable params: 0
_________________________________________________________________
Binary Cross-Entropy
```

# Experiments & Preconditions

### [1] Compare loss functions

Used `SGD` optimizer in common

### [2] Compare optimizers

Used the `binary cross-entropy` loss in common

### [3] Compare mini-batches

Used `SGD` optimizer in common  
Used the `binary cross-entropy` loss in common

---

## Fitting & Evaluating

following is example of when using `binary_crossentropy` for loss function and `SGD` for optimizer.  
detailed experiment is described below.

```python
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

```

# What I Learned

- python “sys” module makes us to be able to use argv
- python “import” keyword not “imports” script but explicitly “runs” the script
- python “**name**” variable usage. When multiple modules
- python -m option & python sys.path
- python “venv” virtual environment. It seperates system-installed modules with in-virtualenv modules
- My Macbook Pro seems faster than google colab  
  Actually 1.60x faster  
  But it seems not uniform overtime
- significance of batch size
- libraries are established well. my previous implementation could have been made very easy through tensorflow
- there’s early stop setting however
