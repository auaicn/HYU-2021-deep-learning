# Binary classification using logistic regression (cross-entropy loss)

Due date : 2021-03-26 (Friday)

# Model

## Layers

Input layer : 2-dim vector, 𝒙 = {𝑥1, 𝑥2}  
Output layer : 1-dim y ∈ {0,1}

## Loss Function

used `Cross-Entrophy` function, which is known working well with binary-classification problem

```python
for i in range(numberTrainingSamples):
	...
	...
	loss -= y_train[i] * np.log(a) + (1 - y_train[i]) * np.log(1 - a)
	...
	...

```

## Activation Function

for all layer, used `sigmoid` function

```python
def sigmoid(z: float) -> float: #! Type Annotation only available on python3
	a = 1 / (1 + np.exp(-z))
	# avoid floating point issues
	if a == 1.0:
		a = 1.0 - 0.00000001
	elif a == 0.0:
		a = 0.00000001
	return a
```

## Sequence

1.  Generate 1000(=m) train samples, 100(=n) test samples:
2.  Update `W`, `𝑏` with 1000 samples for 2000 (=K) iterations: #K updates with the gradient descent

# What I Learned

- `random` module is included, whereas `numpy` module has to be installed
  so `random` is differentf rom `numpy.random`
- Forward-Declaration needed for python's
- python function requires strict type. So Explicit Type Annotation was effective  
  even though python is somewhat free at handling big-integer,
- underflow occurs when calculating e^(-x) when `x >= 730'
- overflow occurs when calculating e^(x) when `x >= 730'
  maybe it's because different integer
- float handling... if a float value become close to 0(or 1) then, it is treated as real 0(or 1)

### To Search

- [ ] except: block's second line code `exit()` or `quit()` was not executed -> why?
