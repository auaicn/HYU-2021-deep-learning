# Binary classification using shallow network

Due date : 2021-04-30 (Friday)

# Model

`task_integrated.py` simply compares `task1`, `task2`, `task3` using same samples

## Layer

<details>
<summary>TASK 1</summary>
  
_(same as previous [project[1]]("https://github.com/auaicn/HYU-2021-deep-learning/[1]-Binary-Clasficification"))_  
`Input` 2-dim vector, 𝒙 = {𝑥1, 𝑥2}  
`Output` 1-dim y ∈ {0,1}
  
</details>

<details>
<summary>TASK 2</summary>

_(simple hidden-layer added)_  
`Input` 2-dim vector, 𝒙 = {𝑥1, 𝑥2}  
`Hidden` 1-dim  
`Output` 1-dim y ∈ {0,1}

</details>

<details>
<summary>TASK 3</summary>

_(single but multiple-node hidden-layer added)_  
`Input` 2-dim vector, 𝒙 = {𝑥1, 𝑥2}  
`Hidden` 3-dim  
`Output` 1-dim y ∈ {0,1}

</details>
  


## Loss Function

used `Cross-Entrophy` function, which is known working well with binary-classification problem

```python
for i in range(numberTrainingSamples):
	...
	loss -= y_train[i] * np.log(a) + (1 - y_train[i]) * np.log(1 - a)
	...

```

## Activation Function

for all layer, used `sigmoid` function  
to prevent underflow when value in `log` is 0, I set `EPSILON` to be 0.0001 and used `EPSILON` when calling sigmoid.

```python
A_Test[2] = sigmoid(Z[2])
Test_probYes = y_test * np.log(EPSILON + A_Test[2])
```

It simplified sigmoid function in contrast previous [project[1]]("https://github.com/auaicn/HYU-2021-deep-learning/[1]-Binary-Clasficification")

```python
def sigmoid(z): #! Type Annotation only available on python3
	return 1 / (1 + np.exp(-z))
```

# Fitting

with randomly-Generated 10000(=m) train samples, 500(=n) test samples:

Update `W`, `𝑏` with 10000 samples for 5000 (=K) iterations: #K updates with the gradient descent

Following is abbreviated version of each update sequence.

```python
for it in range(0, numIteration + 1):
  # Forward propagation
  Z1 = np.dot(W1, x_training) + B1
  A1 = sigmoid(Z1)
  Z2 = np.dot(W2, A1) + B2
  A2 = sigmoid(Z2)

  # supervise training progress
  if it % printPeriod == 0:
    if it != numIteration:
      continue

    print("Deep Net Training Iteration({}) finished".format(numIteration))
    # evaluate
    ...

  # Back propagation
  dZ2 = A2 - y_training
  dW2 = np.dot(dZ2,A1.T) / m
  dB2 = np.sum(dZ2, axis = 1, keepdims = True) / m
  dZ1 = np.dot(W2.T, dZ2) * sigmoid_diff(Z1) # element-wise
  dW1 = np.dot(dZ1, x_training.T) / m
  dB1 = np.sum(dZ1, axis = 1, keepdims = True) / m

  # update params
  W1 -= learningRate * dW1
  W2 -= learningRate * dW2
  B1 -= learningRate * dB1
  B2 -= learningRate * dB2
  return
```

# What I Learned

- with same `seed`, we can obtain pseudo-randomly generated samples, but I didn’t use it
- python uses len() instead not length(). Even it’s not porperty. We should call function
- using EPPSILON constant to avoid Divide-by Zero Error (log 0)
- sigmoid differential was in real very simple sigmoid(z) \* (1-sigmoid(z)).  
  I learned it from my friend! :)
