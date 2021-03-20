import random
import numpy as np
from datetime import datetime

# Set Seed For Random Library
random.seed(datetime.now)

# Treat Warning as Exception
np.seterr(all='raise')

def sigmoid(z: float) -> float: #! Type Annotation only available on python3
	# print("z val is {}".format(str(z)))
	a = 1 / (1 + np.exp(-z))
	return a

numberTrainingSamlples = 1000
numberTestSamples = 100

x1_training = []
x2_training = []
y_training = []

for i in range(numberTrainingSamlples):
	x1_training.append(random.uniform(-10,10))
	x2_training.append(random.uniform(-10,10))
	if(x1_training[-1] + x2_training[-1] > 0):
		y_training.append(1)
	else:
		y_training.append(0)

x1_test = []
x2_test = []
y_test = []

min_x1_test = .0
max_x1_test = .0
min_x2_test = .0
max_x2_test = .0

for i in range(numberTestSamples):
	x1_test.append(random.uniform(-10,10))
	x2_test.append(random.uniform(-10,10))
	# print(type(x1_test[-1]))
	# print(type(min_x1_test))

	min_x1_test = np.min([x1_test[-1],min_x1_test])
	max_x1_test = np.max([x1_test[-1],max_x1_test])
	min_x2_test = np.min([x2_test[-1],min_x2_test])
	max_x2_test = np.max([x2_test[-1],max_x2_test])

	if(x1_test[-1] == 0) or (x2_test[-1] == 0):
		print("somethins wrong")
		quit()

	if(x1_test[-1] + x2_test[-1] > 0):
		y_test.append(1)
	else:
		y_test.append(0)

# now samples available

# set initial random weight vector, bias constant
estimatedWeight_1 = random.random()
estimatedWeight_2 = random.random()
estimatedBias = random.random()
learningRate :float = 10.0


print("initial weight vector set as [{},{}]".format(str(estimatedWeight_1),str(estimatedWeight_2)))
print("initial bias constant set as {}".format(str(estimatedBias)))
print("learningRate set as {}".format(str(learningRate)))
print("")

numberIteration = 2000
for it in range(numberIteration):
	## for each steps

	dw_1 = 0
	dw_2 = 0
	db = 0
	J = 0

	# for each sample, compute loss, compute gradient 
	for i in range(numberTrainingSamlples):

		# Forward Propagation
		z_i = 0
		z_i += x1_training[i] * estimatedWeight_1
		z_i += x2_training[i] * estimatedWeight_2
		z_i += estimatedBias

		a_i = sigmoid(z_i) # probability that this vector is classified as class 1

		if a_i == 1.0:
			a_i = 1.0 - 0.00000001
		elif a_i == 0.0:
			a_i == 0.00000001

		temp = 	y_training[i] * np.log(a_i) + (1 - y_training[i]) * np.log(1 - a_i)
		J += - temp # calculate loss function using current 'w' and 'b' but with log
		# BackWard Propagation
			# dJ / da
		dz_i = a_i - y_training[i] # dJ / dz

		# using dz_i, calculate gradient for weight and bias
		dw_1 += dz_i * x1_training[i]
		dw_2 += dz_i * x2_training[i] # underflow ??
		db += dz_i * 1

	# same weight for each samples results in using mean value
	dw_1 /= numberTrainingSamlples
	dw_2 /= numberTrainingSamlples
	db /= numberTrainingSamlples
	J /= numberTrainingSamlples

	estimatedWeight_1 -= learningRate * dw_1
	estimatedWeight_2 -= learningRate * dw_2
	estimatedBias -= learningRate * db

	if it % 10 == 0:
		print ("after {} gradient descent process, loss become: {}".format(it,J))


print("in result")

print("estimatedWeight_1 : {}".format(estimatedWeight_1))
print("estimatedWeight_2 : {}".format(estimatedWeight_2))
print("estimatedBias : {}".format(estimatedBias))

























