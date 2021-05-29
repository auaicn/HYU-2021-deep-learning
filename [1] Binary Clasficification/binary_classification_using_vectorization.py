import random
import numpy as np
from datetime import datetime

# Treat Warning as Exception
np.seterr(all='raise')

def sigmoid(z): #! Type Annotation only available on python3
	a = 1 / (1 + np.exp(-z))
	lenZ = len(a)
	for i in range(lenZ):
		if a[i] >= 1.0 - 0.00000001:
			a[i] = 1.0 - 0.00000001
		elif a[i] <= 0.00000001:
			a[i] = 0.00000001
	return a

def main():
	# program settings
	sampleDimension = 2
	numberTrainingSamples = 1000
	numberTestSamples = 100
	numberIteration = 2000
	learningRate :float = 0.4

	random.seed()

	# Make Training Samples
	print("Making {} Training Samples....".format(numberTrainingSamples))
	x_training = np.zeros([numberTrainingSamples,sampleDimension])
	y_training = np.zeros(numberTrainingSamples)
	for i in range(numberTrainingSamples):
		x_sum : float = 0.0
		for d in range(sampleDimension):
			x_training[i][d] = random.uniform(-10,10)
			x_sum += x_training[i][d]
		if(x_sum > 0):
			y_training[i] = 1
	x_training = x_training.T
	print("Done\n")

	# Make Test Samples
	print("Making {} Test Samples....".format(numberTestSamples))
	x_test = np.zeros([numberTestSamples,sampleDimension])
	y_test = np.zeros(numberTestSamples)
	for i in range(numberTestSamples):
		x_sum : float = 0.0
		for d in range(sampleDimension):
			x_test[i][d] = random.uniform(-10,10)
			x_sum += x_test[i][d]
		if(x_sum > 0):
			y_test[i] = 1
	x_test = x_test.T
	print("Done\n")

	# now samples available
	# set initial random weight vector, bias constant
	print("Making Parameters....")
	print("Done\n")
	weight = np.zeros(sampleDimension,float)
	bias: float = random.uniform(-10,10)
	for d in range(sampleDimension):
		weight[d] = random.uniform(-10,10)

	print("Hyper Parameter : {}".format(learningRate))
	print("Initial Weight : {}".format(weight))
	print("Initial Bias : {}\n".format(bias))

	print("Now Repeating For {} times".format(numberIteration))
	print("Gradient Descent Optimization Starts...")

	for it in range(numberIteration):
		## for each steps

		dw = np.zeros(sampleDimension)
		db = 0
		J = 0

		# Forward Propagation
		z = np.dot(weight, x_training) + bias
		a = sigmoid(z)

		J -= np.dot(y_training.T, np.log(a)) + np.dot((-1 * y_training + 1).T, np.log(1 - a))

		# Backward Propagation
		dz = a - y_training	
		dw += np.dot(x_training,dz)
		db += np.sum(dz)

		dw /= numberTrainingSamples
		db /= numberTrainingSamples
		J /= numberTrainingSamples

		if it % 10 == 0:
			print("After {:4} Optimization,  W {} , b {:3.8f} , Loss For Sample Data: {}".format(it, weight, bias, J))

		weight -= learningRate * dw
		bias -= learningRate * db


	# After Last Trial, Final Weight, Bias
	print("After {:4} Optimization,  Final W {} , Final b {:3.8f} Final Loss For Sample Data: {}".format(numberIteration, weight, bias, "Check Below"))
	print("Gradient Descent Finished\n")

	# Testing
	print("Testing...")
	# for 1000 Training Samples
	numCorrectlyEstimatedTrainingSample : int = 0
	lossForEstimatedTrainingSample : float = .0

	y_estimated_training = np.dot(weight,x_training) + bias
	a_estimated_training = sigmoid(y_estimated_training)
	lossForEstimatedTrainingSample -= np.dot(y_training, np.log(a_estimated_training)) + np.dot((1 - y_training), np.log(1 - a_estimated_training))
	lossForEstimatedTrainingSample /= numberTrainingSamples

	## Accuracy
	for i in range(numberTrainingSamples):
		if(y_training[i] == 1):
			if(a_estimated_training[i] >= 0.5):
				numCorrectlyEstimatedTrainingSample += 1
		else:
			if(a_estimated_training[i] < 0.5):
				numCorrectlyEstimatedTrainingSample += 1
	traningSamplesEstimationAccuracy : float = (numCorrectlyEstimatedTrainingSample / numberTrainingSamples) * 100

	# for 100 Test sample
	numCorrectlyEstimatedTestSample : int = 0
	lossForEstimatedTestSample : float = .0

	y_estimated_test = np.dot(weight,x_test) + bias
	a_estimated_test = sigmoid(y_estimated_test)
	lossForEstimatedTestSample -= np.dot(y_test, np.log(a_estimated_test)) + np.dot((1 - y_test), np.log(1 - a_estimated_test))
	lossForEstimatedTestSample /= numberTestSamples

	## Accuracy
	for i in range(numberTestSamples):
		if(y_test[i] == 1):
			if(a_estimated_test[i] >= 0.5):
				numCorrectlyEstimatedTestSample += 1
		else:
			if(a_estimated_test[i] < 0.5):
				numCorrectlyEstimatedTestSample += 1
	testSamplesEstimationAccuracy : float = (numCorrectlyEstimatedTestSample / numberTestSamples) * 100
	
	print("'m' {}".format(numberTrainingSamples))
	print("'n' {}".format(numberTestSamples))

	print("#{:30} | Loss {:.8} ".format("For 'm {}' Training Samples", lossForEstimatedTrainingSample))
	print("#{:30} | Loss {:.8} ".format("For 'n {}' Test Samples", lossForEstimatedTestSample))
	print("#{:30} | Accuracy {:.4} % ".format("For 'm {}' Training Samples" ,str(traningSamplesEstimationAccuracy)))
	print("#{:30} | Accuracy {:.4} % ".format("For 'n {}' Test Samples" ,str(testSamplesEstimationAccuracy)))
	print("Done\n")

	return

np.vectorize(sigmoid)
main()




















