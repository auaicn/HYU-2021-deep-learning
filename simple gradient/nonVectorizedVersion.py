import random
import numpy as np
from datetime import datetime

# Treat Warning as Exception
np.seterr(all='raise')

def sigmoid(z: float) -> float: #! Type Annotation only available on python3
	a = 1 / (1 + np.exp(-z))
	# avoid floating point issues
	if a == 1.0:
		a = 1.0 - 0.00000001
	elif a == 0.0:
		a = 0.00000001
	return a

def main():
	# program settings
	sampleDimension = 2
	numberTrainingSamples = 1000
	numberTestSamples = 100
	numberIteration = 2000
	learningRate :float = 0.01

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

	print("Done\n")

	# now samples available
	# set initial random weight vector, bias constant
	print("Making Parameters....")
	print("Done\n")
	weight = np.zeros(sampleDimension,float)
	bias: float = random.uniform(-30,30)
	for d in range(sampleDimension):
		weight[d] = random.uniform(-30,30)

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

		# for each sample, compute loss, compute gradient 
		for i in range(numberTrainingSamples):

			# Forward Propagation
			z_i :float = 0
			for d in range(sampleDimension):
				z_i += x_training[i][d] * weight[d]
			z_i += bias
			a_i = sigmoid(z_i)
			J -= y_training[i] * np.log(a_i) + (1 - y_training[i]) * np.log(1 - a_i)

			# Backward Propagation
			# non-vectorized version's speed degradation factor
			dz_i = a_i - y_training[i]
			db += dz_i * 1
			for d in range(sampleDimension):
				dw[d] += x_training[i][d] * dz_i

		# same weight for each samples results in using mean value
		dw /= numberTrainingSamples
		db /= numberTrainingSamples
		J /= numberTrainingSamples

		if it % 10 == 0:
			print("After {:4} Optimization,  W {} , b {:3.8f} , Loss For Sample Data: {}".format(it, weight, bias, J))

		for d in range(sampleDimension):
			weight[d] -= learningRate * dw[d]
		bias -= learningRate * db

	print("After {:4} Optimization,  Final W {} , Final b {:3.8f} Final Loss For Sample Data: {}".format(numberIteration, weight, bias, "Check Below"))
	print("Gradient Descent Finished\n")

	print("Testing...")
	# for 1000 Training Samples
	numCorrectlyEstimatedTrainingSample : int = 0
	lossForEstimatedTrainingSample : float = .0
	for i in range(numberTrainingSamples):
		y_estimated :float = np.dot(weight, x_training[i]) + bias
		a_estimated :float = sigmoid(y_estimated)
		lossForEstimatedTrainingSample -= y_training[i] * np.log(a_estimated) + (1 - y_training[i]) * np.log(1 - a_estimated)
		if(y_training[i] == 1):
			if(a_estimated >= 0.5):
				numCorrectlyEstimatedTrainingSample += 1
		else:
			if(a_estimated < 0.5):
				numCorrectlyEstimatedTrainingSample += 1
	lossForEstimatedTrainingSample /= numberTrainingSamples
	traningSamplesEstimationAccuracy : float = (numCorrectlyEstimatedTrainingSample / numberTrainingSamples) * 100

	# for 100 Test sample
	numCorrectlyEstimatedTestSample : int = 0
	lossForEstimatedTestSample : float = .0
	for i in range(numberTestSamples):
		y_estimated :float = np.dot(weight, x_test[i]) + bias
		a_estimated : float = sigmoid(y_estimated)
		lossForEstimatedTestSample -= y_test[i] * np.log(a_estimated) + (1 - y_test[i]) * np.log(1 - a_estimated)
		if(y_test[i] == 1):
			if(a_estimated >= 0.5):
				numCorrectlyEstimatedTestSample += 1
		else:
			if(a_estimated < 0.5):
				numCorrectlyEstimatedTestSample += 1
	lossForEstimatedTestSample /= numberTestSamples
	testSamplesEstimationAccuracy : float = (numCorrectlyEstimatedTestSample / numberTestSamples) * 100

	print("#{:30} | Loss {:.8} ".format("For 'm' Training Samples", lossForEstimatedTrainingSample))
	print("#{:30} | Loss {:.8} ".format("For 'n' Test Samples", lossForEstimatedTestSample))
	print("#{:30} | Accuracy {:.4} % ".format("For 'm' Training Samples" ,str(traningSamplesEstimationAccuracy)))
	print("#{:30} | Accuracy {:.4} % ".format("For 'n' Test Samples" ,str(testSamplesEstimationAccuracy)))
	print("Done\n")

	return

main()




















