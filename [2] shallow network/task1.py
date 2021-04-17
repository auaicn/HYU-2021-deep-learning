import random
import numpy as np
from datetime import datetime

# Treat Warning as Exception
np.seterr(all='raise')

EPSILON = 0.0001

def sigmoid(z): #! Type Annotation only available on python3
	return 1 / (1 + np.exp(-z))

def generateRandomMatrix(dim, num):
	print("generating Matrix with shape ({}, {})".format(dim,num))
	M = np.zeros((dim,num),np.float32)
	for i in range(num):
		for d in range(dim):
			M[d][i] = random.uniform(-1,1)
	return M

def generateSamples(dim, num, message):
	# Make Training Sample, "Test"s
	print("Making {} {} Samples....".format(num,message))
	x = generateRandomMatrix(dim,num)
	y = np.zeros(num,np.float32)

	for i in range(num):
		if np.sum(x.T[i], axis = 0) > 0:
			y[i] = 1

	print("Done\n")
	return x, y
	
def main():
	
	# Data Setting
	m = 10000 			# size of training set
	numTest = 500		# size of test set
	inputDimension = 2	# dimension of each input vector

	# Deep Model setting
	numberLayer = 2		# input layer not counted
	numNodesOnLayer = np.zeros(numberLayer + 1,int)
	numNodesOnLayer[0] = inputDimension # input layer
	numNodesOnLayer[1] = 1

	# hyper parameter 
	numIteration = 5000
	learningRate :float = 0.4

	# extra setting
	random.seed()
	printPeriod = 50

	# make samples
	x_training, y_training = generateSamples(inputDimension, m, "Training")
	x_test,y_test = generateSamples(inputDimension, numTest, "Test")

	# make parameters
	W = generateRandomMatrix(numNodesOnLayer[1],numNodesOnLayer[0])
	b = generateRandomMatrix(numNodesOnLayer[1],1)
	print("parmeters Set\n")

	print("Gradient Descent Optimization Starts...")
	print("Deep Net Training Iteration({}) started".format(numIteration))

	for it in range(0, numIteration + 1):

		# Forward propagation
		Z = np.dot(W, x_training) + b
		A = sigmoid(Z)

		# supervise training progress
		if it % printPeriod == 0:
			probYes = y_training * np.log(EPSILON + A) #;print(*probYes)
			probNo = (1-y_training) * np.log(EPSILON + 1 - A) #;print(*probNo)
			Training_loss = - np.sum(probYes + probNo, axis = 1) / m

			# oneline 
			# print("After {:4} Optimization, J {}".format(it, Training_loss))
			
			print("# {:4} J (Training) {}".format(it, Training_loss[0]))

			# as practice2 spec
			# print("{:4} W[1]\n{}".format(it, W))

			if it != numIteration:
				continue

			print("Deep Net Training Iteration({}) finished".format(numIteration))

			numCorrentTrainingAnswer : float = 0
			for i in range(m):
				if A.T[i] > 0.5 and y_training[i] == 1:
					numCorrentTrainingAnswer += 1
				elif A.T[i] < 0.5 and y_training[i] == 0:
					numCorrentTrainingAnswer += 1

			Test_Z = np.dot(W, x_test) + b
			Test_A = sigmoid(Test_Z)
			Test_probYes = y_test * np.log(EPSILON + Test_A)
			Test_probNo = (1 - y_test) * np.log(EPSILON + 1 - Test_A)
			Test_loss = - np.sum(Test_probYes + Test_probNo, axis = 1) / numTest

			numCorrentTestAnswer : float = 0
			for i in range(numTest):
				if Test_A.T[i] > 0.5 and y_test[i] == 1:
					numCorrentTestAnswer += 1
				elif Test_A.T[i] < 0.5 and y_test[i] == 0:
					numCorrentTestAnswer += 1

			print("\nFinal W")
			print(" Final W with shape {}".format(W.shape))
			print(W)
			print(" Final B with shape {}".format(b.shape))
			print(b)

			# trainint set
			print("\n#{:30} | Loss {:.8} ".format("For 'm {}' Training Samples".format(m), Training_loss[0]))
			print("#{:30} | Accuracy {:.4} % ".format("For 'm {}' Training Samples".format(m) ,numCorrentTrainingAnswer / m * 100))

			# test set
			print("#{:30} | Loss {:.8} ".format("For 'n {}' Test Samples".format(numTest), Test_loss[0]))
			print("#{:30} | Accuracy {:.4} % ".format("For 'n {}' Test Samples".format(numTest) ,numCorrentTestAnswer / numTest * 100))
			print("\nDone\n")

		# Back propagation
		dZ = A - y_training
		dW = np.dot(dZ,A.T) / m
		dB = np.sum(dZ, axis = 1, keepdims = True) / m

		# return
		W -= learningRate * dW
		b -= learningRate * dB

	return

main()




















