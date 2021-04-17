import random
import numpy as np
from datetime import datetime

# Treat Warning as Exception
np.seterr(all='raise')

EPSILON = 0.00000001

def sigmoid(z): #! Type Annotation only available on python3
	return 1 / (1 + np.exp(-z))

# diffentiated sigmoid
def sigmoid_(z):
	exp_z = np.exp(z)
	return exp_z / ((exp_z + 1) * (exp_z + 1))

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
	numNodesOnLayer[2] = 1

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
	W1 = generateRandomMatrix(numNodesOnLayer[1],numNodesOnLayer[0])
	W2 = generateRandomMatrix(numNodesOnLayer[2],numNodesOnLayer[1])
	B1 = generateRandomMatrix(numNodesOnLayer[1],1)
	B2 = generateRandomMatrix(numNodesOnLayer[2],1)
	print("parmeters Set\n")

	print("Gradient Descent Optimization Starts...")
	print("Deep Net Training Iteration({}) started".format(numIteration))

	for it in range(0, numIteration + 1):

		# Forward propagation
		Z1 = np.dot(W1, x_training) + B1
		A1 = sigmoid(Z1)
		Z2 = np.dot(W2, A1) + B2
		A2 = sigmoid(Z2)

		# supervise training progress
		if it % printPeriod == 0:
			probYes = y_training * np.log(EPSILON + A2) #;print(*probYes)
			probNo = (1-y_training) * np.log(EPSILON + 1 - A2) #;print(*probNo)
			Training_loss = - np.sum(probYes + probNo, axis = 1) / m

			# oneline 
			# print("After {:4} Optimization, J {}".format(it, Training_loss))
			
			# W 랑 b 출력하는건 좀..
			print("# {:4} J (Training) {}".format(it, Training_loss[0]))

			# as practice2 spec
			# print("{:4} W[1]\n{}".format(it, W1))

			if it != numIteration:
				continue

			print("Deep Net Training Iteration({}) finished".format(numIteration))

			numCorrentTrainingAnswer : float = 0
			for i in range(m):
				if A2.T[i] > 0.5 and y_training[i] == 1:
					numCorrentTrainingAnswer += 1
				elif A2.T[i] < 0.5 and y_training[i] == 0:
					numCorrentTrainingAnswer += 1

			Test_Z1 = np.dot(W1, x_test) + B1
			Test_A1 = sigmoid(Test_Z1)
			Test_Z2 = np.dot(W2, Test_A1) + B2
			Test_A2 = sigmoid(Test_Z2)
			Test_probYes = y_test * np.log(EPSILON + Test_A2) #;print(*Test_probYes)
			Test_probNo = (1-y_test) * np.log(EPSILON + 1 - Test_A2) #;print(*Test_probNo)
			Test_loss = - np.sum(Test_probYes + Test_probNo, axis = 1) / numTest

			numCorrentTestAnswer : float = 0
			for i in range(numTest):
				if Test_A2.T[i] > 0.5 and y_test[i] == 1:
					numCorrentTestAnswer += 1
				elif Test_A2.T[i] < 0.5 and y_test[i] == 0:
					numCorrentTestAnswer += 1


			print("\nFinal W")
			print(" W[1] with shape {}".format(W1.shape))
			print(W1)
			print(" W[2] with shape {}".format(W2.shape))
			print(W2)

			print("\nFinal B")
			print(" B[1] with shape {}".format(B1.shape))
			print(B1)
			print(" B[2] with shape {}".format(B2.shape))
			print(B2)
			print("")

			# trainint set
			print("\n#{:30} | Loss {:.8} ".format("For 'm {}' Training Samples".format(m), Training_loss[0]))
			print("#{:30} | Accuracy {:.4} % ".format("For 'm {}' Training Samples".format(m) ,numCorrentTrainingAnswer / m * 100))

			# test set
			print("#{:30} | Loss {:.8} ".format("For 'n {}' Test Samples".format(numTest), Test_loss[0]))
			print("#{:30} | Accuracy {:.4} % ".format("For 'n {}' Test Samples".format(numTest) ,numCorrentTestAnswer / numTest * 100))
			print("\nDone\n")

		# Back propagation
		dZ2 = A2 - y_training
		dW2 = np.dot(dZ2,A1.T) / m
		dB2 = np.sum(dZ2, axis = 1, keepdims = True) / m
		dZ1 = np.dot(W2.T, dZ2) * sigmoid_(Z1) # element-wise
		dW1 = np.dot(dZ1, x_training.T) / m
		dB1 = np.sum(dZ1, axis = 1, keepdims = True) / m

		# return
		W1 -= learningRate * dW1
		W2 -= learningRate * dW2
		B1 -= learningRate * dB1
		B2 -= learningRate * dB2

	return

main()




















