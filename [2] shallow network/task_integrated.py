import random
import numpy as np
import time
from datetime import datetime

# Treat Warning as Exception
np.seterr(all='raise')

EPSILON = 0.0001

# Data Setting
m = 10000 			# size of training set
n = 500		# size of test set
inputDimension = 2	# dimension of each input vector
printPeriod = 50
numIteration = 5000

# extra setting
random.seed()

def sigmoid(z): #! Type Annotation only available on python3
	return 1 / (1 + np.exp(-z))

# diffentiated sigmoid
def sigmoid_diff(z):
	return sigmoid(z) * (1-sigmoid(z))
	# exp_z = np.exp(z)
	# return exp_z / ((exp_z + 1) * (exp_z + 1))

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

# make samples
x_training, y_training = generateSamples(inputDimension, m, "Training")
x_test,y_test = generateSamples(inputDimension, n, "Test")

def task1():

	learningRate :float = 0.4
	print("Making Parameters....")
	print("Done\n")
	weight = np.zeros(inputDimension,float)
	bias: float = random.uniform(-10,10)
	for d in range(inputDimension):
		weight[d] = random.uniform(-10,10)

	print("Hyper Parameter : {}".format(learningRate))
	print("Initial Weight : {}".format(weight))
	print("Initial Bias : {}\n".format(bias))

	print("Now Repeating For {} times".format(numIteration))
	print("Gradient Descent Optimization Starts...")

	times.append(time.time())
	for it in range(numIteration):
		## for each steps

		dw = np.zeros(inputDimension)
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

		dw /= m
		db /= m
		J /= m

		if it % 10 == 0:
			print("After {:4} Optimization,  W {} , b {:3.8f} , Loss For Sample Data: {}".format(it, weight, bias, J))

		weight -= learningRate * dw
		bias -= learningRate * db

	times.append(time.time())

	# After Last Trial, Final Weight, Bias
	print("After {:4} Optimization,  Final W {} , Final b {:3.8f} Final Loss For Sample Data: {}".format(numIteration, weight, bias, "Check Below"))
	print("Gradient Descent Finished\n")

	# Testing
	print("Testing...")
	# for 1000 Training Samples
	numCorrectlyEstimatedTrainingSample : int = 0
	lossForEstimatedTrainingSample : float = .0

	y_estimated_training = np.dot(weight,x_training) + bias
	a_estimated_training = sigmoid(y_estimated_training)
	lossForEstimatedTrainingSample -= np.dot(y_training, np.log(a_estimated_training)) + np.dot((1 - y_training), np.log(1 - a_estimated_training))
	lossForEstimatedTrainingSample /= m

	## Accuracy
	for i in range(m):
		if(y_training[i] == 1):
			if(a_estimated_training[i] >= 0.5):
				numCorrectlyEstimatedTrainingSample += 1
		else:
			if(a_estimated_training[i] < 0.5):
				numCorrectlyEstimatedTrainingSample += 1
	traningSamplesEstimationAccuracy : float = (numCorrectlyEstimatedTrainingSample / m) * 100

	# for 100 Test sample
	numCorrectlyEstimatedTestSample : int = 0
	lossForEstimatedTestSample : float = .0

	times.append(time.time())
	y_estimated_test = np.dot(weight,x_test) + bias
	a_estimated_test = sigmoid(y_estimated_test)
	lossForEstimatedTestSample -= np.dot(y_test, np.log(a_estimated_test)) + np.dot((1 - y_test), np.log(1 - a_estimated_test))
	lossForEstimatedTestSample /= n

	## Accuracy
	for i in range(n):
		if(y_test[i] == 1):
			if(a_estimated_test[i] >= 0.5):
				numCorrectlyEstimatedTestSample += 1
		else:
			if(a_estimated_test[i] < 0.5):
				numCorrectlyEstimatedTestSample += 1
	testSamplesEstimationAccuracy : float = (numCorrectlyEstimatedTestSample / n) * 100
	times.append(time.time())
	
	print("'m' {}".format(m))
	print("'n' {}".format(n))

	print("#{:30} | Loss {:.8} ".format("For 'm {}' Training Samples", lossForEstimatedTrainingSample))
	print("#{:30} | Loss {:.8} ".format("For 'n {}' Test Samples", lossForEstimatedTestSample))
	print("#{:30} | Accuracy {:.4} % ".format("For 'm {}' Training Samples" ,str(traningSamplesEstimationAccuracy)))
	print("#{:30} | Accuracy {:.4} % ".format("For 'n {}' Test Samples" ,str(testSamplesEstimationAccuracy)))
	print("Done\n")

	return


def task2():
	
	# Deep Model setting
	numberLayer = 2		# input layer not counted
	numNodesOnLayer = np.zeros(numberLayer + 1,int)
	numNodesOnLayer[0] = inputDimension # input layer
	numNodesOnLayer[1] = 1
	numNodesOnLayer[2] = 1

	# hyper parameter 
	learningRate :float = 1.0

	# make parameters
	W1 = generateRandomMatrix(numNodesOnLayer[1],numNodesOnLayer[0])
	W2 = generateRandomMatrix(numNodesOnLayer[2],numNodesOnLayer[1])
	B1 = generateRandomMatrix(numNodesOnLayer[1],1)
	B2 = generateRandomMatrix(numNodesOnLayer[2],1)
	print("parmeters Set\n")

	print("Gradient Descent Optimization Starts...")
	print("Deep Net Training Iteration({}) started".format(numIteration))
	times.append(time.time())
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

			print("# {:4}".format(it))
			print('first layer')
			print(W1," <- W1")
			print(B1," <- B1")
			print('second layer')
			print(W2," <- W2")
			print(B2," <- B2")

			if it != numIteration:
				continue
			times.append(time.time())
			print("2-layered Training ({}) finished".format(numIteration))

			numCorrentTrainingAnswer : float = 0
			for i in range(m):
				if A2.T[i] > 0.5 and y_training[i] == 1:
					numCorrentTrainingAnswer += 1
				elif A2.T[i] < 0.5 and y_training[i] == 0:
					numCorrentTrainingAnswer += 1

			times.append(time.time())
			Test_Z1 = np.dot(W1, x_test) + B1
			Test_A1 = sigmoid(Test_Z1)
			Test_Z2 = np.dot(W2, Test_A1) + B2
			Test_A2 = sigmoid(Test_Z2)
			Test_probYes = y_test * np.log(EPSILON + Test_A2) #;print(*Test_probYes)
			Test_probNo = (1-y_test) * np.log(EPSILON + 1 - Test_A2) #;print(*Test_probNo)
			Test_loss = - np.sum(Test_probYes + Test_probNo, axis = 1) / n

			numCorrentTestAnswer : float = 0
			for i in range(n):
				if Test_A2.T[i] > 0.5 and y_test[i] == 1:
					numCorrentTestAnswer += 1
				elif Test_A2.T[i] < 0.5 and y_test[i] == 0:
					numCorrentTestAnswer += 1
			times.append(time.time())

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
			print("#{:30} | Loss {:.8} ".format("For 'n {}' Test Samples".format(n), Test_loss[0]))
			print("#{:30} | Accuracy {:.4} % ".format("For 'n {}' Test Samples".format(n) ,numCorrentTestAnswer / n * 100))
			print("\nDone\n")

		# Back propagation
		dZ2 = A2 - y_training
		dW2 = np.dot(dZ2,A1.T) / m
		dB2 = np.sum(dZ2, axis = 1, keepdims = True) / m
		dZ1 = np.dot(W2.T, dZ2) * sigmoid_diff(Z1) # element-wise
		dW1 = np.dot(dZ1, x_training.T) / m
		dB1 = np.sum(dZ1, axis = 1, keepdims = True) / m

		# return
		W1 -= learningRate * dW1
		W2 -= learningRate * dW2
		B1 -= learningRate * dB1
		B2 -= learningRate * dB2
	return

def task3():

	# Deep Model setting
	numberLayer = 2		# input layer not counted
	numNodesOnLayer = np.zeros(numberLayer + 1,int)
	numNodesOnLayer[0] = inputDimension # input layer
	numNodesOnLayer[1] = 3
	numNodesOnLayer[2] = 1

	# hyper parameter 
	learningRate :float = 2.0

	# make parameters
	W1 = generateRandomMatrix(numNodesOnLayer[1],numNodesOnLayer[0])
	W2 = generateRandomMatrix(numNodesOnLayer[2],numNodesOnLayer[1])
	B1 = generateRandomMatrix(numNodesOnLayer[1],1)
	B2 = generateRandomMatrix(numNodesOnLayer[2],1)
	print("parmeters Set\n")

	print("Gradient Descent Optimization Starts...")
	print("Deep Net Training Iteration({}) started".format(numIteration))
	times.append(time.time())
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
			# W 랑 b 출력하는건 좀..
			print("# {:4}".format(it))
			print('first layer')
			print(W1," <- W1")
			print(B1," <- B1")
			print('second layer')
			print(W2," <- W2")
			print(B2," <- B2")


			if it != numIteration:
				continue
			times.append(time.time())

			print("Deep Net Training Iteration({}) finished".format(numIteration))

			numCorrentTrainingAnswer : float = 0
			for i in range(m):
				if A2.T[i] > 0.5 and y_training[i] == 1:
					numCorrentTrainingAnswer += 1
				elif A2.T[i] < 0.5 and y_training[i] == 0:
					numCorrentTrainingAnswer += 1

			times.append(time.time())
			Test_Z1 = np.dot(W1, x_test) + B1
			Test_A1 = sigmoid(Test_Z1)
			Test_Z2 = np.dot(W2, Test_A1) + B2
			Test_A2 = sigmoid(Test_Z2)
			Test_probYes = y_test * np.log(EPSILON + Test_A2) #;print(*Test_probYes)
			Test_probNo = (1-y_test) * np.log(EPSILON + 1 - Test_A2) #;print(*Test_probNo)
			Test_loss = - np.sum(Test_probYes + Test_probNo, axis = 1) / n

			numCorrentTestAnswer : float = 0
			for i in range(n):
				if Test_A2.T[i] > 0.5 and y_test[i] == 1:
					numCorrentTestAnswer += 1
				elif Test_A2.T[i] < 0.5 and y_test[i] == 0:
					numCorrentTestAnswer += 1
			times.append(time.time())


			print("\nFinal W")
			print("W[1] with shape {}".format(W1.shape))
			print(W1)
			print("W[2] with shape {}".format(W2.shape))
			print(W2)
			
			print("\nFinal B")
			print("B[1] with shape {}".format(B1.shape))
			print(B1)
			print("B[2] with shape {}".format(B2.shape))
			print(B2)
			print("")

			# trainint set
			print("\n#{:30} | Loss {:.8} ".format("For 'm {}' Training Samples".format(m), Training_loss[0]))
			print("#{:30} | Accuracy {:.4} % ".format("For 'm {}' Training Samples".format(m) ,numCorrentTrainingAnswer / m * 100))

			# test set
			print("#{:30} | Loss {:.8} ".format("For 'n {}' Test Samples".format(n), Test_loss[0]))
			print("#{:30} | Accuracy {:.4} % ".format("For 'n {}' Test Samples".format(n) ,numCorrentTestAnswer / n * 100))
			print("\nDone\n")

		# Back propagation
		dZ2 = A2 - y_training
		dW2 = np.dot(dZ2,A1.T) / m
		dB2 = np.sum(dZ2, axis = 1, keepdims = True) / m
		dZ1 = np.dot(W2.T, dZ2) * sigmoid_diff(Z1) # element-wise
		dW1 = np.dot(dZ1, x_training.T) / m
		dB1 = np.sum(dZ1, axis = 1, keepdims = True) / m

		# return
		W1 -= learningRate * dW1
		W2 -= learningRate * dW2
		B1 -= learningRate * dB1
		B2 -= learningRate * dB2
	return

times = []

def main():

	task1()
	task2()
	task3()

	for i in range(3):
		print('task {}'.format(i+1))
		print('training time {:.4}'.format(times[4*i+1]-times[4*i]))
		print('test time {:.4}'.format(times[4*i+3]-times[4*i+2]))
	
	return

main()




















