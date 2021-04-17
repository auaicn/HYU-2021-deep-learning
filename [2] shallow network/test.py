import numpy as np
import random 

# W1 = np.zeros((3,2),float)
# W2 = np.zeros((1,3),float)

# print(len(aa))

# for i in range(len(aa)):
# 	for j in range(len(aa[0])):
# 		aa[i][j] = i*10 + j

# print(aa)
# print(np.sum(aa, axis = 1, keepdims = True))
# for n in range(6,0,-1):
# 	print(n)

def generateSamples(dim, num):
	# Make Training Samples
	print("Making {} Training Samples....".format(num))
	x = np.zeros((dim,num),np.float32)
	y = np.zeros(num,np.float32)

	for i in range(num):
		for d in range(dim):
			x[d][i] = random.uniform(-10,10)
		
		print(np.sum(x.T[i], axis = 0))
		if np.sum(x.T[i], axis = 0) > 0:
			y[i] = 1

	print("Done\n")
	return x, y

def main():

	for i in range(0,10):
		print(i)

	random.seed(4)
	a = np.zeros((2,4),int)
	b = np.zeros((2,4),int)

	for i in range(2):
		for j in range(4):
			a[i][j] = random.randint(0,10)
			b[i][j] = random.randint(0,10)
	print(a)
	print(b)
	# print(np.dot(a,b))
	print(a * b)

	return

main()