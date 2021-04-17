ef vectorized(alpha):
#Let's Learning
	#alpha=alpha
	print(alpha)
	x=np.array(x_train)
	y=np.array(y_train)

	weight=np.zeros((1,2))
	bias=0

	start=time.time()
	for iteration in range(k):
		j=0
		dweight=np.zeros((1,2))
		dbias=0

		z=np.dot(x,weight.T)+bias

		for i in range(m):
			if z[i]>1231:z[i]=1000

		a=sigmoid(z)

		#j=np.zeros()
		y=np.reshape(y,(m,1))
		dz=a-y
		#print(x.shape,dz.shape)
		#print((x[:,0]*dz).shape,x[:,0].shape,dz.shape)
		dweight=np.zeros((2,1))
		x0=np.reshape(x[:,0],(m,1))
		x1=np.reshape(x[:,1],(m,1))
		dweight[0]+=np.sum(x0*dz)
		dweight[1]+=np.sum(x1*dz)
		
		dbias=np.sum(dz)/m
		#j/=m
		dweight/=m
		#dbias/=m
		#print(dweight.shape,weight.shape)
		weight-=dweight.T*alpha
		bias-=alpha*dbias

		#if iteration%10==0:
		#	print(weight,bias)	

	print("weight=[%.5f, %.5f]" %((weight.T)[0],(weight.T)[1]))
	print("bias=%.5f"%bias)
	#Calculate Accuracy of Training Set
	train_correct=0.
	z=np.dot(x,weight.T)+bias
	a=sigmoid(z)
	for i in range(m):
		if round(a[i])==y[i]: train_correct+=1
	#print(train_correct)
	print("Accuracy with m train set: %.1f"%(train_correct*100/m))

	#Calculate Accuracy of Test Set
	x=np.array(x_test)
	y=np.array(y_test)
	test_correct=0.
	z=np.dot(x,weight.T)+bias
	a=sigmoid(z)
	for i in range(n):
		if round(a[i])==y[i]: test_correct+=1
	#print(test_correct)
	print("Accuracy with n test set: %.1f"%(test_correct*100/n))
	print("VectorizedTime: %.3f"%(time.time()-start))