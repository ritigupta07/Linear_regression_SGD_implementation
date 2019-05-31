import numpy as np
from sklearn.model_selection import train_test_split


class linearReg:
	
	def __init__(self, epochs=10000, learning_rate=0.0001, random_state=1):
		self.coef = []
		self.n_epoch = epochs
		self.l_rate = learning_rate
		self.r_rate = random_state
		self.r = np.random.RandomState()
		self.coef.append(self.r.random_sample())
		self.coef.append(self.r.random_sample())
		self.coef.append(self.r.random_sample())

	def predict(self,test):
		res = np.empty((len(test.tolist())),dtype = int)
		for row in list(test):
			print('val1=%d val2=%d pred=%.3f' %(row[0], row[1], self.predictRow(row.tolist())))
			np.append(res,self.predictRow(row.tolist()))
		return res

	# Make a prediction with coefficients
	def predictRow(self,row):
		yhat = self.coef[0]
		for i in range(len(row)):
			yhat = yhat + (self.coef[i + 1] * row[i])
		return yhat

	# Estimate linear regression coefficients using stochastic gradient descent
	def fit(self,train, y):
		for epoch in range(self.n_epoch):
			sum_error = 0
			index = 0
			for row in list(train):
				a = row.tolist()
				yhat = self.predictRow(row.tolist())
				error = yhat - y[index]
				index = index + 1
				sum_error += error**2
				self.coef[0] = (self.coef[0] - 2*self.l_rate * error)
				for i in range(len(row.tolist())):
					self.coef[i + 1] = self.coef[i + 1] - 2*self.l_rate * error * row[i]
			print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, self.l_rate, (sum_error/(len(train.tolist())))))
 
# Calculate coefficients
X1= np.random.randint(1,10, 100)
X2= np.random.randint(1,10,100)

Y= 4*X1 + 10 *X2
X = np.hstack((X1.T,X2.T))
X1=np.vstack(X1)
X2=np.vstack(X2)
X= np.hstack((X1,X2))
Y = np.vstack(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=11)

linReg = linearReg(10000,0.001,1)
linReg.fit(X_train, y_train)
linReg.predict(X_test)
