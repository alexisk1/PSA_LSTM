import pandas as pd
import numpy as np

#Read CSV with , as separator and first row as header,select range of columns
df = pd.read_csv('./HHData2016.csv',sep = ',',header=0,index_col =0 ,usecols=list(range(1,22))+[27] )

df1 = df.ix[:,-1]
dataset_y = df1.values
dataset_y = dataset_y.astype('float32')

df  = df.ix[:,0:-1]
#print(sum(dataset_y))
#print(df.columns)

#Normalize dataset
from sklearn.preprocessing import MinMaxScaler

dataset = df.values
dataset = dataset.astype('float32')
#print(dataset[0],  "YYY",dataset_y[0])


# fix random seed for reproducibility
np.random.seed(7)

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)



# split into train and test sets
train_size = int(len(dataset) * 0.75)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
train_y, test_y = dataset_y[0:train_size], dataset_y[train_size:len(dataset)]
print(len(train), len(test))
print(len(train_y), len(test_y))


#train=scaler.inverse_transform(train)
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX = []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
	return np.array(dataX)


train =create_dataset(train, look_back=5)
test =create_dataset(test, look_back=5)
#train_y =create_dataset(train_y, look_back=5)
#test_y =create_dataset(test_y, look_back=5)


# reshape input to be [samples, time steps, features]
train = np.reshape(train, (train.shape[0],1,5* train.shape[2]))
test = np.reshape(test, (test.shape[0],1,5* test.shape[2]))
train_y = train_y[0:len(train)]
test_y = test_y[0:len(test)]
print (train.shape)
print (train_y.shape)
print (test.shape)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten

model = Sequential()
model.add(LSTM(100, input_shape=(1,100) ))
model.add(Dense(1) )
model.compile(loss='mae', optimizer='adam')
#model.fit(train,train_y , nb_epoch=60, batch_size=1, verbose=2)
history = model.fit(train, train_y, nb_epoch=50, batch_size=1, verbose=2, shuffle=False)
# make predictions
trainPredict = model.predict(train,batch_size=1)
model.reset_states()
testPredict = model.predict(test,batch_size=1)

#for i in range (len(trainPredict)):
#    print ( trainPredict[i], train_Y[i])


import math
from sklearn.metrics import mean_squared_error
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(train_y, trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(test_y, testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


# make predictions
#trainPredict = model.predict(trainX, batch_size=1)
#model.reset_states()
#testPredict = model.predict(testX, batch_size=1)
plt.plot(train_y)
plt.plot(trainPredict[:,0])

plt.show()

plt.plot(test_y)
plt.plot(testPredict[:,0])
plt.show()

