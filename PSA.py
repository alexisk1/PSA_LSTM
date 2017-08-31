import pandas as pd
import numpy as np

#Read CSV with , as separator and first row as header,select range of columns
df = pd.read_csv('./HHData2016b.csv',sep = ',',header=0,index_col =0 ,usecols=list(range(0,22))+[27] )
df_event = df.loc[df['event'] == 1]
df_i = df_event.ix[:,0]

df_y = df.ix[:, -1]
df  = df.ix[:,1:-1]

dataset_i = df_i.values
dataset_y = df_y.values



dataset_y = dataset_y.astype('int8')
dataset_x = df.values
dataset_x = dataset_x.astype('float32')

# fix random seed for reproducibility
np.random.seed(7)

#Normalize dataset
from sklearn.preprocessing import MinMaxScaler
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset_x = scaler.fit_transform(dataset_x)


#exit 
print ("asdasdadsasd")
#train=scaler.inverse_transform(train)
# convert an array of values into a dataset matrix
def create_dataset(dataset_x,dataset_i,dataset_y, look_back=5):
    dataX, dataY = [], []
    for i in range(len(dataset_i)):
        index = int(dataset_i[i] )
        a = dataset_x[(index-1-look_back):index-1,:]
        b = dataset_y[index-1]
        dataX.append(a)
        dataY.append(b)
        a = dataset_x[(index-1-2*look_back):index-look_back-1,:]
        b = dataset_y[index-look_back-1]

        dataX.append(a)
        dataY.append(b)
    return np.array(dataX), np.array(dataY)


print (dataset_x.shape)
print (dataset_y.shape)

dataset_X, dataset_Y =create_dataset(dataset_x,dataset_i,dataset_y, look_back=5)
print (dataset_X.shape)
print (dataset_Y)
# split into train and test sets
train_size = int(len(dataset_X) * 0.8)
test_size = len(dataset_X) - train_size
train, test = dataset_X[0:train_size,:], dataset_X[train_size:len(dataset_X),:]
train_y, test_y = dataset_Y[0:train_size], dataset_Y[train_size:len(dataset_Y)]
print(len(train), len(test))
print(len(train_y), len(test_y))


print (train.shape)
print (train_y.shape)
# reshape input to be [samples, time steps, features]
train = np.reshape(train, (train.shape[0],1,5* train.shape[2]))
test = np.reshape(test, (test.shape[0],1,5* test.shape[2]))
train_y = train_y[0:len(train)]
#test_y = test_y[len(train):len(train)+len(test)]


print (train.shape)
print (train_y.shape)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten

model = Sequential()
model.add(LSTM(15, input_shape=(1,100) ))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
#model.fit(train,train_y , nb_epoch=60, batch_size=1, verbose=2)
history = model.fit(train, train_y, nb_epoch=400, batch_size=1, verbose=2, shuffle=False)
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

print (test_y.shape, testPredict[:,0].shape)
testScore = math.sqrt(mean_squared_error(test_y, testPredict[:,0]))
#print('Test Score: %.2f RMSE' % (testScore))


# make predictions
#trainPredict = model.predict(trainX, batch_size=1)
#model.reset_states()
#testPredict = model.predict(testX, batch_size=1)
import matplotlib.pyplot as plt
plt.plot(train_y)
plt.plot(trainPredict[:,0])

plt.show()

plt.plot(test_y)
plt.plot(testPredict[:,0])
plt.show()

