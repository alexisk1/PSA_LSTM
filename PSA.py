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
#print(dataset[0],  "YYY",dataset_y[10])


# fix random seed for reproducibility
np.random.seed(7)

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)



# split into train and test sets
train_size = int(len(dataset) * 0.75)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,0], dataset[train_size:len(dataset),0]
train_y, test_y = dataset_y[0:train_size], dataset_y[train_size:len(dataset)]
print(len(train), len(test))
print(len(train_y), len(test_y))

#train=scaler.inverse_transform(train)



