# Now load the data
'''
import numpy as np
with np.load('knn_data.npz') as data:
    print data.files
    train1 = data['train']
    train_labels1 = data['train_labels']
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the data, converters convert the letter to a number
data= np.loadtxt("letter-recognition.data", dtype= 'float32', delimiter = ',',
                    converters= {0: lambda ch: ord(ch)-ord('A')})

# split the data to two, 10000 each for train and test
train, test = np.vsplit(data,2)

# split trainData and testData to features and responses
responses, trainData = np.hsplit(train,[1])
labels, testData = np.hsplit(test,[1])

# Initiate the kNN, classify, measure accuracy.
knn = cv2.KNearest()
knn.train(trainData, responses)
ret, result, neighbours, dist = knn.find_nearest(testData, k=4)

correct = np.count_nonzero(result == labels)
accuracy = correct*100.0/10000
print accuracy

# save the data
np.savez('knn_data_alphabets.npz',train=train, train_labels=responses)