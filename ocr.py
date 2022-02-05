import numpy as np
import cv2

img = cv2.imread("digits.png")
ret,new_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
gray = cv2.cvtColor(new_img,cv2.COLOR_BGR2GRAY)

# Now we split the image to 5000 cells, each 20x20 size
cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]

# Make it into a Numpy array. It size will be (50,100,20,20)
x = np.array(cells)

# Now we prepare train_data and test_data.
train = x[:,:80].reshape(-1,400).astype(np.float32) # Size = (3500,400)
test = x[:,80:100].reshape(-1,400).astype(np.float32) # Size = (1500,400)

# Create labels for train and test data
k = np.arange(10)
train_labels = np.repeat(k,400)[:,np.newaxis]
test_labels = np.repeat(k,100)[:,np.newaxis]

# Initiate kNN, train the data, then test it with test data for k=1
knn = cv2.ml.KNearest_create()
#knn = cv2.KNearest()
knn.train(train,cv2.ml.ROW_SAMPLE,train_labels)
ret,result,neighbours,dist = knn.findNearest(test,k=3)

# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
matches = result==test_labels
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.size
print(accuracy)
'''
# test
cv2.imshow('g',x[39,79])
test = x[39,79].reshape(-1,400).astype(np.float32)
ret,result,neighbours,dist = knn.findNearest(test,k=5)
print(ret,result,neighbours,dist)
cv2.waitKey(0)
cv2.destroyAllWindows()
# save the data
np.savez('knn_data.npz',train=train, train_labels=train_labels)
'''
