# -*- coding: utf-8 -*-
import csv
import numpy as np 
import math


data = []
for i in range(18):
    data.append([])
    
#Read in the data

numRow=0
#open file that written in Chinese.
rawTrain=open('train.csv','r',encoding='big5')
row=csv.reader(rawTrain,delimiter=',')

for r in row:
    #the first row has no valuable data. Just has names of each column
    if numRow!=0:
        #from column 3 to column 27, the csv file stores the data for 24 hours
        for i in range(3,27):
            if r[i]!="NR":
                data[(numRow-1)%18].append(float(r[i]))
            #convert "NR" to 0 to stand for no rainfall
            else:
                data[(numRow-1)%18].append(float(0))
    numRow+=1
rawTrain.close()

#parse data

x=[]
y=[]
#12 months a year
for i in range(12):
    #the data contains 20 days per month. So the total hour per month is 24*20=480 hour per month.
    #Every 9 hour's data will be used to predict the 10th hour's result.
    #store 10th hour's data into y. So the data per month is 480-10+1=471
    for j in range(471):
        x.append([])
        #there are 18 different features
        for t in range(18):
            #use every 9 hour's data to predict 10th hour's data
            for s in range(9):
                #each month's Jth hour start, store the data in 9 continuous hours for each kind of feature
                x[471*i+j].append(data[t][480*i+j+s])
            #parse out every 10th hour's PM2.5 
        y.append(data[9][480*i+j+9])
#convert x, y from list into arraies
x=np.array(x)
y=np.array(y)

# add square term
# x = np.concatenate((x,x**2), axis=1)

# add bias
x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)
#y.shape[0]

#hyperparams initialization
weights = np.zeros(x.shape[1])
print(weights)
learningRate = 0.1
repeat = 10

#starting traning

xT = x.transpose()
sumGraSquare = np.zeros(x.shape[1])
for i in range(repeat):
    #predict = w*x
    hypo = np.dot(x,weights)
    loss = hypo-y
    lossSquareSum = np.sum(loss**2)
    #mean squared error , L2 Loss
    MSE = lossSquareSum/x.shape[1]
    cost = math.sqrt(MSE)
    #gradiantDecent = piandao[y-(w*x+b)]**2 /piandao(w) = -2*x*[y-(w*x+b)]=>gra = x*loss
    gra = np.dot(xT,loss)
    #For adagrad
    #root mean square of the sum of previous gradiants
    sumGraSquare+=gra**2
    ada = np.sqrt(sumGraSquare)
    #weightNew=weightNow-Learningrate*gra/ada
    #simplest self-adaptive way to adjust the Learning rate. Because both ada and gra are related to gra.
    #The Larger gra is, the larger step, however, the Larger gra is, the Larger ada,
    #then it will Lead to a smaller step.
    weights = weights-learningRate*(gra/ada)
    print('Current iteration: %d | Cost: %f '%(i,cost))
    
#Store the model

#save model
np.save('model.npy',weights)
#read model
trained_weights=np.load('model.npy')

#Load testing data

testRaw=[]
nRow=0
testData=open('test.csv','r',encoding='big5')
row=csv.reader(testData,delimiter=',')

for r in row:
    if nRow%18==0:
        testRaw.append([])
        for i in range(2,11):
            testRaw[nRow//18].append(float(r[i]))
    else:
        for i in range(2,11):
            if r[i]!="NR":
                testRaw[nRow//18].append(float(r[i]))
            else:
                testRaw[nRow//18].append(float(0))
    nRow+=1
testData.close()
testData= np.array(testRaw)
# add square term
# test_x = np.concatenate((test_x,test_x**2), axis=1)

# add bias
testData = np.concatenate((np.ones((testData.shape[0],1)),testData), axis=1)
print(testData)


#generate answer and store it to csv

ans=[]
for i in range(testData.shape[0]):
    ans.append(["id_"+str(i)])
    a=np.dot(trained_weights,testData[i])
    ans[i].append(a)
outPutName="predict.csv"
outStream=open(outPutName,"w+")
s=csv.writer(outStream,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])

for i in range(len(ans)):
    s.writerow(ans[i])
outStream.close()