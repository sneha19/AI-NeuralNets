from NeuralNetUtil import buildExamplesFromCarData,buildExamplesFromPenData
from NeuralNet import buildNeuralNet
import cPickle 
from math import pow, sqrt

def average(argList):
    return sum(argList)/float(len(argList))

def stDeviation(argList):
    mean = average(argList)
    diffSq = [pow((val-mean),2) for val in argList]
    return sqrt(sum(diffSq)/len(argList))

penData = buildExamplesFromPenData() 
def testPenData(hiddenLayers = [24]):
    return buildNeuralNet(penData,maxItr = 200, hiddenLayerList =  hiddenLayers)

carData = buildExamplesFromCarData()
def testCarData(hiddenLayers = [16]):
    return buildNeuralNet(carData,maxItr = 200,hiddenLayerList =  hiddenLayers)

#pen = []
#car = []

#for i in range(5):
#    a = testPenData()
#    b = testCarData()
#    pen.append(a[1])
#    car.append(b[1])
#
#print(average(pen),max(pen),stDeviation(pen))   =

#print(average(car),max(car),stDeviation(car))   
   
#Change in Perceptrons

#list1 = []
#list2 = []

#for i in range(0,45,5):
#    for j in range(5):
#        b = testCarData([i])
#        car.append(b[1])
#
#    v = (average(car),max(car),stDeviation(car))
#    print v
#
#    list1.append((aver2,max2,stddev2))
#print list1
 
#for i in range(0,45,5): 
#    for j in range(5):
#        a = testPenData([i])
#        pen.append(a[1])
#
#    u = (average(pen),max(pen),stDeviation(pen))
#    print u
#    
#    list2.append((aver1,max1,stddev1))
#print list2



#listed = ([([0,0],[0]),([0,1],[1]),([1,1],[0]),([1,0],[1])] , [([0,0],[0]),([1,1],[0]),([0,1],[1])])
#count = 0
#acc = buildNeuralNet(listed,maxItr = 300,hiddenLayerList = [count])[1]
#while(accuracy < 0.99):
#     count = count + 1
#     acc = buildNeuralNet(listed,maxItr = 300,hiddenLayerList = [count])[1]
#     print(acc)
#print(count)    


