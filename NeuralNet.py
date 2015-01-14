import copy
import sys
from datetime import datetime
from math import exp
from random import random, randint, choice

class Perceptron(object):
    """
    Class to represent a single Perceptron in the net.
    """
    def __init__(self, inSize=1, weights=None):
        self.inSize = inSize+1#number of perceptrons feeding into this one; add one for bias
        if weights is None:
            #weights of previous layers into this one, random if passed in as None
            self.weights = [1.0]*self.inSize
            self.setRandomWeights()
        else:
            self.weights = weights
    
    def getWeightedSum(self, inActs):
        """
        Returns the sum of the input weighted by the weights.
        
        Inputs:
            inActs (list<float/int>): input values, same as length as inSize
        Returns:
            float
            The weighted sum
        """
        return sum([inAct*inWt for inAct,inWt in zip(inActs,self.weights)])
    
    def sigmoid(self, value):
        """
        Return the value of a sigmoid function.
        
        Args:
            value (float): the value to get sigmoid for
        Returns:
            float
            The output of the sigmoid function parametrized by 
            the value.
        """
        """YOUR CODE"""
        e = exp(-1*value)
        e += 1
        return (1/e)
      
    def sigmoidActivation(self, inActs):                                       
        """
        Returns the activation value of this Perceptron with the given input.
        Same as rounded g(z) in book.
        Remember to add 1 to the start of inActs for the bias input.
        
        Inputs:
            inActs (list<float/int>): input values, not including bias
        Returns:
            int
            The rounded value of the sigmoid of the weighted input
        """
        """YOUR CODE"""
        temp = inActs[:];
        temp.insert(0,1);

        weights = self.getWeightedSum(temp)

        return round(self.sigmoid(weights))
        
    def sigmoidDeriv(self, value):
        """
        Return the value of the derivative of a sigmoid function.
        
        Args:
            value (float): the value to get sigmoid for
        Returns:
            float
            The output of the derivative of a sigmoid function
            parametrized by the value.
        """
        """YOUR CODE"""
        e = exp(value)
        ePlus = e + 1

        return (e/(ePlus*ePlus))
        
    def sigmoidActivationDeriv(self, inActs):
        """
        Returns the derivative of the activation of this Perceptron with the
        given input. Same as g'(z) in book (note that this is not rounded.
        Remember to add 1 to the start of inActs for the bias input.
        
        Inputs:
            inActs (list<float/int>): input values, not including bias
        Returns:
            int
            The derivative of the sigmoid of the weighted input
        """
        """YOUR CODE"""
        temp = inActs[:];
        temp.insert(0,1);

        weights = self.getWeightedSum(temp)

        return (self.sigmoidDeriv(weights))
    
    def updateWeights(self, inActs, alpha, delta):
        """
        Updates the weights for this Perceptron given the input delta.
        Remember to add 1 to the start of inActs for the bias input.
        
        Inputs:
            inActs (list<float/int>): input values, not including bias
            alpha (float): The learning rate
            delta (float): If this is an output, then g'(z)*error
                           If this is a hidden unit, then the as defined-
                           g'(z)*sum over weight*delta for the next layer
        Returns:
            float
            Return the total modification of all the weights (sum of each abs(modification))
        """
        totalModification = 0
        """YOUR CODE"""
        newList = []
        temp = inActs[:]
        temp.insert(0,1)

        for i in range(0,len(temp)):
            curr = self.weights[i]
            newWt = alpha*temp[i]*delta
            next = curr + newWt
            diff = next - curr
            self.weights[i] = next

            newList.append(abs(diff))

        totalModification = sum(newList)
        return totalModification
            
    def setRandomWeights(self):
        """
        Generates random input weights that vary from -1.0 to 1.0
        """
        for i in range(self.inSize):
            self.weights[i] = (random() + .0001) * (choice([-1,1]))
        
    def __str__(self):
        """ toString """
        outStr = ''
        outStr += 'Perceptron with %d inputs\n'%self.inSize
        outStr += 'Node input weights %s\n'%str(self.weights)
        return outStr

class NeuralNet(object):                                    
    """
    Class to hold the net of perceptrons and implement functions for it.
    """          
    def __init__(self, layerSize):#default 3 layer, 1 percep per layer
        """
        Initiates the NN with the given sizes.
        
        Args:
            layerSize (list<int>): the number of perceptrons in each layer 
        """
        self.layerSize = layerSize #Holds number of inputs and percepetrons in each layer
        self.outputLayer = []
        self.numHiddenLayers = len(layerSize)-2
        self.hiddenLayers = [[] for x in range(self.numHiddenLayers)]
        self.numLayers =  self.numHiddenLayers+1
        
        #build hidden layer(s)        
        for h in range(self.numHiddenLayers):
            for p in range(layerSize[h+1]):
                percep = Perceptron(layerSize[h]) # num of perceps feeding into this one
                self.hiddenLayers[h].append(percep)
 
        #build output layer
        for i in range(layerSize[-1]):
            percep = Perceptron(layerSize[-2]) # num of perceps feeding into this one
            self.outputLayer.append(percep)
            
        #build layers list that holds all layers in order - use this structure
        # to implement back propagation
        self.layers = [self.hiddenLayers[h] for h in xrange(self.numHiddenLayers)] + [self.outputLayer]
  
    def __str__(self):
        """toString"""
        outStr = ''
        outStr +='\n'
        for hiddenIndex in range(self.numHiddenLayers):
            outStr += '\nHidden Layer #%d'%hiddenIndex
            for index in range(len(self.hiddenLayers[hiddenIndex])):
                outStr += 'Percep #%d: %s'%(index,str(self.hiddenLayers[hiddenIndex][index]))
            outStr +='\n'
        for i in range(len(self.outputLayer)):
            outStr += 'Output Percep #%d:%s'%(i,str(self.outputLayer[i]))
        return outStr
    
    def feedForward(self, inActs):
        """
        Propagate input vector forward to calculate outputs.
        
        Args:
            inActs (list<float>): the input to the NN (an example) 
        Returns:
            list<list<float/int>>
            A list of lists. The first list is the input list, and the others are
            lists of the output values of all perceptrons in each layer.
        """
        """YOUR CODE"""
        temp  = inActs[:]
        toReturn = []
        next = []
        curr = []

        toReturn.append(temp)

        for i in range(0,self.numHiddenLayers):
            for j in self.hiddenLayers[i]:
                curr.append(j.sigmoidActivation(temp[:]))
            
            toReturn.append(curr)
            temp = curr[:]

        for j in self.outputLayer:
            next.append(j.sigmoidActivation(temp[:]))
        
        toReturn.append(next)
        return toReturn
    
    def backPropLearning(self, examples, alpha):
        """
        Run a single iteration of backward propagation learning algorithm.
        See the text and slides for pseudo code.
        NOTE : the pseudo code in the book has an outputErr - 
        you should not update the weights while backpropagating; 
        follow the comments below or the description in lecture.
        
        Args: 
            examples (list<tuple<list,list>>):for each tuple first element is input(feature) "vector" (list)
                                                             second element is output "vector" (list)
            alpha (float): the alpha to training with
        Returns
           tuple<float,float>
           
           A tuple of averageError and averageWeightChange, to be used as stopping conditions. 
           averageError is the summed outputErr^2/2 of all examples, divided by numExamples*numOutputs.
           averageWeightChange is the summed weight change of all perceptrons, divided by the sum of 
               their input sizes.
        """
        #keep track of output
        averageError = 0
        
        averageWeightChange = 0
        changeWeights = []
        numWeights = 0
        divisorforEx = 0
        errorList=[]
        for example in examples:#for each example
            deltaLayer = []#keep track of deltaLayer to use in weight change
            """YOUR CODE"""
            """Get output of all layers"""
            outputfeed = self.feedForward(example[0])
            """
            Calculate output errors for each output perceptron and keep track 
            of outputErr sum. Add outputErr delta values to list.
            """
            outputErr =0         
            ndx = 0
            tempDelt=[]
            for value in outputfeed[len(outputfeed)-1]:
                errLayer = (example[1][ndx]-value)
                errLayer = (errLayer*errLayer)/2
                tempDelt.append( self.outputLayer[ndx].sigmoidActivationDeriv(outputfeed[len(outputfeed)-2]) * (example[1][ndx]-value))
                outputErr += errLayer
                ndx+=1
            deltaLayer.append(tempDelt)
            """
            Backpropagate through all hidden layers, calculating and storing
            the deltaLayer for each perceptron layer.
            """
            for k in range(len(self.hiddenLayers)-1,-1,-1):
                tempDelt=[] 
                j = 0
                for percept in self.hiddenLayers[k]:
                    inputs = percept.sigmoidActivationDeriv(outputfeed[k])
                    weightsForNode = []
                    if(k!=self.numHiddenLayers-1):
                        for node2 in self.hiddenLayers[k+1]:
                            weightsForNode.append(node2.weights[j+1])
                        
                        #take from output layer
                    else:
                        for node2 in self.outputLayer:
                            weightsForNode.append(node2.weights[j+1])
                    deltaI = sum([a*b for a,b in zip(weightsForNode,deltaLayer[0])]) * inputs
                    tempDelt.append(deltaI)
                    j+=1
                deltaLayer.insert(0,tempDelt[:])
                    
            """
            Having aggregated all deltaLayer, update the weights of the 
            hidden and output layers accordingly.
            """

            deltasFinal = [item for sublist in deltaLayer for item in sublist]
            count1 = 0
            count2 = 0
            for layer in self.hiddenLayers:
                for percept in layer:
                    changeWeights.append((percept.updateWeights(outputfeed[count2],alpha,deltasFinal[count1])))
                    count1+=1
                    divisorforEx+=(len(outputfeed[count2])+1)
                
                count2+=1
            for percept in self.outputLayer:
                changeWeights.append((percept.updateWeights(outputfeed[count2],alpha,deltasFinal[count1])))
                count1+=1
                divisorforEx+=(len(outputfeed[len(outputfeed)-2])+1)         
           
            errorList.append(outputErr)

        divisorForErr = len(examples)*len(self.outputLayer)
        errSum = sum(errorList)
        changeSum =  sum(changeWeights)
        averageWeightChange = changeSum/divisorforEx
        averageError = errSum/divisorForErr
        """Calculate final output"""
        return (averageError, averageWeightChange)
    
def buildNeuralNet(examples, alpha=0.1, weightChangeThreshold = 0.00008,hiddenLayerList = [1], maxItr = sys.maxint, startNNet = None):
    """
    Train a neural net for the given inputs.
    
    Args: 
        examples (tuple<list<tuple<list,list>>,
                        list<tuple<list,list>>>): A tuple of training and test examples
        alpha (float): the alpha to train with
        weightChangeThreshold (float):           The threshold to stop training at
        maxItr (int):                            Maximum number of iterations to run
        hiddenLayerList (list<int>):             The list of numbers of Perceptrons 
                                                 for the hidden layer(s). 
        startNNet (NeuralNet):                   A NeuralNet to train, or none if a new NeuralNet
                                                 can be trained from random weights.
    Returns
       tuple<NeuralNet,float>
       
       A tuple of the trained Neural Network and the accuracy that ndx achieved 
       once the weight modification reached the threshold, or the iteration 
       exceeds the maximum iteration.
    """
    examplesTrain,examplesTest = examples       
    numIn = len(examplesTrain[0][0])
    numOut = len(examplesTest[0][1])     
    time = datetime.now().time()
    if startNNet is not None:
        hiddenLayerList = [len(layer) for layer in startNNet.hiddenLayers]
    print "Starting training at time %s with %d inputs, %d outputs, %s hidden layers, size of training set %d, and size of test set %d"\
                                                    %(str(time),numIn,numOut,str(hiddenLayerList),len(examplesTrain),len(examplesTest))
    layerList = [numIn]+hiddenLayerList+[numOut]
    nnet = NeuralNet(layerList)                                                    
    if startNNet is not None:
        nnet =startNNet
    """
    YOUR CODE
    """
    iteration=1
    trainError=0
    weightMod=0
    weightMod = nnet.backPropLearning(examplesTrain, alpha)[1]
    
    while(weightMod>weightChangeThreshold and iteration<maxItr):
        a = nnet.backPropLearning(examplesTrain, alpha)
        weightMod = a[1]
        trainError = a[0] 
        iteration+=1
    
    """
    Iterate for as long as ndx takes to reach weight modification threshold
    """
        #if iteration%10==0:
        #    print '! on iteration %d; training error %f and weight change %f'%(iteration,trainError,weightMod)
        #else :
        #    print '.',
        
          
    time = datetime.now().time()
    print 'Finished after %d iterations at time %s with training error %f and weight change %f'%(iteration,str(time),trainError,weightMod)
                
    """
    Get the accuracy of your Neural Network on the test examples.
    """ 
    testError = 0.0
    testGood = 0.0   
    
    for (inputs,outputs) in examplesTest:
        count=0
        expectedAnswer = outputs
        list1 = nnet.feedForward(inputs)
        ourAnswer = list1[len(list1)-1]
        list2 = []

        for expected in expectedAnswer:
            fexpected = float(expected)
            list2.append(fexpected)
        expectedAnswer = list2
        for ndx in range(0,len(expectedAnswer)):
            if(expectedAnswer[ndx]==ourAnswer[ndx]):
                count+=1
        if(count!=len(expectedAnswer)):
            testError+=1.0
        else:
            testGood+=1.0
            
    testAccuracy = testGood/(testGood+testError)         
    print 'Feed Forward Test correctly classified %d, incorrectly classified %d, test percent error  %f\n'%(testGood,testError,testAccuracy)
    return(nnet,testAccuracy)
    """return something"""