from tkinter import *
import numpy as np
import time
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

class MLP:
    def __init__(self,inputs,outputs,layerList):
        self.rg = np.random.default_rng(1)
        self.numInputs = inputs
        self.numOutputs = outputs
        self.trainingOk = False
        self.epochsDone = 0
        self.mse = 1000
        self.errorPerEpoch = []
        self.msePerEpoch = []
        self.epochsCount = []
        self.prevMse = 1000
        self.miu = 0.01
        self.beta = 10
        self.rg = np.random.default_rng(1)
        self.w = []
        self.n = []
        self.a = []
        self.s = []
        count = 0
        for item in layerList:
            if(count==0):
                self.w.append(self.rg.random((item,self.numInputs+1)))
            else:
                self.w.append(self.rg.random((item,layerList[count-1]+1)))
               
            
            self.n.append(self.rg.random((layerList[count],1)))
            self.a.append(self.rg.random((layerList[count],1)))
            count = count+1
        self.w.append(self.rg.random((self.numOutputs,layerList[-1]+1)))
        self.n.append(self.rg.random((self.numOutputs,1)))
        self.a.append(self.rg.random((self.numOutputs,1)))
        
        
        for i in range(len(layerList)):
            self.s.append(self.rg.random((layerList[i],1)))
        self.s.append(self.rg.random((self.numOutputs,1)))
        
    def create(self,inputs,outputs,layerList):
        self.rg = np.random.default_rng(1)
        self.numInputs = inputs
        self.numOutputs = outputs
        self.trainingOk = False
        self.epochsDone = 0
        self.mse = 1000
        self.errorPerEpoch = []
        self.msePerEpoch = []
        self.epochsCount = []
        self.prevMse = 1000
        self.miu = 0.01
        self.beta = 10
        self.rg = np.random.default_rng(1)
        self.w = []
        self.n = []
        self.a = []
        self.s = []
        count = 0
        for item in layerList:
            if(count==0):
                self.w.append(self.rg.random((item,self.numInputs+1)))
            else:
                self.w.append(self.rg.random((item,layerList[count-1]+1)))
               
            
            self.n.append(self.rg.random((layerList[count],1)))
            self.a.append(self.rg.random((layerList[count],1)))
            count = count+1
        self.w.append(self.rg.random((self.numOutputs,layerList[-1]+1)))
        self.n.append(self.rg.random((self.numOutputs,1)))
        self.a.append(self.rg.random((self.numOutputs,1)))
        
        
        for i in range(len(layerList)):
            self.s.append(self.rg.random((layerList[i],1)))
        self.s.append(self.rg.random((self.numOutputs,1)))
        
        
        
    def sigmoid(self,x):
         return (1)/(1+np.exp(-x))
        
    def tanh(self,x):
        return np.tanh(x)
    
    def sigmoid_derivative(self,x):
        g = sigmoid(x)
        return g*(1-g)
    
    def tanh_derivative(self,x):
        return (1-self.tanh(x)*self.tanh(x))
    
    def replace_weights(self,weights):
        for i in range(len(self.w)):
            self.w[i] = weights[i]
    
    def feedForward(self,inputs):
        
            
        count = 0
        for k in range(len(self.w)):
            
            
            
            if(count==0):
                inputVector = np.zeros((len(inputs),1),dtype='float')
               
                for i in range(len(inputs)):
                    inputVector[i][0] = inputs[i]
                inputVector = np.append(inputVector,[[1]],axis=0)
                
                self.n[0] = self.w[0]@inputVector
               
                
               
            
                for i in range(self.n[0].shape[0]):
                    self.a[0][i][0] = self.tanh(self.n[0][i][0])
            else:
                inputVector = np.zeros((self.w[count-1].shape[0],1),dtype='float')
                for i in range(self.w[count-1].shape[0]):
                    inputVector[i][0] = self.a[count-1][i][0]
                inputVector = np.append(inputVector,[[1]],axis=0)
                self.n[count] = self.w[count]@inputVector
                
                
                if(k!=len(self.w)-1):
                    for i in range(self.n[count].shape[0]):
                        
                        self.a[count][i][0] = self.tanh(self.n[count][i][0])
                else:
                    for i in range(self.n[count].shape[0]):
                        self.a[count][i][0] = self.sigmoid(self.n[count][i][0])
                        
                
                    
                
                
            count = count+1
        return self.a[-1]
    
    def backPropTrainStochastic(self,trainingData,labels,minError,learningRate,maxEpochs):
        global root2
        self.msePerEpoch = []
        self.minError = minError
        self.learningRate = learningRate
        self.maxEpochs = maxEpochs
        self.error = np.zeros((self.numOutputs,1),dtype='float')
        currentEpoch = 1
        mse = 10
        while(currentEpoch<=self.maxEpochs and self.mse>self.minError):
               
                accumError = 0
                self.errorPerEpoch.append(0)
                
                for i in range(len(trainingData)):
                   
                    a0 = np.zeros((self.numInputs+1,1),dtype='float')
                    for p in range(self.numInputs):
                        a0[p][0] = trainingData[i][p]
                    a0[self.numInputs][0] = 1
                    a0 = np.transpose(a0)
                    
                    
                    r = self.feedForward(trainingData[i])
                   
                   
                    for j in range(self.error.shape[0]):
                        self.error[j][0] = labels[i][j]-r[j][0]
                       
                    totalError = 0
                    for j in range(self.error.shape[0]):
                        totalError = totalError+self.error[j][0]*self.error[j][0]
                    
                  
                    totalError = totalError/(self.error.shape[0])
                   
                    self.errorPerEpoch[currentEpoch-1] = self.errorPerEpoch[currentEpoch-1] + totalError
                    accumError = accumError+totalError
                    
                    for k in range(self.error.shape[0]):
                        
                        self.s[-1][k][0] = -2*self.error[k][0]*self.sigmoid_derivative(self.n[-1][k][0])
                    
                    
                    for k in reversed(range(len(self.s)-1)):
                        fp = np.zeros((self.w[k].shape[0],self.w[k].shape[0]),dtype='float')
                        for o in range(fp.shape[0]):
                            fp[o][o] = self.tanh_derivative(self.n[k][o][0])
                        wt = np.zeros((self.w[k+1].shape[0],self.w[k+1].shape[1]-1),dtype='float')
                        for m in range(self.w[k+1].shape[0]):
                            for n in range(self.w[k+1].shape[1]-1):
                                wt[m][n] = self.w[k+1][m][n]
                                
                        wt = np.transpose(wt)
                        preresult = fp@wt
                        self.s[k] = preresult@self.s[k+1]
                       
                        
                    for k in reversed(range(len(self.a)-1)):
                        at = np.append(self.a[k],[[1]],axis=0)
                        at  = np.transpose(at)
                        preresult = self.s[k+1]@at
                        delta_w = np.multiply(preresult,-self.learningRate)
                        self.w[k+1] = np.add(self.w[k+1],delta_w)
                       
                    
                    preresult = self.s[0]@a0
                    delta_w = np.multiply(preresult,-self.learningRate)
                    self.w[0] = np.add(self.w[0],delta_w)
                    
                    
                        
                        
                   
                self.errorPerEpoch[currentEpoch-1] = self.errorPerEpoch[currentEpoch-1]/(len(trainingData))
                self.epochsCount.append(currentEpoch)
                currentEpoch = currentEpoch+1
                self.mse = accumError/len(trainingData)
                self.msePerEpoch.append(self.mse)
                
                plot_error(self.epochsCount,self.msePerEpoch,False)
                root.update()
                
                
        print('Finished traning')
        
        if(mse<=self.minError):
           
            self.trainingOk = True
        else:
            
            self.trainingOk = False
       
        self.epochsDone = currentEpoch
        res = ""
        if(currentEpoch==(numEpochs+1)):
            
            res = "Failed at traning"
        else:
            res = "Training succeded"
        

        resultString.set("Finished traning.      Number of epochs: "+str(currentEpoch-1)+"\n "+res)
        for i in range(self.w[0].shape[0]):
                        
                        wvector = []
                        wvector.append(self.w[0][i][0])
                        wvector.append(self.w[0][i][1])
                        wvector.append(self.w[0][i][2])
                        
                        drawLine(root2,my_canvas_list[i],wvector,200,100)
                        
        w1String.set(str(self.msePerEpoch[-1]))                
        plot_error(self.epochsCount,self.msePerEpoch,True)
        root.update()
        
        
    def backPropTrainBatch(self,trainingData,labels,minError,learningRate,maxEpochs):
        global root2
        self.msePerEpoch = []
        self.minError = minError
        self.learningRate = learningRate
        self.maxEpochs = maxEpochs
        self.error = np.zeros((self.numOutputs,1),dtype='float')
        currentEpoch = 1
        self.mse = 10
        while(currentEpoch<=self.maxEpochs and self.mse>self.minError):
               
                accumError = 0
                
                delta_w = []
                for item in self.w:
                    delta_w.append(np.zeros((item.shape)))
                
                for i in range(len(trainingData)):
                   
                    a0 = np.zeros((self.numInputs+1,1),dtype='float')
                    for p in range(self.numInputs):
                        a0[p][0] = trainingData[i][p]
                    a0[self.numInputs][0] = 1
                    a0 = np.transpose(a0)
                    
                    
                    r = self.feedForward(trainingData[i])
                   
                    for j in range(self.error.shape[0]):
                        self.error[j][0] = labels[i][j]-r[j][0]
                    totalError = 0
                    for j in range(self.error.shape[0]):
                        totalError = totalError+self.error[j][0]*self.error[j][0]
                    
                    totalError = totalError/(self.error.shape[0])
                    
                    accumError = accumError+totalError
                    
                    for k in range(self.error.shape[0]):
                        
                        self.s[-1][k][0] = -2*self.error[k][0]*self.sigmoid_derivative(self.n[-1][k][0])
                    
                   
                    
                    
                    for k in reversed(range(len(self.s)-1)):
                        fp = np.zeros((self.w[k].shape[0],self.w[k].shape[0]),dtype='float')
                        for o in range(fp.shape[0]):
                            fp[o][o] = self.tanh_derivative(self.n[k][o][0])
                        wt = np.zeros((self.w[k+1].shape[0],self.w[k+1].shape[1]-1),dtype='float')
                        for m in range(self.w[k+1].shape[0]):
                            for n in range(self.w[k+1].shape[1]-1):
                                wt[m][n] = self.w[k+1][m][n]
                                
                        wt = np.transpose(wt)
                        preresult = fp@wt
                        self.s[k] = preresult@self.s[k+1]
                       
                        
                    for k in reversed(range(len(self.a)-1)):
                        at = np.append(self.a[k],[[1]],axis=0)
                        at  = np.transpose(at)
                        preresult = self.s[k+1]@at
                        delta = np.multiply(preresult,-self.learningRate)
                        delta_w[k+1] = np.add(delta_w[k+1],delta)
                        
                        
                       
                    
                    preresult = self.s[0]@a0
                    delta = np.multiply(preresult,-self.learningRate)
                    delta_w[0] = np.add(delta_w[0],delta)
                    
                    
                   
          
                #per Epoch
                for a in range(len(delta_w)):
                    delta_w[a] = np.multiply(delta_w[a],(1.0/len(trainingData)))
                
                for v in range(len(delta_w)):
                    self.w[v] = np.add(self.w[v],delta_w[v])
                
                self.epochsCount.append(currentEpoch)
                currentEpoch = currentEpoch+1
                
               
                self.mse = accumError/len(trainingData)
                self.msePerEpoch.append(self.mse)
                plot_error(self.epochsCount,self.msePerEpoch,False)
                root.update()
              
                
                
        print('Finished traning')
        
        if(self.mse<=self.minError):
           
            self.trainingOk = True
        else:
            
            self.trainingOk = False
       
        self.epochsDone = currentEpoch
        res = ""
        if(currentEpoch==(numEpochs+1)):
            
            
            res = "Failed at traning"
        else:
            res = "Training succeded"
        

        resultString.set("Finished traning.      Number of epochs: "+str(currentEpoch-1)+"\n "+res)
        for i in range(self.w[0].shape[0]):
                        
                        wvector = []
                        wvector.append(self.w[0][i][0])
                        wvector.append(self.w[0][i][1])
                        wvector.append(self.w[0][i][2])
                        
                        drawLine(root2,my_canvas_list[i],wvector,200,100)
        w1String.set(str(self.msePerEpoch[-1]))               
        plot_error(self.epochsCount,self.msePerEpoch,True)
        root.update()
        
    def lmTrain(self,trainingData,labels,minError,maxEpochs,initialMiu,initialBeta):
        
        self.minError = minError
        self.maxEpochs = maxEpochs
        self.error = np.zeros((self.numOutputs,1),dtype='float')
        self.miu = initialMiu
        self.beta = initialBeta
        currentEpoch = 1
        self.mse = 10
        
        while(currentEpoch<=self.maxEpochs and self.mse>self.minError):
            
            
            #print(currentEpoch)
            accumError = 0
            errorVector = np.zeros((len(trainingData)*len(labels[0]),1))
           
            delta_w = []
            for item in self.w:
                delta_w.append(np.zeros((item.shape)))
                
            delta_w2 = []
            for item in self.w:
                delta_w2.append(np.zeros((item.shape)))
                
            temp_w = []
            for item in self.w:
                temp_w.append(np.zeros((item.shape)))
                
            #Calculate number of weights and biases of the network
            partialSum = 0
            for i in range(len(self.w)-2):
                
                partialSum = partialSum+(self.w[i].shape[0]+1)*self.w[i+1].shape[0]
                
            partialSum = partialSum + (self.numInputs+1)*self.w[0].shape[0]
            numberOfWeights = partialSum + (self.w[-2].shape[0]+1)*self.numOutputs
            
            
            
            
                
            J = np.zeros((len(trainingData)*len(labels[0]),numberOfWeights),dtype='float64')
            
            #Create the empty sensitivy list matrix array to be filled
            sensitivityList = []
            for i in range(len(self.w)):
                
                
                sensitivityList.append(np.zeros((self.w[i].shape[0],self.numOutputs),dtype='float64'))
                                       
                                      
            for i in range(len(trainingData)):
                
                
                 
                a0 = np.zeros((self.numInputs+1,1),dtype='float')
                for p in range(self.numInputs):
                    a0[p][0] = trainingData[i][p]
                a0[self.numInputs][0] = 1
                a0 = np.transpose(a0)
            
                r = self.feedForward(trainingData[i])
            
                for j in range(self.error.shape[0]):
                
                    self.error[j][0] = labels[i][j]-r[j][0]
                    errorIndex = self.error.shape[0]*i+j
                    errorVector[errorIndex][0] = self.error[j][0]
                    
                
                totalError = 0
                for j in range(self.error.shape[0]):
                    totalError = totalError+self.error[j][0]*self.error[j][0]
                    
                totalError = totalError/(self.error.shape[0])
            
                accumError = accumError + totalError
                
                
                for k1 in range(sensitivityList[-1].shape[0]):
                    for k2 in range(sensitivityList[-1].shape[1]):
                        if(k1==k2):
                            sensitivityList[-1][k1][k2] = -1*self.sigmoid_derivative(self.n[-1][k1][0])
                        else:
                            sensitivityList[-1][k1][k2] = 0
                            
                
                        
                        
                       
                
                for k in reversed(range(len(sensitivityList)-1)):
                        fp = np.zeros((self.w[k].shape[0],self.w[k].shape[0]),dtype='float')
                        for o in range(fp.shape[0]):
                            fp[o][o] = self.tanh_derivative(self.n[k][o][0])
                        wt = np.zeros((self.w[k+1].shape[0],self.w[k+1].shape[1]-1),dtype='float')
                        for m in range(self.w[k+1].shape[0]):
                            for n in range(self.w[k+1].shape[1]-1):
                                wt[m][n] = self.w[k+1][m][n]
                                
                        wt = np.transpose(wt)
                        preresult = fp@wt
                        sensitivityList[k] = preresult@sensitivityList[k+1]
                       
               
                       
                #Create the input vector including the bias input value of 1
                at = np.zeros((len(trainingData[i])+1,1))
                for y in range(len(trainingData[i])):
                    at[y][0] = trainingData[i][y]
                   
                at[len(trainingData[i])][0] = 1
               
                firstInput = True
               
                columnIndex = 0
                #Assemble the Jacobian matrix
                for k in range(self.numOutputs):
                    firstEntry = True
                    at = np.zeros((self.numInputs+1,1))
                    
                    for v in range(len(sensitivityList)):
                        if(firstEntry):
                            at = np.zeros((self.numInputs+1,1))
                            for m in range(len(trainingData[i])):
                                at[m][0] = trainingData[i][m]
                            at[len(trainingData[i])][0] = 1
                            firstEntry = False
                        else:
                            at = np.zeros((self.a[v-1].shape[0]+1,1))
                            for m in range(self.a[v-1].shape[0]):
                                at[m][0] = self.a[v-1][m]
                            at[self.a[v-1].shape[0]][0] = 1
                            
                        h = (i)*self.numOutputs+k
                       
                        
                        for row in range(sensitivityList[v].shape[0]):
                            
                            for m in range(at.shape[0]):
                                
                                
                                
                                J[h][columnIndex] = sensitivityList[v][row][k]*at[m][0]
                                
                                columnIndex = columnIndex+1
                                if(columnIndex%J.shape[1]==0):
                                    columnIndex = 0
                                
                            
                            
                           
                            
                          
                            
                                
                       
                   
                   
                        
            
            accumError = accumError/len(trainingData)
            
            
            #print(accumError)
            
            loopCounter = 0
            while(loopCounter<=4):
               
               
                
                Jt = np.transpose(J)
                   
                miuMatrix = np.zeros((J.shape[1],J.shape[1]))
                for i1 in range(miuMatrix.shape[0]):
                    miuMatrix[i1][i1] = self.miu
                    
               
                preresult = Jt@J
                preresult = np.add(preresult,miuMatrix)
                preresult = np.linalg.inv(preresult)
                preresult = preresult@Jt
                
                
                result =  preresult@errorVector
                
                #Assemble delta_w matrix
                rowIndex = 0
                for i2 in range(len(self.w)):
                    for i3 in range(self.w[i2].shape[0]):
                        for i4 in range(self.w[i2].shape[1]):
                            delta_w[i2][i3][i4] = result[rowIndex][0]
                            rowIndex = rowIndex+1
                    
                    
                    
               
                    
            
                
                        
            
                for i in range(len(delta_w)):
                    delta_w[i] = np.multiply(delta_w[i],-1.0)
            
                for i in range(len(delta_w2)):
                    delta_w2[i] = np.add(self.w[i],delta_w[i])
                    
                
                for i in range(len(temp_w)):
                    temp_w[i] = self.w[i]
                    
                for i in range(len(self.w)):
                    self.w[i] = delta_w2[i]
                    
                accumError2 = 0
                
                for i in range(len(trainingData)):
                    
                    r = self.feedForward(trainingData[i])
            
                    for j in range(self.error.shape[0]):
                
                        self.error[j][0] = labels[i][j]-r[j][0]
                        
                       
                    
                
                    totalError = 0
                    for j in range(self.error.shape[0]):
                        totalError = totalError+self.error[j][0]*self.error[j][0]
                    
                    totalError = totalError/(self.error.shape[0])
            
                    accumError2 = accumError2 + totalError
                    
                    
                accumError2 = accumError2/len(trainingData)
                    
                    
                
                    
                    
                if(accumError2>=accumError):
                    self.miu = self.miu*self.beta
                    for i in range(len(temp_w)):
                        self.w[i] = temp_w[i]
                
                else:
                    self.miu = self.miu/self.beta
                    break
                
                
                loopCounter = loopCounter+1
                    
                
            
                
                
            
            
                
            
                
            
            
            
            
                
            self.epochsCount.append(currentEpoch)   
            currentEpoch = currentEpoch+1
            self.mse = accumError
            self.msePerEpoch.append(self.mse)
            plot_error(self.epochsCount,self.msePerEpoch,False)
            root.update()
            
            
            
            
            
        print('Finished traning')
            
        if(self.mse<=self.minError):
            
            self.trainingOk = True
        else:
            
            self.trainingOk = False
            
        self.epochsDone = currentEpoch
        res = ""
        if(currentEpoch==(numEpochs+1)):
            
            
            res = "Failed at traning"
        else:
            res = "Training succeded"
        

        resultString.set("Finished traning.      Number of epochs: "+str(currentEpoch-1)+"\n "+res)
        for i in range(self.w[0].shape[0]):
                        
                        wvector = []
                        wvector.append(self.w[0][i][0])
                        wvector.append(self.w[0][i][1])
                        wvector.append(self.w[0][i][2])
                        
                        drawLine(root2,my_canvas_list[i],wvector,200,100)
        w1String.set(str(self.msePerEpoch[-1]))               
        plot_error(self.epochsCount,self.msePerEpoch,True)
        root.update()
            
            
            
            
        
            
                
                
        
        
        
    def showTrainingResults(self):
        if(self.trainingOk):
            print('training succeded')
        else:
            print('training failed')
        print('Epochs done:')
        print(self.epochsDone)
        print('Final MSE')
        print(self.mse)
        
        
                    
                    
                    
                    
                    
                
        
        





def train_perceptron():
    global p_w1
    global p_w2
    global p_bias
    global learningRate
    global numEpochs
    
    
   
    learningRate = float(learningRateEntry.get())
    numEpochs = int(numepochsEntry.get())
    done = False
    currentEpoch = 1
    while(not done and (currentEpoch<(numEpochs+1))):
        done = True
        for item in trainingData:
            net = p_w1*item[0]+p_w2*item[1]-p_bias
            pw = 0
            if(net>=0):
                pw = 1
            else:
                pw = 0
                
            error = item[2]-pw
            if(error!=0):
                
                done = False
                p_bias = p_bias - learningRate*error
                
                p_w1 = p_w1+learningRate*error*item[0]
                
                p_w2 = p_w2+learningRate*error*item[1]
                
                drawLines(1,"purple")
                
                
                
        currentEpoch = currentEpoch+1

def sigmoid(x):
    return (1)/(1+np.exp(-x))


def fill_table():
    global w1
    global w2
    global bias
    correct0 = []
    correct1 = []
    incorrect0 = []
    incorrect1 = []
    table0String.set(str(len(trainingData)))
    for item in trainingData:
        net = w1*item[0]+w2*item[1]-bias
        pw = sigmoid(net)
        if(pw>=0.5):
            if(item[2]==0):
                incorrect0.append((item[0],item[1]))
            else:
                correct1.append((item[0],item[1]))
        else:
            if(item[2]==0):
                correct0.append((item[0],item[1]))
            else:
                incorrect1.append((item[0],item[1]))
    
    table7String.set(str(len(correct0)))
    table8String.set(str(len(incorrect0)))
    table10String.set(str(len(incorrect1)))
    table11String.set(str(len(correct1)))
    table9String.set(str(len(correct0)+len(incorrect0)))
    table12String.set(str(len(incorrect1)+len(correct1)))
    table13String.set(str(len(correct0)+len(incorrect1)))
    table14String.set(str(len(incorrect0)+len(correct1)))
    table15String.set(str(len(correct0)+len(correct1)+len(incorrect0)+len(incorrect1)))
        
    
    

def draw_perceptron_line():
    m = -p_w1/p_w2
    c = p_bias/p_w2
    
    x1 = 1
    y1 = 1*m+c
    x2 = -1
    y2 = -1*m+c
    
               
            
                
    x1 = (x1*WIDTH/2)+WIDTH/2
    x2 = (x2*WIDTH/2)+WIDTH/2
    y1 = (y1*HEIGHT/-2)+HEIGHT/2
    y2 = (y2*HEIGHT/-2)+HEIGHT/2
    
                
    
                
    my_canvas.create_line(x1,y1,x2,y2,fill="purple",width=4)
    

def plot_error(x,y,printAll):
    global firstEntry
    if(len(x)%50==0):
        
        
        plt.xlim(1,len(x))
        if(firstEntry):
            plt.ylim(0,max(y))
            firstEntry = False
        ax.bar(x,y,color='b')
        graph.draw()
        plt.show()
        
    else:
        if(printAll):
            plt.xlim(1,len(x))
            plt.ylim(0,max(y))
            ax.bar(x,y,color='b')
            graph.draw()
            plt.show()
        
    
            
   

def getColor(x):
    maxIndex = 0
    maxValue = x[0]
    if(x[1]>maxValue):
        maxIndex = 1
        maxValue = x[1]
    if(x[2]>maxValue):
        maxIndex = 2
        maxValue = x[2]
    if(x[3]>maxValue):
        maxIndex = 3
        maxValue = x[3]
       
    color = ""
    if(maxIndex==0):
        #color = getBlueGradient(x[maxIndex])
        color = "#1c75fe"
    if(maxIndex==1):
        #color = getRedGradient(x[maxIndex])
        color = "#ff1d1d"
    if(maxIndex==2):
        #color = getGreenGradient(x[maxIndex])
        color = "#7eff25"
    if(maxIndex==3):
        #color = getPurpleGradient(x[maxIndex])
        color = "#9626ff"
        
    return color


def getBlueGradient(value):
    if(value>0.9 and value<=1):
        return "#1c75fe"
    if(value>0.8 and value<=0.9):
        return "#1966dc"
    if(value>0.7 and value<=0.8):
        return "#1759be"
    if(value>0.6 and value<=0.7):
        return "#124592"
    if(value>0.5 and value<=0.6):
        return "#103978"
    if(value>0.4 and value<=0.5):
        return "#0c2d60"
    if(value>0.3 and value<=0.4):
        return "#0a2349"
    if(value>0.2 and value<=0.3):
        return "#071a36"
    if(value>0.1 and value<=0.2):
        return "#051226"
    if(value>=0.0 and value<=0.1):
        return "#030914"
    
def getRedGradient(value):
    if(value>0.9 and value<=1):
        return "#ff1d1d"
    if(value>0.8 and value<=0.9):
        return "#e21a1a"
    if(value>0.7 and value<=0.8):
        return "#c81818"
    if(value>0.6 and value<=0.7):
        return "#af1616"
    if(value>0.5 and value<=0.6):
        return "#911414"
    if(value>0.4 and value<=0.5):
        return "#741212"
    if(value>0.3 and value<=0.4):
        return "#5b0e0e"
    if(value>0.2 and value<=0.3):
        return "#460b0b"
    if(value>0.1 and value<=0.2):
        return "#310808"
    if(value>=0.0 and value<=0.1):
        return "#180404"
    
def getGreenGradient(value):
    if(value>0.9 and value<=1):
        return "#7eff25"
    if(value>0.8 and value<=0.9):
        return "#73e822"
    if(value>0.7 and value<=0.8):
        return "#67ce20"
    if(value>0.6 and value<=0.7):
        return "#5ab21c"
    if(value>0.5 and value<=0.6):
        return "#4d9819"
    if(value>0.4 and value<=0.5):
        return "#407e15"
    if(value>0.3 and value<=0.4):
        return "#305f10"
    if(value>0.2 and value<=0.3):
        return "#24460c"
    if(value>0.1 and value<=0.2):
        return "#1a3409"
    if(value>=0.0 and value<=0.1):
        return "#0e1a05"
    
    
def getPurpleGradient(value):
    if(value>0.9 and value<=1):
        return "#9626ff"
    if(value>0.8 and value<=0.9):
        return "#8422e1"
    if(value>0.7 and value<=0.8):
        return "#7820ca"
    if(value>0.6 and value<=0.7):
        return "#681daf"
    if(value>0.5 and value<=0.6):
        return "#571a90"
    if(value>0.4 and value<=0.5):
        return "#461673"
    if(value>0.3 and value<=0.4):
        return "#38125d"
    if(value>0.2 and value<=0.3):
        return "#260c3e"
    if(value>0.1 and value<=0.2):
        return "#13061e"
    if(value>=0.0 and value<=0.1):
        return "#09030e"

    
        
    


def drawLines(v,color):
                my_canvas.delete("all")
                if(v==0):
                
                    m = -w1/w2
                    c = bias/w2
                elif(v==1):
                    m = -p_w1/p_w2
                    c = p_bias/p_w2
    
                x1 = 1
                y1 = 1*m+c
                x2 = -1
                y2 = -1*m+c
    
               
            
                
                x1 = (x1*WIDTH/2)+WIDTH/2
                x2 = (x2*WIDTH/2)+WIDTH/2
                y1 = (y1*HEIGHT/-2)+HEIGHT/2
                y2 = (y2*HEIGHT/-2)+HEIGHT/2
    
                
    
                
                my_canvas.create_line(x1,y1,x2,y2,fill=color,width=2)
               
                drawTrainingData()
                #root.after(100)
                my_canvas.update()
              


def randomizeWeights():
    global mlp
    global w1
    global w2
    global bias
    global numNeuronsLayerOne
    global numNeuronsLayerTwo
    global numInputs
    global numOutputs
    global weights
    global weightsRandomized
    global my_canvas_list
    global rg
    global my_labels_list
    global root2
    
    if(len(my_canvas_list)!=0):
        for i in range(len(my_canvas_list)):
            my_canvas_list[i].destroy()
            my_labels_list[i].destroy()
            
    
    
    weights = []
    numNeuronsLayerOne = int(neurons1Entry.get())
    numNeuronsLayerTwo = int(neurons2Entry.get())
    
    layerList = []
    if(numNeuronsLayerOne!=0):
        
        layerList.append(numNeuronsLayerOne)
    
    if(numNeuronsLayerTwo!=0):
        layerList.append(numNeuronsLayerTwo)
        
    
    count = 0
    for item in layerList:
            if(count==0):
                weights.append(rg.random((item,numInputs+1)))
            else:
                weights.append(rg.random((item,layerList[count-1]+1)))
                
            count = count+1
            
    weights.append(rg.random((numOutputs,layerList[-1]+1)))
    
    weightsRandomized = True
    
    my_canvas_list = []
    my_labels_list = []
    currentRow = 0
    currentCol = 0
    
    for i in range(numNeuronsLayerOne):
        if(i%2==0):
            currentCol = 0
        else:
            currentCol = 1
        phrase = "Neuron "+str(i+1)
        label = Label(root2,text=phrase)
        label.grid(row=currentRow,column=currentCol,padx=10,pady=10)
        my_labels_list.append(label)
        if(currentCol==1):
            currentRow = currentRow+2
    
    currentRow = 1
    currentCol = 0
    for i in range(numNeuronsLayerOne):
        if(i%2==0):
            currentCol = 0
        else:
            currentCol = 1
        
        
        canvas = Canvas(root2,width=200,height=100,bg="white")
        canvas.grid(row = currentRow,column=currentCol,padx=10,pady=10)
        if(currentCol==1):
            currentRow = currentRow+2
        
        my_canvas_list.append(canvas)
        
    
    
    for i in range(weights[0].shape[0]):
        wvector = []
        wvector.append(weights[0][i][0])
        wvector.append(weights[0][i][1])
        wvector.append(weights[0][i][2])
        
        drawLine(root2,my_canvas_list[i],wvector,200,100)
        
    
        
    
    
    
   
    
    
   
    drawTrainingData()
    
def drawLine(root,canvas,weights,width,height):
    canvas.delete("all")
   
    
    m = -weights[0]/weights[1]
    c = weights[2]/weights[1]
    
   
    x1 = 1
    y1 = 1*m+c
    x2 = -1
    y2 = -1*m+c
  
    x1 = (x1*width/2)+width/2
    x2 = (x2*width/2)+width/2
    y1 = (y1*height/-2)+width/2
    y2 = (y2*height/-2)+width/2
    
    line1 = canvas.create_line(x1,y1,x2,y2,fill="black",width=2)
   
    canvas.update()
   
    
   
  
def resetData():
    trainingData.clear()
    randomizeWeights()
    my_canvas.delete("all")
    resultString.set("")
    errorList.clear()
    currentEpoch=1
    table0String.set("")
    table7String.set("")
    table8String.set("")
    table9String.set("")
    table10String.set("")
    table11String.set("")
    table12String.set("")
    table13String.set("")
    table14String.set("")
    table15String.set("")
    
    
    
def train_stochastic():
    global mlp
    global w1
    global w2
    global bias
    global learningRate
    global numEpochs
    global minError
    global currentEpoch
    global numInputs
    global numOutputs
    global numNeuronsLayerOne
    global numNeuronsLayerTwo
    global weights
    global weightsRandomized
    
    
    
    
    
    
   
    learningRate = float(learningRateEntry.get())
    numEpochs = int(numepochsEntry.get())
    minError = float(minErrorEntry.get())
    numNeuronsLayerOne = int(neurons1Entry.get())
    numNeuronsLayerTwo = int(neurons2Entry.get())
    layerList = []
    if(numNeuronsLayerOne!=0):
        layerList.append(numNeuronsLayerOne)
    if(numNeuronsLayerTwo!=0):
        layerList.append(numNeuronsLayerTwo)
        
    
    mlp.create(numInputs,numOutputs,layerList)
    if(weightsRandomized==True):
        mlp.replace_weights(weights)
    mlp.backPropTrainStochastic(trainingData,labels,minError,learningRate,numEpochs)
    mlp.showTrainingResults()
    
    #fill_table()
   
        
    
    
    
def train_batch():
    global mlp
    global w1
    global w2
    global bias
    global learningRate
    global numEpochs
    global minError
    global currentEpoch
    global numInputs
    global numOutputs
    global numNeuronsLayerOne
    global numNeuronsLayerTwo
    global weights
    global weightsRandomized
    
    
    
    
    
    
   
    learningRate = float(learningRateEntry.get())
    numEpochs = int(numepochsEntry.get())
    minError = float(minErrorEntry.get())
    numNeuronsLayerOne = int(neurons1Entry.get())
    numNeuronsLayerTwo = int(neurons2Entry.get())
    layerList = []
    if(numNeuronsLayerOne!=0):
        layerList.append(numNeuronsLayerOne)
    if(numNeuronsLayerTwo!=0):
        layerList.append(numNeuronsLayerTwo)
        
    
    mlp.create(numInputs,numOutputs,layerList)
    if(weightsRandomized==True):
        mlp.replace_weights(weights)
    mlp.backPropTrainBatch(trainingData,labels,minError,learningRate,numEpochs)
    mlp.showTrainingResults()
    
    #fill_table()
    
    
    
def train_lm():
    global mlp
    global w1
    global w2
    global bias
    global learningRate
    global numEpochs
    global minError
    global currentEpoch
    global numInputs
    global numOutputs
    global numNeuronsLayerOne
    global numNeuronsLayerTwo
    global weights
    global weightsRandomized
    
    
    
    
    
    
   
    learningRate = float(learningRateEntry.get())
    numEpochs = int(numepochsEntry.get())
    minError = float(minErrorEntry.get())
    numNeuronsLayerOne = int(neurons1Entry.get())
    numNeuronsLayerTwo = int(neurons2Entry.get())
    layerList = []
    if(numNeuronsLayerOne!=0):
        layerList.append(numNeuronsLayerOne)
    if(numNeuronsLayerTwo!=0):
        layerList.append(numNeuronsLayerTwo)
        
    
    mlp.create(numInputs,numOutputs,layerList)
    if(weightsRandomized==True):
        mlp.replace_weights(weights)
    mlp.lmTrain(trainingData,labels,minError,numEpochs,.01,5)
    mlp.showTrainingResults()
    
def evaluateData():
    global w1
    global w2 
    global bias
    
    for i in range(0,1000,4):
        for j in range(0,600,4):
            x = (i-WIDTH/2)*(2/WIDTH)
            y = (j-HEIGHT/2)*(-2/HEIGHT)
            maxX = 1
            minX = -1
            maxY = 1
            minY = -1
            xn = (x-minX)/(maxX-minX)
            yn = (y-minY)/(maxY-minY)
            l = []
            l.append(xn)
            l.append(yn)
            result = mlp.feedForward(l)
            resultList = []
            resultList.append(result[0][0])
            resultList.append(result[1][0])
            resultList.append(result[2][0])
            resultList.append(result[3][0])
            shapeColor = getColor(resultList)
            my_canvas.create_rectangle(i,j,i+2,j+2,outline=shapeColor,fill=shapeColor)
            
    
            
    drawTrainingData()
    
    
    
def leftClick(event):
    x = (event.x-WIDTH/2.0)*(2/WIDTH)
    y = (event.y-HEIGHT/2.0)*(-2/HEIGHT)
    maxX = 1
    minX = -1
    maxY = 1
    minY = -1
    xn = (x-minX)/(maxX-minX)
    yn = (y-minY)/(maxY-minY)
    trainingData.append([xn,yn])
    labels.append([1,0,0,0])
    my_canvas.create_oval(event.x,event.y,event.x+8,event.y+8,outline="blue",fill="blue")
    
def rightClick(event):
    x = (event.x-WIDTH/2.0)*(2/WIDTH)
    y = (event.y-HEIGHT/2.0)*(-2/HEIGHT)
    maxX = 1
    minX = -1
    maxY = 1
    minY = -1
    xn = (x-minX)/(maxX-minX)
    yn = (y-minY)/(maxY-minY)
    trainingData.append([xn,yn])
    labels.append([0,1,0,0])
    my_canvas.create_rectangle(event.x,event.y,event.x+8,event.y+8,outline="red",fill="red")
    
def shiftLeftClick(event):
    x = (event.x-WIDTH/2.0)*(2/WIDTH)
    y = (event.y-HEIGHT/2.0)*(-2/HEIGHT)
    maxX = 1
    minX = -1
    maxY = 1
    minY = -1
    xn = (x-minX)/(maxX-minX)
    yn = (y-minY)/(maxY-minY)
    trainingData.append([xn,yn])
    labels.append([0,0,1,0])
    my_canvas.create_oval(event.x,event.y,event.x+8,event.y+8,outline="green",fill="green")
    
def shiftRightClick(event):
    x = (event.x-WIDTH/2.0)*(2/WIDTH)
    y = (event.y-HEIGHT/2.0)*(-2/HEIGHT)
    maxX = 1
    minX = -1
    maxY = 1
    minY = -1
    xn = (x-minX)/(maxX-minX)
    yn = (y-minY)/(maxY-minY)
    trainingData.append([xn,yn])
    labels.append([0,0,0,1])
    my_canvas.create_rectangle(event.x,event.y,event.x+8,event.y+8,outline="purple",fill="purple")
    
    
    
def drawTrainingData():
    for i in range(len(trainingData)):
        
        x = trainingData[i][0]
        y = trainingData[i][1]
        
        xMax = 1
        xMin = -1
        yMax = 1
        yMin = -1
        
        xn = x*(xMax-xMin)+xMin
        yn = y*(yMax-yMin)+yMin
        
        x = (xn*WIDTH/2)+WIDTH/2
        y = (yn*HEIGHT/-2)+HEIGHT/2
        
        if(labels[i][0]==1):
            
            
            
            my_canvas.create_oval(x,y,x+8,y+8,outline="blue",fill="blue")
            
        elif(labels[i][1]==1):
            
            my_canvas.create_rectangle(x,y,x+8,y+8,outline="red",fill="red")
        elif(labels[i][2]==1):
          
            my_canvas.create_rectangle(x,y,x+8,y+8,outline="green",fill="green")
        elif(labels[i][3]==1):
          
            my_canvas.create_rectangle(x,y,x+8,y+8,outline="purple",fill="purple")
            
            
            
   


trainingData = []
labels = []

weights = []
w1 = 0.0
w2 = 0.0
bias = 0.0
p_w1 = 0.0
p_w2 = 0.0
p_bias = 0.0
learningRate = 0.1
numEpochs = 1000
minError = 0.001
errorList = []
currentEpoch = 1
firstEntry = True
numNeuronsLayerOne = 0
numNeuronsLayerTwo = 0
numInputs = 2
numOutputs = 4
weightsRandomized = False


mlp = MLP(2,4,[8])



WIDTH = 500
HEIGHT = 300

fig,ax = plt.subplots(dpi=90)
plt.title('MSE graph')
plt.xlim(1,numEpochs)
plt.ylim(0,1)
ax.set_xlabel("Epochs")
ax.set_ylabel("Mean Square Error")

root = Tk()
root2 = Tk()

my_canvas_list = []
my_labels_list = []



w1String = StringVar()
w2String = StringVar()
biasString = StringVar()
resultString = StringVar()
table0String = StringVar()
table7String = StringVar()
table8String = StringVar()
table9String = StringVar()
table10String = StringVar()
table11String = StringVar()
table12String = StringVar()
table13String = StringVar()
table14String = StringVar()
table15String = StringVar()



w1String.set("")
w2String.set("0.0")
biasString.set("0.0")
resultString.set("")
table0String.set("")
table7String.set("")
table8String.set("")
table9String.set("")
table10String.set("")
table11String.set("")
table12String.set("")
table13String.set("")
table14String.set("")
table15String.set("")


root.title("MLP")
root2.title("Weights")
root.geometry("1920x1080")
rg = np.random.default_rng(1)
my_canvas = Canvas(root,width=WIDTH,height=HEIGHT,bg="white")
my_canvas.bind("<Button-1>",leftClick)
my_canvas.bind("<Button-3>",rightClick)
my_canvas.bind("<Shift Button-1>",shiftLeftClick)
my_canvas.bind("<Shift Button-3>",shiftRightClick)
my_canvas.grid(row=0,column=0,rowspan=5,columnspan=4,pady=20,padx=20)

frame = Frame(root,bg='gray22',bd=3)
frame.grid(row=0,column=4,columnspan=5)

graph = FigureCanvasTkAgg(fig,master= frame)
graph.get_tk_widget().grid(row=0,column=4,columnspan=5,padx=5)


button0 = Button(root2,text="Randomize weights",pady=20,padx=10,command=randomizeWeights)
button0.grid(row=0,column=2)

pbutton = Button(root,text="Batch Train",pady=20,padx=10,command=train_batch)
pbutton.grid(row=5,column=3)

button1 = Button(root,text="Stochastic Train",pady=20,padx=10,command=train_stochastic)
button1.grid(row=5,column=1)

button4 = Button(root,text="LM Train",pady=20,padx=10,command=train_lm)
button4.grid(row=6,column=1)

button2 = Button(root,text="Evaluate data",pady=20,padx=10,command=evaluateData)
button2.grid(row=5,column=2)

learningRateEntry = Entry(root)
learningRateEntry.insert(END,'.01')
learningRateEntry.grid(row=5,column=5)

learningRateLabel = Label(root,text="Learning Rate")
learningRateLabel.grid(row=5,column=4)

numepochsEntry = Entry(root)
numepochsEntry.insert(END,'500')
numepochsEntry.grid(row=6,column=5)

numepochsLabel = Label(root,text="Number of epochs")
numepochsLabel.grid(row=6,column=4)

minErrorEntry = Entry(root)
minErrorEntry.insert(END,".01")
minErrorEntry.grid(row=7,column=5)

minErrorLabel = Label(root,text="Minimum error")
minErrorLabel.grid(row=7,column=4)

neurons1Label = Label(root,text="Neurons - First Layer")
neurons1Label.grid(row=8,column=4)

neurons1Entry = Entry(root)
neurons1Entry.insert(END,"9")
neurons1Entry.grid(row=8,column=5)

neurons2Label = Label(root,text="Neurons - Second Layer")
neurons2Label.grid(row=9,column=4)

neurons2Entry = Entry(root)
neurons2Entry.insert(END,"6")
neurons2Entry.grid(row=9,column=5)



spaceLabel = Label(root,text='         ')
spaceLabel.grid(row=5,column=6)

w1Label = Label(root,text='Final error (MSE)')
w1Label.grid(row=5,column=7)

w1ValueLabel = Label(root,textvariable=w1String)
w1ValueLabel.grid(row=5,column=8)

w2Label = Label(root,text='w2')
#w2Label.grid(row=6,column=7)

w2ValueLabel = Label(root,textvariable=w2String)
#w2ValueLabel.grid(row=6,column=8)

biasLabel = Label(root,text='bias')
#biasLabel.grid(row=7,column=7)

biasValueLabel = Label(root,textvariable=biasString)
#biasValueLabel.grid(row=7,column=8)

resetButton = Button(root,text="Reset training data",command=resetData)
resetButton.grid(row=7,column=0)

ResultLabel = Label(root,textvariable=resultString,font=("Arial",15))
ResultLabel.grid(row=7,column=2)

table0Label = Label(root,textvariable=table0String)
table0Label.grid(row=1,column=4)







root.mainloop()
root2.mainloop()