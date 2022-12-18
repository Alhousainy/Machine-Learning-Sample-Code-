# My-Projects
#Neural_Networks (Sample Code )
import numpy as np 
#neurons=int(input('Please Enter The Number Of Neurons'))
neurons =4 #Intial Value For Sample Example About Neural Networks 
#============================================================
#Def Sigmoid Function 
def sigmoid(num1): return 1.0/(1.0 + np.exp(-num1))
#=======================================================
#Def Devrative Sigmoid Function 
def sigmoid_dervative(num2): return num2  * (1 - num2)
#================================================================
#Make Class Is Called Neural Networks 
class NeuralNetworks:
    def __init__(self,x,y):
        self.input=x
        print('The Matrix Of Inputs Is \n',self.input)
        print('**'*20)
        self.weights1=np.random.rand(self.input.shape[1],neurons)
        print('The Matrix Of Weights One Is \n',self.weights1)
        print('**'*20)
        self.weights2=np.random.rand(neurons,1)
        print('The Matrix Of Weights 2 Is \n',self.weights2)
        print('**'*20)
        self.y=y
        print('The Matrix Of Real Values Is \n',y)
        print('**'*20)
        self.output=np.zeros(self.y.shape)
        print('The Matrix Of Expected Values Is \n',self.output)
        print('**'*20)
    def feedforawrd(self):
        self.layer1=sigmoid(np.dot(self.input,self.weights1))
        self.output=sigmoid(np.dot(self.layer1,self.weights2))
    def backprop(self):
        d_weights2=np.dot(self.layer1.T,(2*(self.y-self.output) * sigmoid_dervative(self.output)))
        d_weights1=np.dot(self.input.T,(np.dot(2*(self.y-self.output) * sigmoid_dervative(self.output),self.weights2.T) * sigmoid_dervative(self.layer1)))
        self.weights1+=d_weights1
        self.weights2+=d_weights2
#===========================================================================================================================================================
x_input=np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
y_output=np.array([[0],[1],[1],[0]])
NN=NeuralNetworks(x_input,y_output)
for i in range (10000):
    NN.feedforawrd()
    NN.backprop()
print('**'*50)
print('The Final Output Is \n',NN.output)
#=========================================================================================================
###Output #####
The Final Output Is 
 [[0.00268566]
 [0.99030833]
 [0.99026374]
 [0.01207529]]
 
