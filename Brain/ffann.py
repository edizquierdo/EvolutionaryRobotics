import numpy as np

# -----------------------
# Transfer functions 
# -----------------------
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def relu(x):
    return x * (x > 0)

# ----------------------------------------------
# Feedforward Artificial Neural Network (v1)
# Input layer, hidden layer, and output layer
# Implemented using for loops for pedagological reasons
# ----------------------------------------------
class ANNv1:
    
    def __init__(self,NI,NH,NO):
        self.nI = NI
        self.nH = NH
        self.nO = NO
        # Parameters of this NN
        self.wIH = np.random.random(size=(NI,NH))
        self.wHO = np.random.random(size=(NH,NO))
        self.bH = np.random.random(size=NH)
        self.bO = np.random.random(size=NO) 
        # State of NN
        self.I = np.zeros(NI)
        self.H = np.zeros(NH)
        self.O = np.zeros(NO)
        
    def calc(self, inputvector):
        self.I = inputvector
        for h in range(self.nH):
            netinput = self.bH[h]
            for i in range(self.nI):
                netinput += self.I[i] * self.wIH[i,h]
            self.H[h] = sigmoid(netinput)
        for o in range(self.nO):
            netinput = self.bO[o]            
            for h in range(self.nH):
                netinput += self.H[h] * self.wHO[h,o]
            self.O[o] = sigmoid(netinput)
        return self.O


# ----------------------------------------------
# Feedforward Artificial Neural Network (v2)
# Same architecture as before: Input layer, hidden layer, and output layer
# This time implemented more efficiently using dot products 
# ----------------------------------------------     
class ANNv2:

    def __init__(self, NIU, NHU, NOU):
        self.nI = NIU
        self.nH = NHU
        self.nO = NOU
        self.wIH = np.random.normal(0,5,size=(NIU,NHU)) #np.zeros((NIU,NHU))
        self.wHO = np.random.normal(0,5,size=(NHU,NOU)) #np.zeros((NHU,NOU))
        self.bH = np.random.normal(0,5,size=NHU) #np.zeros(NHU)
        self.bO = np.random.normal(0,5,size=NOU) #np.zeros(NOU)
        self.HiddenActivation = np.zeros(NHU)
        self.OutputActivation = np.zeros(NOU)
        self.Input = np.zeros(NIU)

    def step(self,Input):
        self.Input = np.array(Input)
        self.HiddenActivation = sigmoid(np.dot(self.Input.T,self.wIH)+self.bH)
        self.OutputActivation = sigmoid(np.dot(self.HiddenActivation,self.wHO)+self.bO)
        return self.OutputActivation

    def output(self):
        return self.OutputActivation