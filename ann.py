from random import uniform
from math import exp

def sigmoid(z):
    return 1/(1+exp(-z))

class Neuron :
    def __init__(self,n):
        self.b = uniform(-1,1)
        self.w = []
        for i in range(n):
            self.w.append(uniform(-1,1))
    def forward(self,x):
        self.x = x
        s = self.b 
        for i in range(len(self.x)):
            s = s + self.x[i]*self.w[i]
        self.z = sigmoid(s)
        return self.z
    def backward(self,dz):
        dz = dz * self.z * (1-self.z)
        db = dz
        dx = []
        dw = []
        for i in range(len(self.x)):
            dx.append(self.w[i]*dz)
            dw.append(self.x[i]*dz)
        return dw,db,dx
    def update(self,dw,db,lr):
        for i in range(len(self.w)):
            self.w[i] = self.w[i] - lr*dw[i]
        self.b = self.b - lr*db
    
class Layer :
    def __init__(self,nn,n):
        self.neurons = []
        for i in range(nn):
            self.neurons.append(Neuron(n))
    def forward(self,x):
        outputs = []
        for neuron in self.neurons :
            outputs.append(neuron.forward(x))
        return outputs
    def backward(self,dz):
        self.gradients=[]
        dx = [0]*len(self.neurons[0].x)
        for i in range(len(self.neurons)):
            dw,db,n_dx = self.neurons[i].backward(dz[i])
            self.gradients.append((dw,db))
            for j in range(len(dx)):
                dx[j] = dx[j] + n_dx[j]
        return dx
    def update(self,lr):
        for i in range(len(self.neurons)):
            self.neurons[i].update(self.gradients[i][0],self.gradients[i][1],lr)
    
class Network :
    def __init__(self):
        self.layers = []
    def add(self,nn,n):
        self.layers.append(Layer(nn,n))
    def forward(self,x):
        self.outputs = []
        for layer in self.layers :
            x = layer.forward(x)
            self.outputs.append(x)
        return self.outputs
    def backward(self,y):
        output = self.outputs[-1]
        dz = []
        for i in range(len(output)):
            dz.append(output[i] - y)
        for i in reversed(range(len(self.layers))):
            dz = self.layers[i].backward(dz)
    def update(self,lr):
        for layer in self.layers :
            layer.update(lr)
