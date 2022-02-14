import numpy as np
from dataset import Dataset

# print(np.random.rand(5,4))
np.random.seed(0)

l = int(input('Depth of the XOR network:'))
X = Dataset(l)
inputs, labels= X.create_data()
# print("Input data :{} label data:{}".format(inputs, labels))

def one_hot(x):
    ls = []
    for i in x:
        temp = np.zeros((2, ))
        temp[i] = 1
        ls.append(temp)
    return np.array(ls)

labels = one_hot(labels)
# print(labels.shape) 

h_depth = int(input("Depth of the hidden layer:"))
epochs  = int(input("No of epochs:" ))
alpha = float(input("Learning rate: "))

def sigmoid_act(x):
    return 1/(1 + np.exp(-x))


class NN:
    def __init__(self, input_depth, inputs, labels, hidden_layer_depth, epochs, alpha):
        self.l = input_depth
        self.X = inputs
        self.Y = np.reshape(labels,(len(self.X), 2))
        self.n = len(self.X)
        self.o = 2
        self.h_depth = hidden_layer_depth
        self.w_ij = np.random.randn(self.l, self.h_depth)
        self.b_ij = np.zeros((self.n, self.h_depth))
        self.w_jk = np.random.randn(self.h_depth, self.o)
        self.b_jk = np.zeros((self.n, self.o))
        self.epochs = epochs
        self.alpha = alpha
    def forward_1(self):
        self.h = np.dot(self.X, self.w_ij) + self.b_ij
        self.z_j = sigmoid_act(self.h)
    
    def forward_2(self):
        self.s = np.dot(self.z_j, self.w_jk)
        self.z_k = sigmoid_act(self.s)
    def loss(self):
        loss = np.mean(np.dot((self.Y-1).T,np.log(1 - self.z_k)) - np.dot(self.Y.T, np.log(self.z_k)))
        print('loss = {}'.format(loss))
    def backprop(self):
        grad_w_jk = np.dot(self.z_j.T, (self.z_k - self.Y))
        grad_b_jk = self.z_k - self.Y
        grad_w_ij = np.dot(self.X.T,np.dot((self.z_k-self.Y)*self.z_k*(1-self.Y),self.w_jk.T)*self.z_j*(1-self.z_j))
        grad_b_ij = np.dot((self.z_k-self.Y)*self.z_k*(1-self.Y),self.w_jk.T)*self.z_j*(1-self.z_j)
        self.w_ij -= self.alpha*grad_w_ij
        self.b_ij -= self.alpha*grad_b_ij
        self.w_jk -= self.alpha*grad_w_jk
        self.b_jk -= self.alpha*grad_b_jk
        # print('{} {} {} {}'.format(self.w_ij,self.b_ij,self.w_jk, self.b_jk))
    def accuracy(self):
        acc_mat = []
        for i in range(len(self.Y)):
            if(np.argmax(self.z_k[i]) == np.argmax(self.Y[i])):
                acc_mat.append(1)
            else:
                acc_mat.append(0)
        acc = (sum(acc_mat)/len(acc_mat))*100
        # print(acc_mat)
        print('Accuracy: {}'.format(acc))
        return acc
    def fit(self):
        for epo in range(epochs):
            print("epoch no - {}".format(epo))
            self.forward_1()
            self.forward_2()
            self.backprop()
            self.loss()
            acc = self.accuracy()
            # if(acc>= 98.9):
            #     break
        print(self.z_k)
        print(self.Y)
            
            
        
    def train(self):
        pass
        

model = NN(l, inputs, labels, h_depth, epochs, alpha)
model.fit()
# model.forward_1()
# model.forward_2()
# model.backprop()
# model.loss()
# model.accuracy()
