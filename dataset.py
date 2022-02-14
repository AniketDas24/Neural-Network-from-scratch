import random
import numpy as np
class Dataset:
    def __init__(self,depth):
        self.depth=depth
        self.inputs=[]
        self.labels=[]

    def create_data(self):
        max=2*self.depth
        for i in range(max):
            binary=str(bin(i)).replace('0b','') 
            temp1=(self.depth-len(binary))*'0'+ binary
            temp2=[int(i) for i in temp1]
            res=0
            for i in temp2:
                res=res^i
            self.labels.append(res)
            self.inputs.append(temp2)
        index=list(zip(self.inputs,self.labels))
        random.shuffle(index)
        self.inputs,self.labels=zip(*index)
        return (np.array(self.inputs),np.array(self.labels))


if __name__ == '__main__':
    X = Dataset(5)
    input, labels = X.create_data()
    print("Input data = {}".format(input))
    print("Label =  {}".format(labels))




