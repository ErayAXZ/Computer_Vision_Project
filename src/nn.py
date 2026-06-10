class Node :
    def __init__(self,inputs) :
        self.inputs = []
        self.weights = []
        self.bias = 0.1
        self.output = 0
        
    def forward_pass(self,inputs) :
        for i in range(len(self.inputs)) :
            self.output += self.inputs[i] * self.weights[i] 
        self.output += self.bias
    
    def link_input(self,io_links) :
        for i in io_links:
            self.inputs.append(i)
        self.weights = [0.1 for connection in range(len(self.inputs))]

    def activate(self):
        if self.output < 0:
            self.output = 0
class Network:
    def __init__(self,layer_sizes):
        self.network = []
        for layer_num in range(len(layer_sizes)):
            self.network.append([])
            for nodes in range(layer_num):
                new_node = Node()
                self.network[layer_num].append(new_node)
    
    def forward(self,x):
        for layer in self.network:
            for node in layer:
                node.forward_pass()
            
    def predict(self,x):
        outputs = self.forward_pass(x)
        return outputs.index.max(outputs)
        

class Sample:
        
    def reshape(self):
        for row in range(27) :
            self.image.append([])
            for index in range(27) : 
                self.image[row].append(self.int_flat[row*28 + index]) 
            
    def normalise(self):
        for i in self.flat:
            if isinstance(i,str) :
                self.int_flat.append(int(i))
            elif isinstance(i,int):
                self.int_flat.append(i)
            else : 
                raise Exception("unprocessed pixel")
        max_val = max(self.int_flat)
        min_val = min(self.int_flat)
        self.int_flat = [(val - min_val) / (max_val - min_val) for val in self.int_flat ]

    def __init__(self,label,image):
        self.int_flat = []
        self.image = []
        self.label = label
        self.flat = image
        self.prediction = -1
        self.one_hot_label =    [[1,0,0,0,0,0,0,0,0,0],
                                [0,1,0,0,0,0,0,0,0,0],
                                [0,0,1,0,0,0,0,0,0,0],
                                [0,0,0,1,0,0,0,0,0,0],
                                [0,0,0,0,1,0,0,0,0,0],
                                [0,0,0,0,0,1,0,0,0,0],
                                [0,0,0,0,0,0,1,0,0,0],
                                [0,0,0,0,0,0,0,1,0,0],
                                [0,0,0,0,0,0,0,0,1,0],
                                [0,0,0,0,0,0,0,0,0,1]]
        self.normalise()
        self.reshape()
        
    
    def set_prediction(self,prediciton):
        self.prediction = prediciton
    

def image_sample(file,samples):
    plt.figure(figsize=(10, 4))
    plt.title(f'\n\n{file}')
    plt.axis("off")
    for i in range(len(samples)):
        plt.axis()
        plt.subplot(1,len(samples),i+1)
        plt.imshow(samples[i].image, cmap='gray_r', interpolation='nearest')  # gray_r = white background
        plt.tight_layout()
        plt.title(f'Label: {samples[i].label}')
    plt.show()

import matplotlib.pyplot as plt # pyright: ignore[reportMissingModuleSource]
import sys
from pathlib import Path

neighbour_folder = Path(__file__).parent.parent / "data"
samples = []
for i in range(1,10):
    file =  Path(__file__).parent.parent / "data" / f"mnist_train-{i}.csv"
    
    with open(file) as file:
        for line in file:
            line = line.replace('"',"").split(",")
            new_sample = Sample(line[0],line[1:])
            samples.append(new_sample)
        image_sample(f"mnist_train-{i}",samples[(i*28+0):(i*28+5)])
        print("pogram terminated")
        break
    
    nn = Network([784,64,10])
    for layer in 
