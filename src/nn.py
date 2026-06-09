class Node :
    def __init__(self) :
        self.inputs = []
        self.weights = []
        self.bias = 0
        self.output = 0
        
    def forward_pass(self,inputs) :
        for i in range(len(self.inputs)) :
            self.output += self.inputs[i] * self.weights[i] 
        self.output += self.bias
    
    def activate(self):
        if self.output < 0:
            self.output = 0

class Sample:
        
    def reshape(self):
        for row in range(27) :
            self.image.append([])
            for index in range(27) : 
                self.image[row].append(self.int_flat[row*28 + index]) 
        for i in self.image:
            print(i)     
            
    def normalise(self):
        for i in self.flat:
            if isinstance(i,str) :
                self.int_flat.append(int(i))
            elif isinstance(i,int):
                self.int_flat.append(i)
            else : 
                raise Exception("unprocessed pixel")
        max_val = max(self.int_flat)
        print(max_val)
        min_val = min(self.int_flat)
        print(min_val)
        print(self.int_flat)
        self.int_flat = [(val - min_val) / (max_val - min_val) for val in self.int_flat ]
        print(self.int_flat)

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
    
class Layer:
    def __init__(self,layer_size,input_size):
        self.layer_size = layer_size
        self.input_size = input_size
        self.nodes = []
        for i in range(self.layer_size):
            new_node = Node()
            self.nodes.append(new_node)
            
    def forward(self,input) : 
        for node in nodes:
            node.forward_pass(input)
    

from pathlib import Path
import sys
neighbour_folder = Path(__file__).parent.parent / "data"

for i in range(1,10):
    file =  Path(__file__).parent.parent / "data" / f"mnist_train-{i}.csv"
    with open(file) as file:
        for line in file:
            line = line.replace('"',"").split(",")
            print(line[0])
            new_sample = Sample(line[0],line[1:])
            print("pogram terminated")
            sys.exit()