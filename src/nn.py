import random
class Network:
    def __init__(self,layer_sizes):
        self.network = []
        self.layers = []
        for layer_size in range(len(layer_sizes)):
            new_layer = Layer(layer_size)
            self.layers.append(new_layer)
    
    def forward(self,x):
        for layer in self.network:
            layer.forward()
            
    def predict(self,x):
        outputs = self.forward(x)
        return outputs.index.max(outputs)
        
class Layer:
    def __init__(self,filter_num, filter_size = 3):
        self.filter_num = filter_num
        self.filter_size = filter_size
        
        self.filters = [[[random.uniform(-0.1,0.1) for i in range(filter_size)] for j in range(filter_size)] for k in range(filter_num)]
        self.biases = [0.0 for i in range(filter_num)]
            
    def forward(self,input_image):
        h = len(input_image)
        w = len(input_image[0])
        
        out_h = h - self.filter_size + 1
        out_w = w - self.filter_size + 1
        
        self.output = [[[0 for i in range (out_w)] for j in range(out_h)] for k in range(self.filter_num)]
        
        for filter in range(self.filter_num):
            for y in range(out_h):
                for x in range(out_w):
                    pixel = 0
                    
                    for ky in range(self.filter_size):
                        for kx in range(self.filter_size):
                            pixel += input_image[y+ky][x+kx] * self.filters[filter][ky][kx]
                            
                    self.output[filter][y][x] = max(pixel + self.biases[filter], 0)
        return self.output
                    
class Sample:
        
    def reshape(self):
        for row in range(28) :
            self.image.append([])
            for index in range(28) : 
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
        self.label = int(label)
        self.flat = image
        self.prediction = -1
        identity_matrix =  [[1,0,0,0,0,0,0,0,0,0],
                            [0,1,0,0,0,0,0,0,0,0],
                            [0,0,1,0,0,0,0,0,0,0],
                            [0,0,0,1,0,0,0,0,0,0],
                            [0,0,0,0,1,0,0,0,0,0],
                            [0,0,0,0,0,1,0,0,0,0],
                            [0,0,0,0,0,0,1,0,0,0],
                            [0,0,0,0,0,0,0,1,0,0],
                            [0,0,0,0,0,0,0,0,1,0],
                            [0,0,0,0,0,0,0,0,0,1]]
        self.one_hot_label = identity_matrix[self.label]
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
    print(f"file {i} processed")
    break # remove when you want all files analysed
    
nn = Network([784,676,400,100,10])

test_image = [[0 for _ in range(28)] for _ in range(28)]

# Initialize the layer with 8 filters
layer = Layer(filter_num=8, filter_size=3) 

# Forward pass only requires the image now
layer.forward(test_image)

print("Output shape:", len(layer.output), "filters x", len(layer.output[0]), "x", len(layer.output[0][0]))