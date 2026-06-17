class Network:
    def __init__(self,layer_sizes):
        self.network = []
        self.layers = []
        for layer_size in range(len(layer_sizes)):
            new_layer = Layer(layer_size)
            self.layers.append(new_layer)
    
    def forward(self,x):
        for layer in self.network:
            layer.forward_pass()
            
    def predict(self,x):
        outputs = self.forward_pass(x)
        return outputs.index.max(outputs)
        
class Layer:
    def __init__(self,size):
        self.data = []
        self.output = None
        self.nodes = []
        self.weights = [[0.1,0.1,0.1],[0.1,0.1,0.1],[0.1,0.1,0.1]]
        self.bias = 0.1

    def forward_pass(self,input,output_x,output_y,filter_len):
        filter = []
        self.output = [[0 for x in range(output_x)] for y in range(output_y)]

        for y in range(output_y):
            for x in range(output_x) :
                pixel = 0

                for ky in range(filter_len):
                    for kx in range(filter_len):
                        pixel += input[y+ky][x+kx] *self.weights[ky][kx]
                self.output[y][x] = max(self.bias + pixel , 0)

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
    print(f"file {i} processed")
    break # remove when you want all files analysed
    
nn = Network([784,676,400,100,10])

# Create a simple 28x28 test image (all zeros)
test_image = [[0 for _ in range(28)] for _ in range(28)]

layer = Layer(size=676)
layer.forward_pass(test_image, 26, 26, 3)

print("Output shape:", len(layer.output), "x", len(layer.output[0]))
print("First value:", layer.output[0][0])
