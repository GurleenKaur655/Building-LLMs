import torch
import torch.nn as nn 

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))

class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        #construct a neural network using sequential layers with GELU activation functions
        self.layers = nn.ModuleList([
            #1st in input and 2nd is the output and gelu applied then 
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU()),
        ])

    def forward(self, x):
        for layer in self.layers:
            #compute the output of the current layer 
            layer_output = layer(x)
            #check if the shortcut can be applied 
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output 
            else:
                x = layer_output
        return x
    
layer_sizes = [3,3,3,3,3,1] # how many neurons u want in each layer 5 layers with 3 neurons each and 1 output neuron
sample_input = torch.tensor([[1., 0., -1.]])
# 3*3 weight matrix because 3 neurons in the first layer and 3 neurons in the second layer w.shape = (3,3) (out, in)
# rows = neurons (3) columns = inputs (3)
torch.manual_seed(123) # specify random seed for the initial weights for reproducibility 
model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)

def print_gradients(model, x):
    # Forward pass 
    output = model(x)
    target = torch.tensor([[0.]])

    #Calculate loss based on how close the target 
    # and output are 
    loss = nn.MSELoss() # MSELoss is a common loss function for regression tasks, measuring the average squared difference between predicted and actual values.
    loss = loss(output, target)

    #Backward pass to calculate the gradients 
    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            # Print the mean absolute gradient of the weights 
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

print_gradients(model_without_shortcut, sample_input)
# mean gradient is basically the average of the absolute values of the gradients for each weight parameter in the model. It gives an indication of how much the weights are being updated during training. A higher mean gradient suggests that the weights are being adjusted more significantly, while a lower mean gradient indicates smaller updates.

# layers.0.0.weight has gradient mean of 0.00020173587836325169
# layers.1.0.weight has gradient mean of 0.0001201116101583466
# layers.2.0.weight has gradient mean of 0.0007152041071094573
# layers.3.0.weight has gradient mean of 0.0013988735154271126
# layers.4.0.weight has gradient mean of 0.005049645435065031

torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)
print_gradients(model_with_shortcut, sample_input)
# layers.0.0.weight has gradient mean of 0.22169791162014008
# layers.1.0.weight has gradient mean of 0.20694106817245483
# layers.2.0.weight has gradient mean of 0.32896995544433594
# layers.3.0.weight has gradient mean of 0.2665732204914093
# layers.4.0.weight has gradient mean of 1.3258540630340576