from engine import Value
import numpy as np

class Module:
    '''base class for all neural network modules'''
    def zero_grad(self):
        '''sets all the gradients to zero'''
        for p in self.parameters():
            p.grad = 0.0
    
    def parameters(self):
        '''returns the list of all parameters in the module'''
        return []

class Neuron(Module):
  '''class to define a single neuron and its properties'''

  def __init__(self, nin, nonlin=True):
    '''nin -> number of inputs to each neuron'''
    self.w = [Value(np.random.uniform(-1,1)) for _ in range(nin)] # initialise the neuron with random weights for each input to the neuron
    self.b = Value(0) # initialise a random bias for each neuron
    self.nonlinear = nonlin # by default we use a non linear activation function

  def __call__(self, x):
    ''' w * x + b -> forward pass'''
    activation = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
    out = activation.relu() if self.nonlinear else activation
    return out

  def parameters(self):
    return self.w + [self.b]

  def __repr__(self):
    return f"{'ReLU' if self.nonlinear else 'Linear'} Neuron({len(self.w)})"

class Layer(Module):
  '''
    Class to define a layer of neurons.
    A layer of neurons is a set of neurons evaluated independently.
    A layer of neurons is a set of neurons connected to the same inputs and outputs
    but not connected to each other.
    '''

  def __init__(self, nin, non, **kwargs):
    '''
    nin -> number of inputs to each neuron
    non -> number of neurons in the layer
    '''
    self.neurons = [Neuron(nin, non) for _ in range(non)]

  def __call__(self, x):
    '''does the forward pass'''
    out = [n(x) for n in self.neurons]
    return out[0] if len(out) == 1 else out

  def parameters(self):
    return [p for neuron in self.neurons for p in neuron.parameters()]

  def __repr__(self):
    return f"Layer of [{', '.join(str(n) for n in self.neurons)}"

class MLP(Module):

  def __init__(self, nin, nons):
    '''nin -> number of inputs to each neuron
       nons -> list of the number of neurons per
              layer and it defines the sizes of all the
              layers we want in the MLP'''
    size = [nin] + nons
    self.layers = [Layer(size[i], size[i+1], nonlinear=i != len(nons)-1) for i in range(len(nons))]

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]
  
  def __repr__(self):
    return f"MLP of [{','.join(str(layer) for layer in self.layers)}]"