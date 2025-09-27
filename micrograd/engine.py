import numpy as np

class Value:
  '''Defines a scalar value, some math operations and its gradients'''

  def __init__(self, data, _children=(), _op='', label=''):
    self.data = data
    self.grad = 0.0 # derivative of the output, wrt this variable

    # internal variables used for math operations and gradient calculation
    self._backward = lambda: None
    self._prev = set(_children) # set to hold the child nodes
    self._op = _op
    self.label = label

  def __hash__(self):
    return id(self)

  def __repr__(self):
    '''print out a nice looking expression rather that the memory address'''

    return f"Value(Data={self.data})"

# -------------------------------------------------------------------
# Mathematical functions

  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
      '''
      Function to calculate the local gradient.
      Since this is a '+' node, the local gradient wrt to the output is 1,
      then we multiply by the gradient(effect) of the gradient being
      propagated backwards all the way from the output -> chain rule
      '''
      self.grad += 1.0 * out.grad # += is to accumulate the gradients rather than set them outright as using the same variable more than once will overide what was initially set
      other.grad += 1.0 * out.grad
    out._backward = _backward
    return out

  def __radd__(self, other):
    return self + other

  def __mul__(self, other):
    '''performs a multiplication function'''
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
      '''returns the function's gradient'''
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward
    return out

  def __rmul__(self, other):
    '''fallback multiplication function'''
    return self * other

  def __pow__(self, other):
    '''performs a power function'''
    assert isinstance(other, (int, float)), "only supporting int or float powers"
    out = Value(self.data**other, (self,), f'**{other}')

    def _backward():
      '''returns the functions gradient'''
      self.grad += other * (self.data**(other-1)) * out.grad
    out._backward = _backward
    return out

  def __truediv__(self, other):
    '''performs a division'''
    other = other if isinstance(other, Value) else Value(other)
    self = self if isinstance(self, Value) else Value(self)
    return self * other**-1

  def __rtruediv__(self, other): # other / self
        return other * self**-1

  def __neg__(self):
    '''returns the negative of the value'''
    return self * -1

  def __sub__(self, other):
    '''performs a subtraction'''
    return self + (-other)

  def __rsub__(self, other): # other - self
    return other + (-self)

  def exp(self):
    '''performs an exponential function'''
    x = self.data
    out = Value(np.exp(x), (self, ), 'exp')

    def _backward():
      '''returns the functions gradient'''
      self.grad += out.data * out.grad
    out._backward = _backward
    return out

# ------------------------------------------------------------------
# Comparison operators
  # def __lt__(self, other):
  #   return True if self.data < other.data else False

  # def __gt__(self, other):
  #   return True if self.data > other.data else False

  # def __le__(self, other):
  #   return True if self.data <= other.data else False

  # def __ge__(self, other):
  #   return True if self.data >= other.data else False

  # def __eq__(self, other):
  #   return True if self.data == other.data else False
  
# ------------------------------------------------------------------
# activation functions
  def tanh(self):
    x = self.data
    t = (np.exp(2*x) - 1) / (np.exp(2*x) + 1)
    out = Value(t, (self, ), 'tanh')

    def _backward():
      '''returns the functions gradient'''
      self.grad += (1 - t**2) * out.grad
    out._backward = _backward
    return out

  def relu(self):
    out = Value(0 if self.data < 0 else self.data, (self, ), 'ReLU')

    def _backward():
      '''returns the functions gradient'''
      self.grad += (out.data > 0) * out.grad
    out._backward = _backward
    return out

  def leaky_relu(self, alpha=0.01):
    out = Value(self.data if self.data > 0 else alpha * self.data, (self, ), 'LeakyReLU')

    def _backward():
        self.grad += (1 if out.data > 0 else alpha) * out.grad
    out._backward = _backward
    return out

# -------------------------------------------------------------------
# backpropagation
  def backward(self):
     '''autogradient function'''
     # first we build a topographical map of the nodes
     topo = []
     visited = set()
     def build_topo(v):
       if v not in visited:
         visited.add(v)
         for child in v._prev:
           build_topo(child)
         topo.append(v)
     build_topo(self)
     # base case is that the output node has a gradient of 1
     self.grad = 1.0
     # call the gradient function on each node in reverse
     for node in reversed(topo):
       node._backward()