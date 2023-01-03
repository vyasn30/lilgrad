import numpy as np
from typing import Optional

class Tensor:
  def __init__(self, data, requires_grad=None):
    
    if not isinstance(data, np.ndarray):
      self.data = np.array(data, dtype=np.float32)

    elif np.isscalar(data):
      self.data = np.float_(data)
    else:
      self.data = data

  
    self.requires_grad = requires_grad
    self.grad = 0
    self._backward = lambda:None

  def __repr__(self):
    return f"{self.data}"

  @classmethod
  def eye(cls,dim) : return cls(np.eye(dim))
  

  def matmul(self, other):
    out = Tensor(np.matmul(self.data, other.data))

    def _backward():
      print(f"calculating grads of x and y")
      print(f"ygrad = x.data({other.data}) * out.grad({out.grad})")
      self.grad = other.data @ out.grad
      print(self.grad)
      print(f"x.grad = y.data({self.data}) * out.grad({out.grad})")
      other.grad = self.data @ out.grad
      print(other.grad)
    out._backward = _backward
    return out


  def sum(self): 
    out = Tensor(np.float_(self.data.sum()))
    
    def _backward():
      self.grad = out.grad * np.ones_like(self.data)
      

    out._backward = _backward
    return out
      
     