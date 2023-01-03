from tensor import Tensor

x = Tensor.eye(dim=3)
print(x.data.shape)
y = Tensor([2.0, 1.0, 3.0])
print(y.data.shape)
# print(f"y = {y}")
xy = y.matmul(x)
print(xy.data.shape)
# print(xy)
# print(xy.grads)
# xy._backward()
z = xy.sum()
# print(z)
# print(z.grads)
z.grad = 1
z._backward()

print(xy.grad)    #dz/dxy
xy._backward() 
# print(z.grad)     #dz/dz
 
# print(x.grad)
# print(y.grad)
# print(z._backward())