import torch
import time

if torch.cuda.is_available():
    torch.cuda.synchronize()
starttime1 = time.time()
# Your CUDA code here



starttime = time.time()
tensor = torch.rand(3,4)
endtime = time.time() - starttime
print(endtime)
print(f"Device tensor is stored on: {tensor.device}")
# Device tensor is stored on: cpu

print(torch.cuda.is_available())
#True

starttime = time.time()
tensor = tensor.to('cuda')
endtime = time.time() - starttime
print(endtime)
print(f"Device tensor is stored on: {tensor.device}")


torch.cuda.synchronize()
endtime1 = time.time() - starttime1
print(endtime1)
# Device tensor is stored on: cuda:0
