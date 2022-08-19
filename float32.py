# %%
from urllib import request
import torch

import time
import torch.nn as nn


if torch.cuda.is_available():
    print("yes")
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
# %%

start = time.time()

A = torch.randint(10,(10000, 10000),  dtype = torch.int8,device=device)

x = torch.randint(10,(10000, 1),dtype = torch.int8, device=device)
b = torch.randint(10, (10000, 1),dtype = torch.int8,  device=device)

Ap = nn.Parameter(A, requires_grad=False)
bp = nn.Parameter(b, requires_grad=False)
xp = nn.Parameter(x,requires_grad=False)


sum = Ap @ xp + bp


end = time.time()

print(end - start)

# %%
