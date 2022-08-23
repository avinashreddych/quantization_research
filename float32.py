# %%
import torch

import time
import torch.nn as nn


if torch.cuda.is_available():
    print("yes")
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
# %%

def calculate_time(arithmetic, n):
    start = time.time()

    A = torch.rand((10000, 10000),  dtype = arithmetic,device=device)

    x = torch.rand((10000, 5000),dtype = arithmetic, device=device)

    Ap = nn.Parameter(A, requires_grad=False)
    xp = nn.Parameter(x,requires_grad=False)


    for _ in range(n):
        sum = Ap @ xp 


    end = time.time()

    return end-start



# %%
