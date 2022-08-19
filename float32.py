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

def calculate_time(arithmetic):
    start = time.time()

    A = torch.randint(10,(10000, 10000),  dtype = arithmetic,device=device)

    x = torch.randint(10,(10000, 5000),dtype = arithmetic, device=device)
    b = torch.randint(10, (10000, 5000),dtype = arithmetic,  device=device)

    Ap = nn.Parameter(A, requires_grad=False)
    bp = nn.Parameter(b, requires_grad=False)
    xp = nn.Parameter(x,requires_grad=False)


    sum = Ap @ xp + bp


    end = time.time()

    return end-start

float16_time = calculate_time(torch.float16)
float32_time = calculate_time(torch.float32)
float64_time = calculate_time(torch.float64)

print(f"float-16 time {float16_time}" )
print(f"float-32 time {float32_time}" )
print(f"float-64 time {float64_time}" )


# %%
