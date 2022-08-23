# %%
import torch

import time
import torch.nn as nn


if torch.cuda.is_available():
    print("yes")
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


sizes = [
    [4096, 4048, 4096],
    [4096, 4056, 4096],
    [4096, 4080, 4096],
    [4096, 4096, 4096],
    [4096, 4104, 4096],
    [4096, 4128, 4096],
    [4096, 4144, 4096],
    [4096, 5096, 4096],
    [4096, 5104, 4096],
    [4096, 5112, 4096],
    [4096, 5120, 4096],
    [4096, 9728, 4096],
    [4096, 16384, 4096],
]
# %%


def calculate_time(arithmetic, n, size):

    A = torch.rand(size=(size[0], size[1]), dtype=arithmetic, device=device)

    x = torch.rand(size=(size[1], size[2]), dtype=arithmetic, device=device)

    Ap = nn.Parameter(A, requires_grad=False)
    xp = nn.Parameter(x, requires_grad=False)

    prod = torch.mm(Ap, xp)

    start = time.time()

    for _ in range(n):
        prod = Ap @ xp

    end = time.time()

    return (end - start) / n


# %%

print(f"---------------------------------------------------------------")
print(f"|         size           |    fp16_time     |    fp32_time    |")
print(f"---------------------------------------------------------------")
for size in sizes:
    fp_16_time = calculate_time(torch.float16, 1000, size=size)
    fp_32_time = calculate_time(torch.float32, 1000, size=size)
    print(
        f"| [{size[0]}],[{size[1]}],[{size[2]}]   |     {round(fp_16_time, 4)}       |      {round(fp_32_time,4)}     |"
    )

print(f"---------------------------------------------------------------")
