import torch

import time
import torch.nn as nn
import tensorflow as tf
import cupy as cp

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


def torch_calculate_time(arithmetic, n, size):

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


def tf_calculate_time(arithmetic, n, size):
    A = tf.random.normal(shape=(size[0], size[1]), dtype=arithmetic)
    x = tf.random.normal(shape=(size[1], size[2]), dtype=arithmetic)

    Ap = tf.Variable(A)
    xp = tf.Variable(x)

    prod = tf.matmul(Ap, xp)

    start = time.time()
    for _ in range(n):
        prod = tf.matmul(Ap, xp)

    end = time.time()

    return (end - start) / n


def cupy_int_calculate_time(n, size):
    A = cp.random.randint(low=0, high=255, size=(size[0], size[1]))
    x = cp.random.randint(low=0, high=255, size=(size[1], size[2]))

    prod = A @ x

    start = time.time()
    for _ in range(n):
        prod = A @ x

    end = time.time()
    return (end - start) / n


def cupy_calculate_time(arithmetic, n, size):
    A = cp.random.rand(size[0], size[1], dtype=arithmetic)
    x = cp.random.rand(size[1], size[2], dtype=arithmetic)

    prod = A @ x

    start = time.time()
    for _ in range(n):
        prod = A @ x

    end = time.time()
    return (end - start) / n


n = 1000
print(
    f"|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|"
)
print(
    f"|         size           |  torch_fp16_time   |  torch_fp32_time  |  tf_fp16_time    |    tf_fp32_time    |   cupy_int8    |    cupy_fp16_time    |    cupy_fp32_time    |"
)
print(
    f"|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|"
)
for size in sizes:
    torch_fp16_time = torch_calculate_time(torch.float16, n, size)
    torch_fp32_time = torch_calculate_time(torch.float, n, size)
    tf_fp16_time = tf_calculate_time(tf.dtypes.float16, n, size)
    tf_fp32_time = tf_calculate_time(tf.dtypes.float32, n, size)
    cp_int_time = cupy_int_calculate_time(n, size)
    cp_fp16_time = cupy_calculate_time(cp.float16, n, size)
    cp_fp32_time = cupy_calculate_time(cp.float32, n, size)
    print(
        f"| [{size[0]}],[{size[1]}],[{size[2]}]   |     {round(torch_fp16_time, 4)}       |      {round(torch_fp32_time,4)}     |        {round(tf_fp16_time, 4)}    |      {round(tf_fp32_time,4)}     |     {round(cp_int_time, 4)}    |    {round(cp_fp16_time,4)}    |      {round(cp_fp32_time,4)}     |"
    )

