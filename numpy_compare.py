import numpy as np

import time

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


def numpy_int_calculate_time(n, size):
    A = np.random.randint(low=0, high=255, size=(size[0], size[1]))
    x = np.random.randint(low=0, high=255, size=(size[1], size[2]))

    prod = A @ x

    start = time.time()
    for _ in range(n):
        prod = A @ x

    end = time.time()
    return (end - start) / n


def numpy_calculate_time(arithmetic, n, size):
    A = np.random.rand(size[0], size[1], dtype=arithmetic)
    x = np.random.rand(size[1], size[2], dtype=arithmetic)

    prod = A @ x

    start = time.time()
    for _ in range(n):
        prod = A @ x

    end = time.time()
    return (end - start) / n


print(
    f"|-----------------------------------------------------------------------------------------|"
)
print(
    f"|         size           |  numpy_int8    |    numpy_fp16_time    |    numpy_fp32_time    |"
)
print(
    f"|-----------------------------------------------------------------------------------------|"
)
n = 100
for size in sizes:
    np_int_time = numpy_int_calculate_time(n, size)
    np_fp16_time = numpy_calculate_time(np.float16, n, size)
    np_fp32_time = numpy_calculate_time(np.float32, n, size)
    print(
        f"| [{size[0]}],[{size[1]}],[{size[2]}]   |     {round(np_int_time, 4)}    |    {round(np_fp16_time,4)}    |      {round(np_fp32_time,4)}     |"
    )
