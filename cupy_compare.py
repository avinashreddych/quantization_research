import cupy as cp
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
    A = cp.random.rand(size[0], size[1]).astype(arithmetic)
    x = cp.random.rand(size[1], size[2]).astype(arithmetic)

    prod = A @ x

    start = time.time()
    for _ in range(n):
        prod = A @ x

    end = time.time()
    return (end - start) / n


print(
    f"|--------------------------------------------------------------------------------------|"
)
print(
    f"|         size           |  cupy_int8    |    cupy_fp16_time    |    cupy_fp32_time    |"
)
print(
    f"|--------------------------------------------------------------------------------------|"
)
n = 100
for size in sizes:
    cp_int_time = cupy_int_calculate_time(n, size)
    cp_fp16_time = cupy_calculate_time(cp.float16, n, size)
    cp_fp32_time = cupy_calculate_time(cp.float32, n, size)
    print(
        f"| [{size[0]}],[{size[1]}],[{size[2]}]   |     {round(cp_int_time, 4)}    |    {round(cp_fp16_time,4)}    |      {round(cp_fp32_time,4)}     |"
    )
