import cupy as cp
import time

import numpy as np

# a = cp.random.randint(low=1, high=10, size=(1000, 1000))
# b = cp.random.randint(low=1, high=10, size=(1000, 1000))


a = cp.random.rand(10, 10)
b = cp.random.rand(10, 10)

n = 1000
prod = a @ b
start = time.time()
for _ in range(1000):
    prod = a @ b

end = time.time()


print(f"using cupy {(end - start) / n}")

