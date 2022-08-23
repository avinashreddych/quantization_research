import torch
import time


def calculate_time(arithmetic, n, tf32: bool):
    a_full = torch.randn(10240, 10240, dtype=arithmetic, device="cuda")
    b_full = torch.randn(10240, 10240, dtype=arithmetic, device="cuda")

    if tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    start = time.time()
    for _ in range(n):
        ab_full = torch.matmul(a_full, b_full)

    end = time.time()

    return end - start


# with_tf_32 = calculate_time(torch.float32, 1000, True)
# print(f"with_tf32_time: {with_tf_32/1000} ")


without_tf_32 = calculate_time(torch.float32, 1000, False)
print(f"without_tf_32_time: {without_tf_32/1000}")


# import torch
# import time

# a_full = torch.randn(10240, 10240, dtype=torch.double, device="cuda")
# b_full = torch.randn(10240, 10240, dtype=torch.double, device="cuda")
# ab_full = a_full @ b_full
# mean = ab_full.abs().mean()  # 80.7277

# a = a_full.float()
# b = b_full.float()

# # Do matmul at TF32 mode.

# torch.backends.cuda.matmul.allow_tf32 = True
# start = time.time()
# ab_tf32 = a @ b  # takes 0.016s on GA100
# end = time.time()
# print(end - start)
# error = (ab_tf32 - ab_full).abs().max()  # 0.1747
# relative_error = error / mean  # 0.0022

# print(relative_error)

# # Do matmul with TF32 disabled.
# torch.backends.cuda.matmul.allow_tf32 = False
# start = time.time()

# ab_fp32 = a @ b  # takes 0.11s on GA100
# end = time.time()
# print(end - start)
# error = (ab_fp32 - ab_full).abs().max()  # 0.0031
# relative_error = error / mean  # 0.000039

# print(relative_error)
