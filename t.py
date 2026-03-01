import torch

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

if torch.cuda.is_available():
    n = torch.cuda.device_count()
    print("Number of GPUs:", n)
    for i in range(n):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
