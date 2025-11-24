import torch
import platform
import pytest

def print_system_info():
    print(f"OS: {platform.system()}")
    print(f"Architecture: {platform.machine()}")

def test_matrix_multiplication():
    """测试矩阵乘法的精度"""
    print_system_info()
    
    N = 128
    x = torch.linspace(0, 1, steps=N*N, dtype=torch.float32).reshape(N, N)
    y = torch.linspace(0, 1, steps=N*N, dtype=torch.float32).reshape(N, N).mT
    result = torch.matmul(x, y)
    final_sum = result.sum()
    val_as_float = final_sum.item()
    val_as_hex = hex(final_sum.view(torch.int32).item())

    print(f"Decimal Result: {val_as_float:.20f}")
    print(f"Hex Representation: {val_as_hex}")
    
    # 使用 pytest 的 assert
    assert val_as_float == 524298.68750000000000000000

# 保留原有的主函数以便单独运行
if __name__ == "__main__":
    test_matrix_multiplication()
