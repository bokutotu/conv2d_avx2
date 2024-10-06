import torch
import torch.nn.functional as F

def run_test(N, K, S, P, D):
    # 入力とカーネルの初期化
    input = torch.tensor([(i % 10) + 1 for i in range(N)], dtype=torch.float32).view(1, 1, -1)
    kernel = torch.tensor([(i % 3 + 1) * 0.1 for i in range(K)], dtype=torch.float32).view(1, 1, -1)

    # PyTorchでの計算
    output = F.conv1d(input, kernel, stride=S, padding=P, dilation=D)
    output = output.view(-1).tolist()

    # 結果の表示
    print("Conv1D Test Results (Python):")
    print(f"Parameters: N={N}, K={K}, S={S}, P={P}, D={D}")
    print(f"Input: {input.view(-1).tolist()}")
    print(f"Kernel: {kernel.view(-1).tolist()}")
    print(f"Output: {output}")
    print()

def main():
    # 複数のテストケース
    run_test(60, 6, 3, 3, 1)  # 大きな入力、大きなストライドとパディング

if __name__ == "__main__":
    main()
