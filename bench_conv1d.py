import torch
import torch.nn.functional as F
import time

def benchmark_conv1d(N, K, S, P, D, num_runs):
    # 入力とカーネルの初期化
    input = torch.randn(1, 1, N, dtype=torch.float32)
    kernel = torch.randn(1, 1, K, dtype=torch.float32)

    # GPUが利用可能な場合はGPUを使用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = input.to(device)
    kernel = kernel.to(device)

    # ウォームアップ実行
    _ = F.conv1d(input, kernel, stride=S, padding=P, dilation=D)

    # ベンチマーク実行
    torch.cuda.synchronize()  # GPUの場合、同期を確実にする
    start = time.time()
    for _ in range(num_runs):
        _ = F.conv1d(input, kernel, stride=S, padding=P, dilation=D)
    torch.cuda.synchronize()  # GPUの場合、同期を確実にする
    end = time.time()

    # 実行時間の計算
    avg_time = (end - start) * 1000 / num_runs  # ミリ秒単位

    # 結果の表示
    print("PyTorch Conv1D Benchmark Results:")
    print(f"Parameters: N={N}, K={K}, S={S}, P={P}, D={D}")
    print(f"Device: {device}")
    print(f"Number of runs: {num_runs}")
    print(f"Average execution time: {avg_time:.2f} ms")
    print()

def main():
    # ベンチマークの実行
    benchmark_conv1d(10000, 101, 1, 50, 1, 100)  # 100万要素の入力、100回実行

if __name__ == "__main__":
    main()
