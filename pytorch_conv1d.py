import torch
import torch.nn as nn
import time

def print_tensor(tensor, name):
    # テンソルを1次元に変換し、各要素を小数点以下2桁で表示
    formatted = [f"{x:.2f}" for x in tensor.reshape(-1).tolist()]
    print(f"{name}: [{', '.join(formatted)}]")


def run_test(N, C_in, C_out, K, S, P, D):
    # 入力とカーネルの初期化
    input = torch.tensor([(i % 10) + 1 for i in range(N * C_in)], dtype=torch.float32).reshape(1, C_in, N)
    conv1d = nn.Conv1d(C_in, C_out, K, stride=S, padding=P, dilation=D)
    
    # カーネルの重みを設定
    with torch.no_grad():
        conv1d.weight.data = torch.tensor([(i % 3 + 1) * 0.1 for i in range(K * C_in * C_out)], 
                                          dtype=torch.float32).reshape(C_out, C_in, K)
        conv1d.bias.data.zero_()

    # Conv1d 実行
    output = conv1d(input)

    # 結果の表示
    print("PyTorch Conv1D Test Results:")
    print(f"Parameters: N={N}, C_in={C_in}, C_out={C_out}, K={K}, S={S}, P={P}, D={D}")
    print_tensor(input, "Input")
    print_tensor(conv1d.weight, "Kernel")
    print_tensor(output, "Output")
    print()

def benchmark_conv1d(N, C_in, C_out, K, S, P, D, num_runs):
    # 入力の初期化
    input = torch.randn(1, C_in, N)
    conv1d = nn.Conv1d(C_in, C_out, K, stride=S, padding=P, dilation=D)

    # ウォームアップ
    for _ in range(10):
        _ = conv1d(input)

    # ベンチマーク実行
    start_time = time.time()
    for _ in range(num_runs):
        _ = conv1d(input)
    end_time = time.time()

    # 実行時間の計算
    avg_time = (end_time - start_time) * 1000 / num_runs

    # 結果の表示
    print("PyTorch Conv1D Benchmark Results:")
    print(f"Parameters: N={N}, C_in={C_in}, C_out={C_out}, K={K}, S={S}, P={P}, D={D}")
    print(f"Number of runs: {num_runs}")
    print(f"Average execution time: {avg_time:.4f} ms")
    print()

if __name__ == "__main__":
    # テストの実行
    run_test(10, 2, 3, 3, 1, 0, 1)

    # ベンチマークの実行
    benchmark_conv1d(1000, 3, 5, 5, 1, 1, 1, 10000)
