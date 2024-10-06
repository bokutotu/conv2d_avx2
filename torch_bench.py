import torch
import time

def run_benchmark(batch_size, in_channels, out_channels, input_height, input_width, kernel_size, num_iterations):
    # Ensure we're using CPU
    device = "cpu"
    
    # Create random input and weight tensors
    input = torch.randn(batch_size, in_channels, input_height, input_width)
    weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size)
    
    # Create the convolutional layer
    conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, bias=False)
    conv.weight.data = weight
    
    # Warm-up run
    _ = conv(input)
    
    # Start the timer
    start_time = time.time()
    
    # Run the benchmark
    for _ in range(num_iterations):
        output = conv(input)
    
    # Stop the timer
    end_time = time.time()
    
    # Calculate the total and average time
    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    
    # Print benchmark results
    print(f"Benchmark results:")
    print(f"Device: CPU (single thread)")
    print(f"Input size: {batch_size}x{in_channels}x{input_height}x{input_width}")
    print(f"Filter size: {out_channels}x{in_channels}x{kernel_size}x{kernel_size}")
    print(f"Output size: {batch_size}x{out_channels}x{input_height}x{input_width}")
    print(f"Number of iterations: {num_iterations}")
    print(f"Total time: {total_time:.6f} seconds")
    print(f"Average time per iteration: {avg_time:.6f} seconds")
    
    # Print first few output values
    print("\nFirst few output values:")
    print(output[0, 0, :5, :5].detach().numpy())

if __name__ == "__main__":
    # Define benchmark parameters
    batch_size = 32
    in_channels = 64
    out_channels = 128
    input_height = 224
    input_width = 224
    kernel_size = 3
    num_iterations = 10
    
    # Run benchmark
    run_benchmark(batch_size, in_channels, out_channels, input_height, input_width, kernel_size, num_iterations)
