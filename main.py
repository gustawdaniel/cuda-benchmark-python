import torch
import time


def check_cuda():
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available. Running benchmark with GPU.")
        return True
    else:
        print("CUDA is not available. Running benchmark with CPU.")
        return False


def benchmark(device, size=100, repetitions=1):
    # Create random matrices for the benchmark
    # size = 10000# Change this size for different benchmarking durations
    x = torch.randn(size, size, device=device)
    y = torch.randn(size, size, device=device)

    # Warm-up
    _ = torch.matmul(x, y)

    # Benchmark start time
    start_time = time.time()

    # Run matrix multiplication
    for _ in range(repetitions):  # Run multiple times to get a more stable time measurement
        _ = torch.matmul(x, y)

    # Calculate duration
    duration = time.time() - start_time
    return duration


def main():
    use_cuda = check_cuda()

    # Benchmark on CPU
    cpu_duration = benchmark("cpu")
    print(f"CPU Benchmark Duration: {cpu_duration:.4f} seconds")

    # Benchmark on GPU if CUDA is available
    if use_cuda:
        gpu_duration = benchmark("cuda")
        print(f"GPU (CUDA) Benchmark Duration: {gpu_duration:.4f} seconds")
    else:
        gpu_duration = None

    # Compare results if both tests were conducted
    if gpu_duration and cpu_duration:
        print(f"Speedup: {cpu_duration / gpu_duration:.2f}x faster on GPU")


if __name__ == "__main__":
    main()
