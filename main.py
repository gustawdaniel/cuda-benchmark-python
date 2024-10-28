import torch
import time

def is_cuda_available():
    """Check if CUDA is available."""
    if torch.cuda.is_available():
        return True
    return False

def benchmark(device, size=10, repetitions=1):
    """Benchmark matrix multiplication on the specified device."""
    x = torch.randn(size, size, device=device)
    y = torch.randn(size, size, device=device)

    # Warm-up to ensure accurate timing
    _ = torch.matmul(x, y)

    if device == 'cuda':
        torch.cuda.synchronize()

    start_time = time.perf_counter_ns()
    for _ in range(repetitions):
        _ = torch.matmul(x, y)

    if device == 'cuda':
        torch.cuda.synchronize()

    duration = (time.perf_counter_ns() - start_time) / 1e9  # Convert to seconds

    return duration

def main():
    use_cuda = is_cuda_available()
    use_cpu = True

    size = 10000
    repetitions = 1

    print(f"CUDA is {'' if use_cuda else 'not '}available. Benchmark matrix {size}x{size} multiplied {repetitions} times.")
    # Run CPU benchmark
    cpu_duration = None
    if use_cpu:
        cpu_duration = benchmark("cpu", size, repetitions)
        print(f"CPU Benchmark Duration: {cpu_duration:.8f} seconds")

    # Run GPU benchmark if CUDA is available
    gpu_duration = None
    if use_cuda:
        gpu_duration = benchmark("cuda", size, repetitions)
        print(f"GPU Benchmark Duration: {gpu_duration:.8f} seconds")

    # Compare results if both tests were conducted
    if gpu_duration and cpu_duration:
        speedup = cpu_duration / gpu_duration
        print(f"Speedup: {speedup:.4f}x faster on GPU")

if __name__ == "__main__":
    main()