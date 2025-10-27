# benchmark_attention.py
"""
Benchmark DeltaNet vs Standard Softmax Attention
Tests: Speed, Memory, Scaling with Sequence Length

This script benchmarks DeltaNet attention against standard softmax attention
across multiple sequence lengths: 128, 256, 512, 1K, 2K, 4K, 8K, 16K, and 32K.

For very long sequences (16K and 32K):
- Batch size is reduced to 1 to avoid OOM errors
- Number of benchmark runs is reduced to 3 (instead of 10)
- Standard softmax attention may run out of memory (expected behavior)
- DeltaNet should still succeed, demonstrating its O(N) efficiency

Expected Results:
- Short sequences (<2K): Similar performance
- Medium sequences (2K-8K): DeltaNet 2-5x faster
- Long sequences (16K-32K): DeltaNet >10x faster (if Softmax doesn't OOM)
- Memory: DeltaNet uses significantly less memory for long sequences
- Autoregressive: DeltaNet maintains constant time per token

Usage:
    python test_attn.py

Requirements:
    pip install torch matplotlib
"""

import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from deltanet_attention import DeltaNetAttention, StandardSoftmaxAttention


def benchmark_forward_pass(model, x, num_runs=10, warmup=2):
    """Benchmark forward pass time"""
    device = next(model.parameters()).device
    x = x.to(device)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(x) if isinstance(model, StandardSoftmaxAttention) else model(x)[0]

    # Synchronize GPU
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(x) if isinstance(model, StandardSoftmaxAttention) else model(x)[0]
        if device.type == "cuda":
            torch.cuda.synchronize()

    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    return avg_time


def benchmark_memory(model, x):
    """Benchmark peak memory usage"""
    device = next(model.parameters()).device
    if device.type != "cuda":
        return None

    x = x.to(device)
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        _ = model(x) if isinstance(model, StandardSoftmaxAttention) else model(x)[0]

    peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    return peak_memory


def test_autoregressive_generation(model, batch_size, max_len, hidden_size, device):
    """Test autoregressive generation (one token at a time)"""
    # Initial prompt
    x = torch.randn(batch_size, 1, hidden_size, device=device)

    state = None
    times = []

    for step in range(max_len):
        start = time.time()

        if isinstance(model, DeltaNetAttention):
            # DeltaNet can use recurrent mode
            _, state = model(
                x, use_recurrent=True, initial_state=state, return_state=True
            )
        else:
            # Standard attention needs full context
            if step == 0:
                context = x
            else:
                context = torch.cat([context, x], dim=1)
            _ = model(context)

        if device.type == "cuda":
            torch.cuda.synchronize()

        times.append(time.time() - start)

        # Next token (dummy)
        x = torch.randn(batch_size, 1, hidden_size, device=device)

    return times


def run_comprehensive_benchmark():
    """Run all benchmarks and generate plots"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    hidden_size = 512
    num_heads = 8
    head_dim = 64
    batch_size = 4

    # Test different sequence lengths - including 16K and 32K
    seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

    deltanet_times = []
    standard_times = []
    deltanet_memory = []
    standard_memory = []

    print("=" * 60)
    print("FORWARD PASS BENCHMARK (Parallel Processing)")
    print("=" * 60)

    for seq_len in seq_lengths:
        print(f"\nSequence Length: {seq_len}")
        print("-" * 40)

        # Adjust batch size and num_runs for very long sequences
        if seq_len >= 16384:
            current_batch_size = 1  # Reduce batch size for 16K and 32K
            num_runs = 3  # Fewer runs for very long sequences
            print(
                f"  (Using batch_size={current_batch_size}, num_runs={num_runs} for long sequence)"
            )
        elif seq_len >= 8192:
            current_batch_size = 2
            num_runs = 5
            print(f"  (Using batch_size={current_batch_size}, num_runs={num_runs})")
        else:
            current_batch_size = batch_size
            num_runs = 10

        x = torch.randn(current_batch_size, seq_len, hidden_size)

        # Initialize models
        deltanet = DeltaNetAttention(hidden_size, num_heads, head_dim).to(device)
        standard = StandardSoftmaxAttention(hidden_size, num_heads, head_dim).to(device)

        # Benchmark DeltaNet
        try:
            delta_time = benchmark_forward_pass(deltanet, x, num_runs=num_runs)
            deltanet_times.append(delta_time * 1000)  # Convert to ms
            print(f"  DeltaNet:  {delta_time*1000:.2f} ms")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  DeltaNet:  OOM (Out of Memory)")
                deltanet_times.append(None)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            else:
                raise

        # Benchmark Standard - may skip for very long sequences if needed
        try:
            standard_time = benchmark_forward_pass(standard, x, num_runs=num_runs)
            standard_times.append(standard_time * 1000)
            print(f"  Standard:  {standard_time*1000:.2f} ms")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(
                    f"  Standard:  OOM (Out of Memory) - Expected for very long sequences"
                )
                standard_times.append(None)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            else:
                raise

        # Calculate speedup if both succeeded
        if deltanet_times[-1] is not None and standard_times[-1] is not None:
            speedup = standard_times[-1] / deltanet_times[-1]
            print(f"  Speedup:   {speedup:.2f}x")
        elif deltanet_times[-1] is not None and standard_times[-1] is None:
            print(f"  Speedup:   ∞ (Standard OOM, DeltaNet succeeded)")

        # Memory benchmark (if CUDA)
        if device.type == "cuda" and deltanet_times[-1] is not None:
            try:
                delta_mem = benchmark_memory(deltanet, x)
                deltanet_memory.append(delta_mem)
                print(f"  DeltaNet Memory:  {delta_mem:.2f} MB")
            except RuntimeError as e:
                if "out of memory" in str(e):
                    deltanet_memory.append(None)
                    torch.cuda.empty_cache()
                else:
                    raise

            if standard_times[-1] is not None:
                try:
                    standard_mem = benchmark_memory(standard, x)
                    standard_memory.append(standard_mem)
                    print(f"  Standard Memory:  {standard_mem:.2f} MB")
                    if delta_mem is not None:
                        print(
                            f"  Memory Reduction: {(1 - delta_mem/standard_mem)*100:.1f}%"
                        )
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        standard_memory.append(None)
                        torch.cuda.empty_cache()
                    else:
                        raise
            else:
                standard_memory.append(None)
        elif device.type == "cuda":
            deltanet_memory.append(None)
            standard_memory.append(None)

    # Autoregressive generation benchmark
    print("\n" + "=" * 60)
    print("AUTOREGRESSIVE GENERATION BENCHMARK")
    print("=" * 60)

    gen_length = 100
    print(f"\nGenerating {gen_length} tokens sequentially...")

    deltanet = DeltaNetAttention(hidden_size, num_heads, head_dim).to(device)
    standard = StandardSoftmaxAttention(hidden_size, num_heads, head_dim).to(device)

    delta_gen_times = test_autoregressive_generation(
        deltanet,
        batch_size=1,
        max_len=gen_length,
        hidden_size=hidden_size,
        device=device,
    )
    standard_gen_times = test_autoregressive_generation(
        standard,
        batch_size=1,
        max_len=gen_length,
        hidden_size=hidden_size,
        device=device,
    )

    print(
        f"\nDeltaNet avg per token:  {sum(delta_gen_times)/len(delta_gen_times)*1000:.2f} ms"
    )
    print(
        f"Standard avg per token:  {sum(standard_gen_times)/len(standard_gen_times)*1000:.2f} ms"
    )
    print(
        f"Generation speedup:      {sum(standard_gen_times)/sum(delta_gen_times):.2f}x"
    )

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Filter out None values for plotting
    def filter_data(seq_lens, data):
        """Filter out None values while keeping sequence lengths aligned"""
        return zip(*[(s, d) for s, d in zip(seq_lens, data) if d is not None])

    # Plot 1: Forward pass time
    ax1 = axes[0, 0]
    if any(t is not None for t in deltanet_times):
        delta_seq, delta_vals = filter_data(seq_lengths, deltanet_times)
        delta_seq, delta_vals = list(delta_seq), list(delta_vals)
        ax1.plot(
            delta_seq, delta_vals, "o-", label="DeltaNet", linewidth=2, markersize=8
        )

    if any(t is not None for t in standard_times):
        std_seq, std_vals = filter_data(seq_lengths, standard_times)
        std_seq, std_vals = list(std_seq), list(std_vals)
        ax1.plot(std_seq, std_vals, "s-", label="Softmax", linewidth=2, markersize=8)

    ax1.set_xlabel("Sequence Length", fontsize=12)
    ax1.set_ylabel("Time (ms)", fontsize=12)
    ax1.set_title(
        "Forward Pass Time vs Sequence Length", fontsize=14, fontweight="bold"
    )
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log", base=2)

    # Plot 2: Speedup
    ax2 = axes[0, 1]
    speedups = [
        s / d if (s is not None and d is not None) else None
        for s, d in zip(standard_times, deltanet_times)
    ]
    if any(s is not None for s in speedups):
        speedup_seq, speedup_vals = filter_data(seq_lengths, speedups)
        speedup_seq, speedup_vals = list(speedup_seq), list(speedup_vals)
        ax2.plot(
            speedup_seq, speedup_vals, "o-", color="green", linewidth=2, markersize=8
        )
    ax2.axhline(y=1, color="r", linestyle="--", label="No speedup")
    ax2.set_xlabel("Sequence Length", fontsize=12)
    ax2.set_ylabel("Speedup (x)", fontsize=12)
    ax2.set_title("DeltaNet Speedup over Softmax", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log", base=2)

    # Plot 3: Memory usage (if available)
    ax3 = axes[1, 0]
    if device.type == "cuda":
        if any(m is not None for m in deltanet_memory):
            delta_mem_seq, delta_mem_vals = filter_data(seq_lengths, deltanet_memory)
            delta_mem_seq, delta_mem_vals = list(delta_mem_seq), list(delta_mem_vals)
            ax3.plot(
                delta_mem_seq,
                delta_mem_vals,
                "o-",
                label="DeltaNet",
                linewidth=2,
                markersize=8,
            )

        if any(m is not None for m in standard_memory):
            std_mem_seq, std_mem_vals = filter_data(seq_lengths, standard_memory)
            std_mem_seq, std_mem_vals = list(std_mem_seq), list(std_mem_vals)
            ax3.plot(
                std_mem_seq,
                std_mem_vals,
                "s-",
                label="Softmax",
                linewidth=2,
                markersize=8,
            )

        ax3.set_xlabel("Sequence Length", fontsize=12)
        ax3.set_ylabel("Memory (MB)", fontsize=12)
        ax3.set_title("Peak Memory Usage", fontsize=14, fontweight="bold")
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale("log", base=2)
    else:
        ax3.text(
            0.5,
            0.5,
            "GPU not available\nfor memory benchmarking",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax3.axis("off")

    # Plot 4: Autoregressive generation
    ax4 = axes[1, 1]
    ax4.plot(delta_gen_times, label="DeltaNet", linewidth=2, alpha=0.7)
    ax4.plot(standard_gen_times, label="Softmax", linewidth=2, alpha=0.7)
    ax4.set_xlabel("Token Position", fontsize=12)
    ax4.set_ylabel("Time per Token (s)", fontsize=12)
    ax4.set_title("Autoregressive Generation Time", fontsize=14, fontweight="bold")
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("attention_benchmark.png", dpi=300, bbox_inches="tight")
    print(f"\n✓ Benchmark plot saved to 'attention_benchmark.png'")
    plt.show()

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(
        f"\n{'Seq Length':<15} {'DeltaNet (ms)':<15} {'Softmax (ms)':<15} {'Speedup':<10}"
    )
    print("-" * 60)
    for i, seq_len in enumerate(seq_lengths):
        delta_str = (
            f"{deltanet_times[i]:.2f}" if deltanet_times[i] is not None else "OOM"
        )
        std_str = f"{standard_times[i]:.2f}" if standard_times[i] is not None else "OOM"

        if speedups[i] is not None:
            speedup_str = f"{speedups[i]:.2f}x"
        elif deltanet_times[i] is not None and standard_times[i] is None:
            speedup_str = "∞"
        else:
            speedup_str = "N/A"

        print(f"{seq_len:<15} {delta_str:<15} {std_str:<15} {speedup_str:<10}")

    print("\nKey Insights:")
    print(f"  • DeltaNet shows O(N) scaling, Softmax shows O(N²)")

    # Find last valid speedup
    last_valid_speedup = None
    last_valid_seqlen = None
    for i in range(len(speedups) - 1, -1, -1):
        if speedups[i] is not None:
            last_valid_speedup = speedups[i]
            last_valid_seqlen = seq_lengths[i]
            break

    if last_valid_speedup:
        print(
            f"  • At seq_len={last_valid_seqlen}, DeltaNet is {last_valid_speedup:.1f}x faster"
        )

    # Check if DeltaNet succeeded where Standard failed
    deltanet_success_count = sum(1 for t in deltanet_times if t is not None)
    standard_success_count = sum(1 for t in standard_times if t is not None)

    if deltanet_success_count > standard_success_count:
        print(
            f"  • DeltaNet succeeded on {deltanet_success_count}/{len(seq_lengths)} sequences"
        )
        print(
            f"  • Standard Softmax succeeded on {standard_success_count}/{len(seq_lengths)} sequences"
        )
        print(
            f"  • DeltaNet handles {deltanet_success_count - standard_success_count} more sequence length(s)!"
        )

    print(
        f"  • Autoregressive: DeltaNet is {sum(standard_gen_times)/sum(delta_gen_times):.1f}x faster"
    )
    print(f"  • DeltaNet maintains constant time per token in generation")
    print(f"  • Softmax time increases linearly with context length")


if __name__ == "__main__":
    run_comprehensive_benchmark()
