#!/usr/bin/env python3
"""
Visualize how A_log affects the exponential decay in DeltaNet
This helps understand why uniform(0, 16) is chosen
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


def visualize_decay_curves():
    """Show how different A values lead to different memory spans"""
    print("=" * 60)
    print("Visualizing DeltaNet Decay Mechanism")
    print("=" * 60)

    # Different A values spanning the range
    A_values = [0.1, 1.0, 5.0, 10.0, 16.0]
    colors = ["blue", "green", "orange", "red", "purple"]
    dt = 1.0  # Fixed time step

    steps = 50
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Decay curves (linear scale)
    ax1 = axes[0, 0]
    for A, color in zip(A_values, colors):
        decay_over_time = []
        state = 1.0

        for t in range(steps):
            g = -A * dt
            decay = np.exp(g)
            state = state * decay
            decay_over_time.append(state)

        ax1.plot(decay_over_time, label=f"A = {A}", color=color, linewidth=2)

    ax1.set_xlabel("Time Steps", fontsize=12)
    ax1.set_ylabel("Remaining State Value", fontsize=12)
    ax1.set_title("Exponential Decay (Linear Scale)", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)

    # Plot 2: Decay curves (log scale)
    ax2 = axes[0, 1]
    for A, color in zip(A_values, colors):
        decay_over_time = []
        state = 1.0

        for t in range(steps):
            g = -A * dt
            decay = np.exp(g)
            state = state * decay
            decay_over_time.append(state)

        ax2.plot(decay_over_time, label=f"A = {A}", color=color, linewidth=2)

    ax2.set_xlabel("Time Steps", fontsize=12)
    ax2.set_ylabel("Remaining State Value (log scale)", fontsize=12)
    ax2.set_title("Exponential Decay (Log Scale)", fontsize=14, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    # Plot 3: Per-step decay factor
    ax3 = axes[1, 0]
    A_range = np.linspace(0, 16, 100)
    dt = 1.0
    decay_factors = np.exp(-A_range * dt)

    ax3.plot(A_range, decay_factors, linewidth=2, color="darkblue")
    ax3.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="50% retention")
    ax3.axhline(y=0.1, color="orange", linestyle="--", alpha=0.5, label="10% retention")

    # Mark specific A values
    for A in A_values:
        decay = np.exp(-A * dt)
        ax3.plot(A, decay, "o", markersize=8, label=f"A={A}: {decay:.4f}")

    ax3.set_xlabel("A Value", fontsize=12)
    ax3.set_ylabel("Per-Step Decay Factor exp(-A*dt)", fontsize=12)
    ax3.set_title("Decay Factor vs A Value", fontsize=14, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Half-life (steps to decay to 50%)
    ax4 = axes[1, 1]
    A_range = np.linspace(0.1, 16, 100)
    half_lives = np.log(2) / (A_range * dt)  # Steps to reach 50%

    ax4.plot(A_range, half_lives, linewidth=2, color="darkgreen")

    # Mark specific A values
    for A in [0.1, 1.0, 5.0, 10.0, 16.0]:
        hl = np.log(2) / (A * dt)
        ax4.plot(A, hl, "o", markersize=8, label=f"A={A}: {hl:.1f} steps")

    ax4.set_xlabel("A Value", fontsize=12)
    ax4.set_ylabel("Half-Life (steps)", fontsize=12)
    ax4.set_title("Memory Span: Steps to 50% Decay", fontsize=14, fontweight="bold")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale("log")

    plt.tight_layout()
    plt.savefig("decay_visualization.png", dpi=300, bbox_inches="tight")
    print("\n✓ Visualization saved to 'decay_visualization.png'")
    plt.show()

    # Print summary
    print("\n" + "=" * 60)
    print("DECAY CHARACTERISTICS")
    print("=" * 60)
    print(f"\n{'A Value':<10} {'Decay/Step':<15} {'Half-Life':<15} {'Memory Span'}")
    print("-" * 60)

    for A in [0.1, 1.0, 2.0, 5.0, 8.0, 10.0, 16.0]:
        decay_per_step = np.exp(-A * dt)
        half_life = np.log(2) / (A * dt)

        if half_life > 100:
            span = "Very long"
        elif half_life > 20:
            span = "Long"
        elif half_life > 5:
            span = "Medium"
        elif half_life > 1:
            span = "Short"
        else:
            span = "Very short"

        print(f"{A:<10.1f} {decay_per_step:<15.6f} {half_life:<15.2f} {span}")

    print("\nKey Insights:")
    print("  • A = 0.1:  Half-life = 6.9 steps  → Long-term memory")
    print("  • A = 1.0:  Half-life = 0.7 steps  → Medium-term")
    print("  • A = 5.0:  Half-life = 0.14 steps → Short-term")
    print("  • A = 16.0: Half-life = 0.04 steps → Immediate context only")


def demonstrate_input_modulation():
    """Show how alpha (input-dependent) modulates the base A"""
    print("\n" + "=" * 60)
    print("Demonstrating Input-Dependent Modulation")
    print("=" * 60)

    # Fixed base decay rate
    A = 5.0
    dt_bias = 1.0

    # Different input-dependent alpha values
    alpha_values = [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0]

    print(f"\nBase A = {A}")
    print(f"dt_bias = {dt_bias}")
    print("\nHow input alpha modulates decay:")
    print(f"{'Alpha':<10} {'dt':<10} {'g':<12} {'exp(g)':<12} {'Interpretation'}")
    print("-" * 70)

    for alpha in alpha_values:
        dt = F.softplus(torch.tensor(alpha + dt_bias)).item()
        g = -A * dt
        decay = np.exp(g)

        if decay > 0.5:
            interp = "Weak decay (remember more)"
        elif decay > 0.1:
            interp = "Medium decay"
        else:
            interp = "Strong decay (forget more)"

        print(f"{alpha:<10.1f} {dt:<10.3f} {g:<12.3f} {decay:<12.6f} {interp}")

    print("\nConclusion:")
    print("  • Negative alpha → smaller dt → less decay → remember more")
    print("  • Positive alpha → larger dt → more decay → forget more")
    print("  • Model learns to adjust alpha based on input content!")


def show_multi_head_example():
    """Demonstrate how different heads can have different A values"""
    print("\n" + "=" * 60)
    print("Multi-Head Decay Example")
    print("=" * 60)

    # Simulate 8 heads with different A values
    num_heads = 8
    A_values = torch.tensor([0.5, 2.0, 5.0, 8.0, 10.0, 12.0, 15.0, 16.0])
    dt = 1.0

    print(f"\nSimulating {num_heads} attention heads processing a sequence:")
    print(f"\n{'Head':<8} {'A Value':<10} {'Decay Rate':<15} {'Specialization'}")
    print("-" * 60)

    specializations = [
        "Document structure",
        "Paragraph coherence",
        "Sentence context",
        "Phrase patterns",
        "Word relationships",
        "Bigram patterns",
        "Adjacent tokens",
        "Current token focus",
    ]

    for i, (A, spec) in enumerate(zip(A_values, specializations)):
        decay = np.exp(-A * dt)
        print(f"{i:<8} {A:<10.1f} {decay:<15.6f} {spec}")

    print("\nThis multi-scale processing is automatic!")
    print("Each head learns its optimal A during training.")

    # Visualize state evolution for different heads
    fig, ax = plt.subplots(figsize=(12, 6))
    steps = 20

    for i, A in enumerate(A_values):
        states = []
        state = 1.0
        for t in range(steps):
            g = -A * dt
            state = state * np.exp(g)
            states.append(state)

        ax.plot(states, label=f"Head {i} (A={A:.1f})", linewidth=2)

    ax.set_xlabel("Time Steps", fontsize=12)
    ax.set_ylabel("State Retention", fontsize=12)
    ax.set_title("Multi-Head State Evolution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig("multihead_decay.png", dpi=300, bbox_inches="tight")
    print("\n✓ Multi-head visualization saved to 'multihead_decay.png'")
    plt.show()


if __name__ == "__main__":
    print("\n" + "█" * 60)
    print("█" + " " * 58 + "█")
    print("█" + " " * 15 + "A_log VISUALIZATION" + " " * 24 + "█")
    print("█" + " " * 12 + "Understanding DeltaNet Decay" + " " * 19 + "█")
    print("█" + " " * 58 + "█")
    print("█" * 60)

    try:
        # Main visualization
        visualize_decay_curves()

        # Input modulation demo
        demonstrate_input_modulation()

        # Multi-head example
        show_multi_head_example()

        print("\n" + "=" * 60)
        print("✅ All visualizations complete!")
        print("=" * 60)
        print("\nGenerated files:")
        print("  📊 decay_visualization.png")
        print("  📊 multihead_decay.png")
        print("\nFor detailed explanation, see:")
        print("  📚 UNDERSTANDING_A_LOG.md")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
