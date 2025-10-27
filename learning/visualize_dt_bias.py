#!/usr/bin/env python3
"""
Visualize why we need BOTH alpha (a) AND dt_bias
Shows they serve different purposes: global baseline vs local adjustment
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


def visualize_dt_computation():
    """Show how dt_bias and alpha work together"""

    print("=" * 70)
    print("WHY BOTH alpha AND dt_bias?")
    print("=" * 70)

    # Simulate 3 heads with different dt_bias
    dt_bias_values = [0.5, 1.0, 2.0]
    head_names = ["Head 0 (small bias)", "Head 1 (medium bias)", "Head 2 (large bias)"]
    colors = ["blue", "green", "red"]

    # Alpha varies by position (input-dependent)
    seq_len = 20
    alpha = (
        np.sin(np.linspace(0, 4 * np.pi, seq_len)) * 0.5
    )  # Oscillating between -0.5 and 0.5

    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Plot 1: Alpha (same for all heads)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(
        alpha,
        "o-",
        color="purple",
        linewidth=2,
        markersize=6,
        label="alpha (input-dependent)",
    )
    ax1.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    ax1.set_xlabel("Position (token index)", fontsize=12)
    ax1.set_ylabel("Alpha value", fontsize=12)
    ax1.set_title(
        "Alpha: Input-Dependent (SAME for all heads)", fontsize=14, fontweight="bold"
    )
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.text(
        0.02,
        0.98,
        "Computed from input\nVaries by position",
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.3),
    )

    # Plot 2: dt_bias (different per head)
    ax2 = fig.add_subplot(gs[1, 0])
    for i, (dt_bias, name, color) in enumerate(zip(dt_bias_values, head_names, colors)):
        ax2.bar(i, dt_bias, color=color, alpha=0.7, label=name)
    ax2.set_ylabel("dt_bias value", fontsize=12)
    ax2.set_title(
        "dt_bias: Per-Head Baseline (DIFFERENT per head)",
        fontsize=13,
        fontweight="bold",
    )
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(["Head 0", "Head 1", "Head 2"])
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.text(
        0.02,
        0.98,
        "Learned parameter\nConstant per head",
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.3),
    )

    # Plot 3: Resulting dt for each head
    ax3 = fig.add_subplot(gs[1, 1])
    for dt_bias, name, color in zip(dt_bias_values, head_names, colors):
        dt_values = [F.softplus(torch.tensor(a + dt_bias)).item() for a in alpha]
        ax3.plot(
            dt_values,
            "o-",
            color=color,
            linewidth=2,
            markersize=4,
            label=name,
            alpha=0.7,
        )

    ax3.set_xlabel("Position", fontsize=12)
    ax3.set_ylabel("dt = softplus(α + dt_bias)", fontsize=12)
    ax3.set_title(
        "Final dt: Different Baselines + Same Modulation",
        fontsize=13,
        fontweight="bold",
    )
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.text(
        0.02,
        0.98,
        "Each head has different\nbaseline but responds\nto same input signal",
        transform=ax3.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="orange", alpha=0.3),
    )

    # Plot 4: Comparison at one position
    ax4 = fig.add_subplot(gs[2, :])
    pos = 10
    alpha_val = alpha[pos]

    x_pos = np.arange(len(dt_bias_values))
    dt_without_bias = [F.softplus(torch.tensor(alpha_val)).item()] * 3
    dt_with_bias = [
        F.softplus(torch.tensor(alpha_val + dt_bias)).item()
        for dt_bias in dt_bias_values
    ]

    width = 0.35
    ax4.bar(
        x_pos - width / 2,
        dt_without_bias,
        width,
        label="Without dt_bias (all same)",
        alpha=0.7,
        color="gray",
    )
    ax4.bar(
        x_pos + width / 2,
        dt_with_bias,
        width,
        label="With dt_bias (different)",
        alpha=0.7,
        color=["blue", "green", "red"],
    )

    ax4.set_ylabel(f"dt value at position {pos}", fontsize=12)
    ax4.set_title(
        f"Effect of dt_bias at Position {pos} (α={alpha_val:.2f})",
        fontsize=14,
        fontweight="bold",
    )
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(["Head 0", "Head 1", "Head 2"])
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3, axis="y")

    # Add annotations
    for i, (val, dt_bias) in enumerate(zip(dt_with_bias, dt_bias_values)):
        ax4.text(
            i, val + 0.05, f"{val:.2f}", ha="center", fontsize=10, fontweight="bold"
        )

    plt.suptitle(
        "dt = softplus(α + dt_bias): Global Baseline + Local Adjustment",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    plt.savefig("dt_bias_visualization.png", dpi=300, bbox_inches="tight")
    print("\n✓ Saved to 'dt_bias_visualization.png'")
    plt.show()

    # Print breakdown
    print("\n" + "=" * 70)
    print("BREAKDOWN AT POSITION", pos)
    print("=" * 70)
    print(f"Alpha at this position: {alpha_val:.3f}")
    print(
        f"\n{'Head':<10} {'dt_bias':<12} {'α + dt_bias':<15} {'dt=softplus(...)':<15}"
    )
    print("-" * 70)
    for i, dt_bias in enumerate(dt_bias_values):
        sum_val = alpha_val + dt_bias
        dt_val = F.softplus(torch.tensor(sum_val)).item()
        print(f"Head {i:<5} {dt_bias:<12.1f} {sum_val:<15.3f} {dt_val:<15.3f}")

    print("\nKey Insight:")
    print("  • Same alpha (input) for all heads")
    print("  • Different dt_bias per head")
    print("  • Result: Different dt values!")
    print("  • dt_bias sets the BASELINE for each head")


def show_without_vs_with_bias():
    """Compare behavior without and with dt_bias"""

    print("\n" + "=" * 70)
    print("WITHOUT vs WITH dt_bias")
    print("=" * 70)

    alpha_range = np.linspace(-2, 2, 100)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Without dt_bias (all heads same)
    ax1 = axes[0]
    dt_no_bias = [F.softplus(torch.tensor(a)).item() for a in alpha_range]
    ax1.plot(alpha_range, dt_no_bias, linewidth=3, color="gray")
    ax1.fill_between(alpha_range, 0, dt_no_bias, alpha=0.3, color="gray")
    ax1.set_xlabel("Alpha (α)", fontsize=12)
    ax1.set_ylabel("dt = softplus(α)", fontsize=12)
    ax1.set_title(
        "❌ WITHOUT dt_bias\n(All heads have same curve)",
        fontsize=13,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)
    ax1.axhline(
        y=F.softplus(torch.tensor(0.0)).item(),
        color="red",
        linestyle="--",
        alpha=0.5,
        label="dt when α=0",
    )
    ax1.legend(fontsize=10)
    ax1.text(
        0.02,
        0.98,
        "Problem: All heads\nstart at same baseline!",
        transform=ax1.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="red", alpha=0.2),
    )

    # With dt_bias (heads can differ)
    ax2 = axes[1]
    dt_bias_values = [0.0, 1.0, 2.0]
    colors = ["blue", "green", "red"]

    for dt_bias, color in zip(dt_bias_values, colors):
        dt_with_bias = [
            F.softplus(torch.tensor(a + dt_bias)).item() for a in alpha_range
        ]
        ax2.plot(
            alpha_range,
            dt_with_bias,
            linewidth=2,
            color=color,
            label=f"dt_bias={dt_bias}",
            alpha=0.7,
        )
        ax2.fill_between(alpha_range, 0, dt_with_bias, alpha=0.1, color=color)

    ax2.set_xlabel("Alpha (α)", fontsize=12)
    ax2.set_ylabel("dt = softplus(α + dt_bias)", fontsize=12)
    ax2.set_title(
        "✓ WITH dt_bias\n(Each head has different baseline)",
        fontsize=13,
        fontweight="bold",
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.text(
        0.02,
        0.98,
        "Solution: Each head can\nlearn its own baseline!",
        transform=ax2.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="green", alpha=0.2),
    )

    plt.tight_layout()
    plt.savefig("dt_bias_comparison.png", dpi=300, bbox_inches="tight")
    print("\n✓ Saved to 'dt_bias_comparison.png'")
    plt.show()


def show_analogy():
    """Visual analogy to clarify the roles"""

    print("\n" + "=" * 70)
    print("ANALOGY: Thermostat")
    print("=" * 70)

    print(
        """
    Think of temperature control in different rooms:
    
    🏠 Room 1 (Head 0):
       • Thermostat setting: 68°F   ← dt_bias = 0.5
       • Window adjustment: ±5°F    ← alpha varies
       • Actual temp: 63-73°F       ← dt varies around 68°F
    
    🏠 Room 2 (Head 1):
       • Thermostat setting: 72°F   ← dt_bias = 1.0
       • Window adjustment: ±5°F    ← alpha varies (same as Room 1!)
       • Actual temp: 67-77°F       ← dt varies around 72°F
    
    🏠 Room 3 (Head 2):
       • Thermostat setting: 78°F   ← dt_bias = 2.0
       • Window adjustment: ±5°F    ← alpha varies (same as others!)
       • Actual temp: 73-83°F       ← dt varies around 78°F
    
    Key Points:
    ✓ Each room has DIFFERENT baseline (thermostat/dt_bias)
    ✓ All rooms respond to SAME weather (window/alpha)
    ✓ Final temperature DIFFERS by room (dt)
    
    Without dt_bias: All rooms forced to same thermostat setting! 
    With dt_bias: Each room can choose its preferred baseline!
    """
    )


if __name__ == "__main__":
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + " " * 20 + "dt_bias vs alpha" + " " * 28 + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)

    try:
        # Main visualization
        visualize_dt_computation()

        # Comparison
        show_without_vs_with_bias()

        # Analogy
        show_analogy()

        print("\n" + "=" * 70)
        print("✅ SUMMARY")
        print("=" * 70)
        print(
            """
        Q: Why both alpha AND dt_bias?
        
        A: They serve DIFFERENT purposes:
        
        alpha (a):
        • Computed from INPUT
        • VARIES by position (token-dependent)
        • Same across heads (shared signal)
        • Role: "Adjust for THIS specific token"
        
        dt_bias:
        • LEARNED parameter
        • CONSTANT per head (but learned during training)
        • Different across heads
        • Role: "This head's DEFAULT baseline"
        
        Together:
        dt = softplus(alpha + dt_bias)
                     ↑        ↑
                     |        └── Per-head learned baseline
                     └─────────── Per-token input adjustment
        
        Like: Thermostat (dt_bias) + Window (alpha) = Temperature (dt)
        
        This allows each head to:
        1. Have its own baseline time step (via dt_bias)
        2. Adjust based on input (via alpha)
        3. Specialize to different temporal scales!
        """
        )

        print("\n✓ Visualizations saved:")
        print("  📊 dt_bias_visualization.png")
        print("  📊 dt_bias_comparison.png")
        print("\n✓ Read more:")
        print("  📚 DT_BIAS_EXPLAINED.md")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
