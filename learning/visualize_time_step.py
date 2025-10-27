#!/usr/bin/env python3
"""
Visualize why dt_bias and A_log are called "time step parameters"
even though they don't have a time dimension
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


def visualize_time_step_computation():
    """Show how static parameters create time-varying dt"""

    print("=" * 70)
    print("UNDERSTANDING 'TIME STEP PARAMETERS'")
    print("=" * 70)

    # Simulate parameters for one head
    dt_bias = 1.0  # Static parameter
    A = 5.0  # Static parameter

    # Simulate input-dependent alpha over sequence
    seq_len = 20
    # Alpha varies based on "input" - simulating different patterns
    alpha = np.array(
        [
            0.2,
            0.5,
            -0.3,
            1.0,
            0.8,  # Some variation
            -0.5,
            0.0,
            1.5,
            0.3,
            -0.2,
            0.7,
            1.2,
            -0.8,
            0.4,
            0.9,
            -0.4,
            0.6,
            0.1,
            1.3,
            -0.1,
        ]
    )

    # Compute time-varying dt at each position
    dt = np.array([F.softplus(torch.tensor(a + dt_bias)).item() for a in alpha])

    # Compute time-varying decay factor g
    g = -A * dt
    decay = np.exp(g)

    # Create visualization
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))

    # Plot 1: Alpha (input-dependent, time-varying)
    ax1 = axes[0, 0]
    ax1.bar(range(seq_len), alpha, color="steelblue", alpha=0.7)
    ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax1.set_xlabel("Position (time)", fontsize=12)
    ax1.set_ylabel("Alpha (α)", fontsize=12)
    ax1.set_title("α: Input-Dependent (TIME-VARYING)", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.text(
        0.02,
        0.98,
        "Computed from input at each position",
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.3),
    )

    # Plot 2: dt_bias (static parameter)
    ax2 = axes[0, 1]
    ax2.bar(range(seq_len), [dt_bias] * seq_len, color="orange", alpha=0.7)
    ax2.set_xlabel("Position (time)", fontsize=12)
    ax2.set_ylabel("dt_bias", fontsize=12)
    ax2.set_title(
        "dt_bias: Static Parameter (CONSTANT)", fontsize=13, fontweight="bold"
    )
    ax2.set_ylim(0, 2)
    ax2.grid(True, alpha=0.3)
    ax2.text(
        0.02,
        0.98,
        f"Fixed value = {dt_bias} for all positions",
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.3),
    )

    # Plot 3: dt (computed, time-varying)
    ax3 = axes[1, 0]
    ax3.bar(range(seq_len), dt, color="purple", alpha=0.7)
    ax3.set_xlabel("Position (time)", fontsize=12)
    ax3.set_ylabel("dt = softplus(α + dt_bias)", fontsize=12)
    ax3.set_title("dt: THE TIME STEP (TIME-VARYING!)", fontsize=13, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    ax3.text(
        0.02,
        0.98,
        'dt changes at every position!\nThis is why they\'re "time step params"',
        transform=ax3.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="pink", alpha=0.3),
    )

    # Plot 4: A (static parameter)
    ax4 = axes[1, 1]
    ax4.bar(range(seq_len), [A] * seq_len, color="green", alpha=0.7)
    ax4.set_xlabel("Position (time)", fontsize=12)
    ax4.set_ylabel("A = exp(A_log)", fontsize=12)
    ax4.set_title("A: Base Decay Rate (CONSTANT)", fontsize=13, fontweight="bold")
    ax4.set_ylim(0, 8)
    ax4.grid(True, alpha=0.3)
    ax4.text(
        0.02,
        0.98,
        f"Fixed value = {A} for all positions",
        transform=ax4.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.3),
    )

    # Plot 5: g = -A * dt (computed, time-varying)
    ax5 = axes[2, 0]
    ax5.bar(range(seq_len), g, color="red", alpha=0.7)
    ax5.set_xlabel("Position (time)", fontsize=12)
    ax5.set_ylabel("g = -A × dt", fontsize=12)
    ax5.set_title("g: Decay Factor (TIME-VARYING)", fontsize=13, fontweight="bold")
    ax5.grid(True, alpha=0.3)
    ax5.text(
        0.02,
        0.02,
        "g varies because dt varies!",
        transform=ax5.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.3),
    )

    # Plot 6: exp(g) (final decay, time-varying)
    ax6 = axes[2, 1]
    ax6.bar(range(seq_len), decay, color="darkred", alpha=0.7)
    ax6.set_xlabel("Position (time)", fontsize=12)
    ax6.set_ylabel("exp(g) = State Retention", fontsize=12)
    ax6.set_title(
        "Final Decay: Applied to State (TIME-VARYING)", fontsize=13, fontweight="bold"
    )
    ax6.grid(True, alpha=0.3)
    ax6.set_yscale("log")
    ax6.text(
        0.02,
        0.98,
        "Different decay at each position!\nAdaptive forgetting based on input",
        transform=ax6.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="orange", alpha=0.3),
    )

    plt.tight_layout()
    plt.savefig("time_step_visualization.png", dpi=300, bbox_inches="tight")
    print("\n✓ Saved visualization to 'time_step_visualization.png'")
    plt.show()

    # Print the data
    print("\n" + "=" * 70)
    print("POSITION-BY-POSITION BREAKDOWN")
    print("=" * 70)
    print(f"\nStatic parameters: dt_bias={dt_bias}, A={A}")
    print(
        f"\n{'Pos':<5} {'α':<8} {'dt_bias':<10} {'dt':<10} {'A':<8} {'g':<12} {'exp(g)':<12}"
    )
    print("-" * 70)

    for i in range(min(10, seq_len)):  # Show first 10
        print(
            f"{i:<5} {alpha[i]:<8.2f} {dt_bias:<10.1f} {dt[i]:<10.3f} "
            f"{A:<8.1f} {g[i]:<12.3f} {decay[i]:<12.6f}"
        )

    print("\nKey Observation:")
    print("  • dt_bias and A are CONSTANT across all positions")
    print("  • α varies based on input")
    print("  • dt = softplus(α + dt_bias) VARIES at each position")
    print("  • g = -A * dt VARIES because dt varies")
    print("  • exp(g) gives position-specific decay!")
    print("\nConclusion:")
    print("  → dt_bias is called a 'time step parameter' because it")
    print("    PARAMETERIZES the time step dt, which IS time-varying!")


def show_comparison():
    """Compare static vs dynamic"""

    print("\n" + "=" * 70)
    print("STATIC PARAMETERS vs DYNAMIC COMPUTATION")
    print("=" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    seq_len = 30

    # Left: What's stored (static)
    ax1 = axes[0]
    ax1.text(
        0.5,
        0.8,
        "STORED PARAMETERS",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        transform=ax1.transAxes,
    )
    ax1.text(
        0.5,
        0.6,
        "Shape: [num_heads]",
        ha="center",
        va="center",
        fontsize=14,
        color="blue",
        transform=ax1.transAxes,
    )
    ax1.text(
        0.5,
        0.4,
        "dt_bias = [1.0, 1.0, ..., 1.0]",
        ha="center",
        va="center",
        fontsize=12,
        family="monospace",
        transform=ax1.transAxes,
    )
    ax1.text(
        0.5,
        0.3,
        "A_log = [log(5), log(5), ...]",
        ha="center",
        va="center",
        fontsize=12,
        family="monospace",
        transform=ax1.transAxes,
    )
    ax1.text(
        0.5,
        0.15,
        "✗ No time dimension",
        ha="center",
        va="center",
        fontsize=14,
        color="red",
        fontweight="bold",
        transform=ax1.transAxes,
    )
    ax1.text(
        0.5,
        0.05,
        "✗ Don't vary by position",
        ha="center",
        va="center",
        fontsize=14,
        color="red",
        fontweight="bold",
        transform=ax1.transAxes,
    )
    ax1.axis("off")
    ax1.set_title("PARAMETERS (Static)", fontsize=16, fontweight="bold", pad=20)

    # Right: What's computed (dynamic)
    ax2 = axes[1]
    ax2.text(
        0.5,
        0.8,
        "COMPUTED VALUES",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        transform=ax2.transAxes,
    )
    ax2.text(
        0.5,
        0.6,
        "Shape: [batch, seq_len, num_heads]",
        ha="center",
        va="center",
        fontsize=14,
        color="blue",
        transform=ax2.transAxes,
    )
    ax2.text(
        0.5,
        0.45,
        "dt[t] = softplus(α[t] + dt_bias)",
        ha="center",
        va="center",
        fontsize=12,
        family="monospace",
        transform=ax2.transAxes,
    )
    ax2.text(
        0.5,
        0.35,
        "g[t] = -A * dt[t]",
        ha="center",
        va="center",
        fontsize=12,
        family="monospace",
        transform=ax2.transAxes,
    )
    ax2.text(
        0.5,
        0.25,
        "decay[t] = exp(g[t])",
        ha="center",
        va="center",
        fontsize=12,
        family="monospace",
        transform=ax2.transAxes,
    )
    ax2.text(
        0.5,
        0.10,
        "✓ Has time dimension",
        ha="center",
        va="center",
        fontsize=14,
        color="green",
        fontweight="bold",
        transform=ax2.transAxes,
    )
    ax2.text(
        0.5,
        0.00,
        "✓ Varies by position t",
        ha="center",
        va="center",
        fontsize=14,
        color="green",
        fontweight="bold",
        transform=ax2.transAxes,
    )
    ax2.axis("off")
    ax2.set_title("TIME STEP (Dynamic)", fontsize=16, fontweight="bold", pad=20)

    # Add arrow between them
    fig.text(0.5, 0.5, "→", fontsize=60, ha="center", va="center", color="purple")
    fig.text(
        0.5,
        0.42,
        "Used to compute",
        fontsize=12,
        ha="center",
        va="center",
        color="purple",
    )

    plt.tight_layout()
    plt.savefig("static_vs_dynamic.png", dpi=300, bbox_inches="tight")
    print("\n✓ Saved comparison to 'static_vs_dynamic.png'")
    plt.show()


def show_analogy():
    """Visual analogy to clarify the concept"""

    print("\n" + "=" * 70)
    print("ANALOGY: Recipe vs Cooking")
    print("=" * 70)

    print(
        """
    Think of it like a recipe for adjusting playback speed:
    
    📝 RECIPE (Static Parameters):
       dt_bias = 1.0     ← "Base speed multiplier" 
       A = 5.0           ← "Slowdown factor"
       
    👨‍🍳 COOKING (Dynamic Process):
       For each video frame (position t):
         1. Check frame content → get α[t]
         2. Calculate speed: dt[t] = softplus(α[t] + 1.0)
         3. Apply slowdown: g[t] = -5.0 * dt[t]
         4. Play at speed: exp(g[t])
    
    The RECIPE doesn't change (static parameters)
    But the SPEED changes every frame (dynamic time step)!
    
    That's why dt_bias and A are "time step parameters" -
    they define HOW to compute the time step, not the time step itself.
    """
    )


if __name__ == "__main__":
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + " " * 18 + "TIME STEP PARAMETERS EXPLAINED" + " " * 20 + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)

    try:
        # Main visualization
        visualize_time_step_computation()

        # Comparison
        show_comparison()

        # Analogy
        show_analogy()

        print("\n" + "=" * 70)
        print("✅ SUMMARY")
        print("=" * 70)
        print(
            """
        Q: Why are dt_bias and A_log called "time step parameters"
           if they don't have a time dimension?
        
        A: Because they PARAMETERIZE the computation of dt,
           which IS time-varying!
        
        Static Parameters     →    Dynamic Computation
        ----------------           -------------------
        dt_bias [H]          →     dt[B,T,H] = softplus(α[B,T,H] + dt_bias[H])
        A_log [H]            →     g[B,T,H] = -exp(A_log[H]) * dt[B,T,H]
        
        The parameters themselves are static,
        but they create time-varying behavior through α (input-dependent).
        
        Think: "Parameters FOR the time step" not "Parameters OF the time step"
        """
        )

        print("\n✓ Visualizations saved:")
        print("  📊 time_step_visualization.png")
        print("  📊 static_vs_dynamic.png")
        print("\n✓ Read more:")
        print("  📚 TIME_STEP_EXPLANATION.md")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
