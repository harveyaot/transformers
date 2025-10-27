#!/usr/bin/env python3
"""
Demonstrate WHY we use log-parameterization (A_log) instead of storing A directly
This shows it's NOT duplicated - A_log changes during training!
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


def demonstrate_log_parameterization():
    """Show that A_log changes during training, so exp() is NOT redundant"""

    print("=" * 70)
    print("LOG-PARAMETERIZATION DEMO: Why A_log → exp(A_log)?")
    print("=" * 70)

    # Simulate training process
    num_iterations = 100

    # Method 1: Store A_log (CORRECT - what DeltaNet does)
    A_init = torch.tensor([5.0, 10.0])
    A_log = nn.Parameter(torch.log(A_init))  # [log(5), log(10)]

    # Method 2: Store A directly (PROBLEMATIC)
    A_direct = nn.Parameter(A_init.clone())

    # Track evolution
    A_log_history = [A_log.detach().clone()]
    A_from_log_history = [A_log.exp().detach().clone()]
    A_direct_history = [A_direct.detach().clone()]

    print(f"\nInitialization:")
    print(f"  A_log     = {A_log.data}")
    print(f"  exp(A_log) = {A_log.exp().data}")
    print(f"  A_direct  = {A_direct.data}")

    # Simulate training with some dummy gradients
    for i in range(num_iterations):
        # Simulate gradient descent
        # (In real training, these would come from backprop)

        # For log-space: gradient can be anything
        fake_grad_log = torch.randn(2) * 0.05
        A_log.data = A_log.data - fake_grad_log

        # For direct: gradient might push A negative!
        fake_grad_direct = torch.randn(2) * 0.5
        A_direct.data = A_direct.data - fake_grad_direct
        # Need to clamp to keep positive
        A_direct.data = torch.clamp(A_direct.data, min=0.1)

        # Record
        A_log_history.append(A_log.detach().clone())
        A_from_log_history.append(A_log.exp().detach().clone())
        A_direct_history.append(A_direct.detach().clone())

    print(f"\nAfter {num_iterations} iterations:")
    print(f"  A_log     = {A_log.data}")
    print(f"  exp(A_log) = {A_log.exp().data}  ← DIFFERENT from init!")
    print(f"  A_direct  = {A_direct.data}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: A_log evolution
    ax1 = axes[0, 0]
    A_log_array = torch.stack(A_log_history).numpy()
    ax1.plot(A_log_array[:, 0], label="A_log[0] (Head 0)", linewidth=2)
    ax1.plot(A_log_array[:, 1], label="A_log[1] (Head 1)", linewidth=2)
    ax1.axhline(
        y=np.log(5), color="b", linestyle="--", alpha=0.5, label="Initial log(5)"
    )
    ax1.axhline(
        y=np.log(10), color="orange", linestyle="--", alpha=0.5, label="Initial log(10)"
    )
    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel("A_log value", fontsize=12)
    ax1.set_title("A_log Changes During Training!", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: exp(A_log) evolution
    ax2 = axes[0, 1]
    A_from_log_array = torch.stack(A_from_log_history).numpy()
    ax2.plot(A_from_log_array[:, 0], label="exp(A_log)[0]", linewidth=2)
    ax2.plot(A_from_log_array[:, 1], label="exp(A_log)[1]", linewidth=2)
    ax2.axhline(y=5, color="b", linestyle="--", alpha=0.5, label="Initial A=5")
    ax2.axhline(y=10, color="orange", linestyle="--", alpha=0.5, label="Initial A=10")
    ax2.set_xlabel("Iteration", fontsize=12)
    ax2.set_ylabel("A = exp(A_log)", fontsize=12)
    ax2.set_title(
        "A Values After exp() - Also Changes!", fontsize=14, fontweight="bold"
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: A_direct evolution (with clamping issues)
    ax3 = axes[1, 0]
    A_direct_array = torch.stack(A_direct_history).numpy()
    ax3.plot(A_direct_array[:, 0], label="A_direct[0]", linewidth=2, color="red")
    ax3.plot(A_direct_array[:, 1], label="A_direct[1]", linewidth=2, color="darkred")
    ax3.axhline(y=5, color="r", linestyle="--", alpha=0.5, label="Initial A=5")
    ax3.axhline(y=10, color="darkred", linestyle="--", alpha=0.5, label="Initial A=10")
    ax3.axhline(y=0.1, color="black", linestyle=":", alpha=0.5, label="Clamp boundary")
    ax3.set_xlabel("Iteration", fontsize=12)
    ax3.set_ylabel("A (direct)", fontsize=12)
    ax3.set_title(
        "Direct A - Needs Clamping (Problematic)", fontsize=14, fontweight="bold"
    )
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Comparison
    ax4 = axes[1, 1]
    ax4.text(
        0.5,
        0.9,
        "Key Insight",
        fontsize=20,
        fontweight="bold",
        ha="center",
        transform=ax4.transAxes,
    )
    ax4.text(
        0.5,
        0.7,
        "A_log is a LEARNABLE parameter",
        fontsize=14,
        ha="center",
        transform=ax4.transAxes,
        color="blue",
    )
    ax4.text(
        0.5,
        0.6,
        "It CHANGES during training!",
        fontsize=14,
        ha="center",
        transform=ax4.transAxes,
        color="blue",
        fontweight="bold",
    )
    ax4.text(
        0.5,
        0.45,
        "Initial: A_log = [1.61, 2.30]",
        fontsize=12,
        ha="center",
        transform=ax4.transAxes,
        family="monospace",
    )
    ax4.text(
        0.5,
        0.35,
        "After training: A_log = [?.??, ?.??]",
        fontsize=12,
        ha="center",
        transform=ax4.transAxes,
        family="monospace",
        color="red",
    )
    ax4.text(
        0.5,
        0.2,
        "exp() is NOT redundant!",
        fontsize=14,
        ha="center",
        transform=ax4.transAxes,
        color="green",
        fontweight="bold",
    )
    ax4.text(
        0.5,
        0.1,
        "It transforms learned A_log to valid A > 0",
        fontsize=11,
        ha="center",
        transform=ax4.transAxes,
        color="green",
    )
    ax4.axis("off")

    plt.tight_layout()
    plt.savefig("log_param_demo.png", dpi=300, bbox_inches="tight")
    print("\n✓ Saved to 'log_param_demo.png'")
    plt.show()


def show_gradient_problem():
    """Demonstrate why direct parameterization causes gradient issues"""

    print("\n" + "=" * 70)
    print("GRADIENT PROBLEM WITH DIRECT PARAMETERIZATION")
    print("=" * 70)

    # Test values
    A_values = [0.01, 0.1, 1.0, 5.0, 10.0, 20.0]

    print("\nCompare gradients for different A values:")
    print(f"{'A':<10} {'Direct ∂L/∂A':<20} {'Log ∂L/∂A_log':<20} {'Ratio':<15}")
    print("-" * 70)

    for A in A_values:
        # Assume ∂L/∂A = 1.0 for simplicity
        grad_direct = 1.0

        # With log parameterization: ∂L/∂A_log = ∂L/∂A * ∂A/∂A_log = ∂L/∂A * A
        grad_log = 1.0 * A

        ratio = grad_log / grad_direct if grad_direct != 0 else float("inf")

        print(f"{A:<10.2f} {grad_direct:<20.4f} {grad_log:<20.4f} {ratio:<15.2f}x")

    print("\nObservation:")
    print("  • Direct: Gradient is same for all A (unscaled)")
    print("  • Log-space: Gradient scales with A (automatic scaling!)")
    print("  • Small A → smaller updates (stable)")
    print("  • Large A → larger updates (efficient)")


def show_constraint_problem():
    """Show why constraining A > 0 is problematic"""

    print("\n" + "=" * 70)
    print("CONSTRAINT PROBLEM: Keeping A > 0")
    print("=" * 70)

    # Simulate gradient descent on A directly
    A = 0.5  # Small positive value
    learning_rate = 0.1

    print(f"\nStarting A = {A}")
    print(f"Learning rate = {learning_rate}")

    for i in range(5):
        # Random gradient (might be positive or negative)
        grad = np.random.randn() * 2
        A_new = A - learning_rate * grad

        print(f"\nStep {i+1}:")
        print(f"  Gradient = {grad:.2f}")
        print(f"  A_new = {A:.2f} - {learning_rate} * {grad:.2f} = {A_new:.2f}")

        if A_new < 0:
            print(f"  ❌ PROBLEM: A became negative! ({A_new:.2f})")
            print(f"  Need to clamp: A = max(0.01, {A_new:.2f}) = 0.01")
            A = 0.01
        else:
            print(f"  ✓ OK: A is still positive")
            A = A_new

    print("\n" + "-" * 70)
    print("With log-parameterization:")
    print("  A_log can be ANY value (even negative)")
    print("  exp(A_log) is ALWAYS positive")
    print("  No clamping needed!")


def visual_summary():
    """Create a visual summary diagram"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Wrong understanding
    ax1 = axes[0]
    ax1.text(
        0.5,
        0.85,
        "❌ WRONG Understanding",
        ha="center",
        fontsize=16,
        fontweight="bold",
        color="red",
        transform=ax1.transAxes,
    )

    ax1.text(
        0.5,
        0.65,
        "A = [5, 10]",
        ha="center",
        fontsize=14,
        family="monospace",
        transform=ax1.transAxes,
    )
    ax1.arrow(
        0.5,
        0.55,
        0,
        -0.08,
        head_width=0.05,
        head_length=0.03,
        transform=ax1.transAxes,
        color="gray",
    )
    ax1.text(
        0.5,
        0.45,
        "A_log = log(A)",
        ha="center",
        fontsize=14,
        family="monospace",
        transform=ax1.transAxes,
    )
    ax1.arrow(
        0.5,
        0.35,
        0,
        -0.08,
        head_width=0.05,
        head_length=0.03,
        transform=ax1.transAxes,
        color="gray",
    )
    ax1.text(
        0.5,
        0.25,
        "A = exp(A_log)",
        ha="center",
        fontsize=14,
        family="monospace",
        transform=ax1.transAxes,
    )
    ax1.text(
        0.5,
        0.15,
        "= [5, 10] again",
        ha="center",
        fontsize=14,
        family="monospace",
        color="red",
        transform=ax1.transAxes,
    )
    ax1.text(
        0.5,
        0.05,
        "Seems redundant! 🤔",
        ha="center",
        fontsize=12,
        color="red",
        transform=ax1.transAxes,
    )
    ax1.axis("off")

    # Right: Correct understanding
    ax2 = axes[1]
    ax2.text(
        0.5,
        0.85,
        "✓ CORRECT Understanding",
        ha="center",
        fontsize=16,
        fontweight="bold",
        color="green",
        transform=ax2.transAxes,
    )

    ax2.text(
        0.5,
        0.65,
        "A_log = [1.61, 2.30] (learnable!)",
        ha="center",
        fontsize=12,
        family="monospace",
        transform=ax2.transAxes,
    )
    ax2.arrow(
        0.5,
        0.55,
        0,
        -0.06,
        head_width=0.05,
        head_length=0.03,
        transform=ax2.transAxes,
        color="blue",
    )
    ax2.text(
        0.3,
        0.50,
        "Training...",
        ha="center",
        fontsize=10,
        color="blue",
        style="italic",
        transform=ax2.transAxes,
    )
    ax2.arrow(
        0.5,
        0.45,
        0,
        -0.06,
        head_width=0.05,
        head_length=0.03,
        transform=ax2.transAxes,
        color="blue",
    )
    ax2.text(
        0.5,
        0.35,
        "A_log = [0.69, 2.77] (changed!)",
        ha="center",
        fontsize=12,
        family="monospace",
        color="blue",
        fontweight="bold",
        transform=ax2.transAxes,
    )
    ax2.arrow(
        0.5,
        0.25,
        0,
        -0.06,
        head_width=0.05,
        head_length=0.03,
        transform=ax2.transAxes,
        color="green",
    )
    ax2.text(
        0.5,
        0.15,
        "A = exp([0.69, 2.77])",
        ha="center",
        fontsize=12,
        family="monospace",
        transform=ax2.transAxes,
    )
    ax2.text(
        0.5,
        0.08,
        "= [2.0, 16.0]",
        ha="center",
        fontsize=12,
        family="monospace",
        color="green",
        transform=ax2.transAxes,
    )
    ax2.text(
        0.5,
        0.00,
        "Different from init! ✓",
        ha="center",
        fontsize=12,
        color="green",
        fontweight="bold",
        transform=ax2.transAxes,
    )
    ax2.axis("off")

    plt.tight_layout()
    plt.savefig("log_param_summary.png", dpi=300, bbox_inches="tight")
    print("\n✓ Saved to 'log_param_summary.png'")
    plt.show()


if __name__ == "__main__":
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + " " * 18 + "WHY LOG-PARAMETERIZATION?" + " " * 20 + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)

    try:
        # Main demo
        demonstrate_log_parameterization()

        # Gradient problem
        show_gradient_problem()

        # Constraint problem
        show_constraint_problem()

        # Visual summary
        visual_summary()

        print("\n" + "=" * 70)
        print("✅ CONCLUSION")
        print("=" * 70)
        print(
            """
        Q: Why log(A) then exp() to get A back? Isn't it duplicated?
        
        A: NO! Because A_log CHANGES during training!
        
        Flow:
        1. Init:    A_log = log([5, 10]) = [1.61, 2.30]
        2. Train:   A_log → [0.69, 2.77]  (learned!)
        3. Use:     A = exp([0.69, 2.77]) = [2.0, 16.0]
        
        Benefits:
        ✓ Unconstrained optimization (A_log can be any value)
        ✓ Always positive A (exp always > 0)
        ✓ Better gradients (automatically scaled)
        ✓ Numerical stability
        
        It's NOT redundant - it's a parameterization trick!
        """
        )

        print("\n✓ Visualizations saved:")
        print("  📊 log_param_demo.png")
        print("  📊 log_param_summary.png")
        print("\n✓ Read more:")
        print("  📚 WHY_LOG_SPACE.md")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
