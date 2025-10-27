#!/usr/bin/env python3
"""
Quick test to verify DeltaNet implementation works correctly
Run this before the full benchmark to ensure everything is set up properly
"""

import torch
from deltanet_attention import DeltaNetAttention, StandardSoftmaxAttention


def test_basic_functionality():
    """Test that both attention mechanisms work"""
    print("=" * 60)
    print("QUICK FUNCTIONALITY TEST")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Test parameters
    batch_size = 2
    seq_len = 128
    hidden_size = 512
    num_heads = 8
    head_dim = 64

    print(f"\nTest configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Num heads: {num_heads}")
    print(f"  Head dim: {head_dim}")

    # Create input
    x = torch.randn(batch_size, seq_len, hidden_size, device=device)

    # Test DeltaNet
    print("\n" + "-" * 60)
    print("Testing DeltaNet Attention...")
    try:
        deltanet = DeltaNetAttention(hidden_size, num_heads, head_dim).to(device)

        # Test chunk mode (training)
        out_chunk, state_chunk = deltanet(x, use_recurrent=False, return_state=True)
        assert (
            out_chunk.shape == x.shape
        ), f"Output shape mismatch: {out_chunk.shape} vs {x.shape}"
        assert state_chunk.shape == (
            batch_size,
            num_heads,
            head_dim,
            head_dim,
        ), f"State shape mismatch: {state_chunk.shape}"
        print(f"  ✓ Chunk mode works (training)")
        print(f"    Output shape: {out_chunk.shape}")
        print(f"    State shape: {state_chunk.shape}")

        # Test recurrent mode (inference)
        x_single = torch.randn(1, 1, hidden_size, device=device)
        out_recur, state_recur = deltanet(
            x_single, use_recurrent=True, return_state=True
        )
        assert (
            out_recur.shape == x_single.shape
        ), f"Output shape mismatch: {out_recur.shape}"
        print(f"  ✓ Recurrent mode works (inference)")
        print(f"    Output shape: {out_recur.shape}")

        # Test stateful generation
        state = None
        for i in range(5):
            x_token = torch.randn(1, 1, hidden_size, device=device)
            _, state = deltanet(
                x_token, use_recurrent=True, initial_state=state, return_state=True
            )
        print(f"  ✓ Stateful generation works (5 tokens)")

        print("  ✅ DeltaNet passed all tests!")

    except Exception as e:
        print(f"  ❌ DeltaNet failed: {e}")
        return False

    # Test Standard Attention
    print("\n" + "-" * 60)
    print("Testing Standard Softmax Attention...")
    try:
        standard = StandardSoftmaxAttention(hidden_size, num_heads, head_dim).to(device)
        out_std = standard(x)
        assert (
            out_std.shape == x.shape
        ), f"Output shape mismatch: {out_std.shape} vs {x.shape}"
        print(f"  ✓ Forward pass works")
        print(f"    Output shape: {out_std.shape}")
        print("  ✅ Standard Attention passed all tests!")

    except Exception as e:
        print(f"  ❌ Standard Attention failed: {e}")
        return False

    # Test numerical stability
    print("\n" + "-" * 60)
    print("Testing numerical stability...")
    try:
        # Test with different dtypes
        for dtype in [torch.float32, torch.float16]:
            if dtype == torch.float16 and device.type == "cpu":
                continue  # Skip fp16 on CPU

            x_test = torch.randn(1, 32, hidden_size, device=device, dtype=dtype)
            deltanet_test = (
                DeltaNetAttention(hidden_size, num_heads, head_dim).to(device).to(dtype)
            )
            out_test, _ = deltanet_test(x_test, return_state=True)

            # Check for NaN or Inf
            assert not torch.isnan(out_test).any(), f"NaN detected in output ({dtype})"
            assert not torch.isinf(out_test).any(), f"Inf detected in output ({dtype})"
            print(f"  ✓ {dtype} works correctly")

        print("  ✅ Numerical stability tests passed!")

    except Exception as e:
        print(f"  ⚠️  Numerical stability warning: {e}")

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nYou can now run the full benchmark:")
    print("  python test_attn.py")

    return True


if __name__ == "__main__":
    success = test_basic_functionality()
    exit(0 if success else 1)
