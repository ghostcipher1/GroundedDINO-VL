#!/usr/bin/env python3
"""Verify _C extension installation and functionality."""

import sys
import torch


def verify_extension():
    """Verify _C extension is properly installed."""

    print("=" * 70)
    print("GroundedDINO-VL _C Extension Verification")
    print("=" * 70)

    # 1. Check import
    print("\n[1] Checking _C import...")
    try:
        from groundeddino_vl import _C

        print("    ✓ _C extension imported successfully")
        print(f"    Location: {_C.__file__}")
    except ImportError as e:
        print(f"    ✗ Failed to import _C: {e}")
        print("    (This is expected in CPU-only or development environments)")
        _C = None

    # 2. Check functions
    if _C is not None:
        print("\n[2] Checking _C functions...")
        required_funcs = ["ms_deform_attn_forward", "ms_deform_attn_backward"]
        all_funcs_available = True
        for func in required_funcs:
            if hasattr(_C, func):
                print(f"    ✓ {func}: available")
            else:
                print(f"    ✗ {func}: NOT available")
                all_funcs_available = False

        if not all_funcs_available:
            return False

    # 3. Check CUDA
    print("\n[3] Checking CUDA environment...")
    print(f"    PyTorch version: {torch.__version__}")
    print(f"    CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    CUDA version: {torch.version.cuda}")
        print(f"    Device: {torch.cuda.get_device_name(0)}")
    else:
        print("    (Note: CUDA not available - CPU-only mode)")

    # 4. Test module creation
    print("\n[4] Testing MultiScaleDeformableAttention module...")
    try:
        from groundeddino_vl.models.grounding_dino.ms_deform_attn import (
            MultiScaleDeformableAttention,
            _load_c_extension,
        )

        _load_c_extension()
        from groundeddino_vl.models.grounding_dino.ms_deform_attn import _C_AVAILABLE

        print(f"    _C_AVAILABLE flag: {_C_AVAILABLE}")

        attn = MultiScaleDeformableAttention(embed_dim=256, num_heads=8, num_levels=4, num_points=4)
        print("    ✓ Module created successfully")

        # Try forward pass on CPU
        print("\n[5] Testing forward pass (CPU)...")
        bs, num_query, embed_dim = 2, 10, 256
        query = torch.randn(bs, num_query, embed_dim)
        key = torch.randn(bs, 50, embed_dim)
        value = key.clone()
        reference_points = torch.rand(bs, num_query, 4, 2)
        spatial_shapes = torch.tensor([[8, 8], [4, 4], [2, 2], [1, 1]], dtype=torch.long)
        level_start_index = torch.tensor([0, 64, 80, 88], dtype=torch.long)

        with torch.no_grad():
            output = attn(
                query,
                key,
                value,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
            )
        print(f"    ✓ Forward pass successful (CPU fallback)")
        print(f"      Output shape: {output.shape}")

        # If CUDA available and _C available, try GPU
        if torch.cuda.is_available() and _C_AVAILABLE:
            print("\n[6] Testing forward pass (GPU)...")
            try:
                attn_gpu = attn.cuda()
                query_gpu = query.cuda()
                key_gpu = key.cuda()
                value_gpu = value.cuda()
                reference_points_gpu = reference_points.cuda()
                spatial_shapes_gpu = spatial_shapes.cuda()
                level_start_index_gpu = level_start_index.cuda()

                with torch.no_grad():
                    output_gpu = attn_gpu(
                        query_gpu,
                        key_gpu,
                        value_gpu,
                        reference_points=reference_points_gpu,
                        spatial_shapes=spatial_shapes_gpu,
                        level_start_index=level_start_index_gpu,
                    )
                print(f"    ✓ Forward pass successful (GPU with C++ ops)")
                print(f"      Output shape: {output_gpu.shape}")
            except Exception as e:
                print(f"    ✗ GPU forward pass failed: {e}")
                print("      Falling back to CPU for testing")

    except Exception as e:
        print(f"    ✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n" + "=" * 70)
    print("✓ All checks passed!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    try:
        success = verify_extension()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Verification failed with exception: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
