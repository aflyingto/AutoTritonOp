"""
Accuracy Comparison Test for Triton-Ascend Operators
================================================

This script provides accuracy comparison between Triton-Ascend implementations
and PyTorch reference implementations for various operators.
"""

import torch
import torch_npu
import sys
import os

# Add ops directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ops'))

# Import operators
from vector_add import add as triton_add
from softmax import softmax as triton_softmax
from layer_norm import layer_norm as triton_layer_norm
from flash_attention import attention as triton_attention
from matmul import matmul as triton_matmul


def accuracy_comparison(y_cal, y_ref, op_name, dtype_name):
    """
    Accuracy comparison function: select appropriate comparison strategy based on data type.

    Different data type handling strategies:
    - Floating point types (float16/32, bfloat16): use torch.testing.assert_close with relative/absolute error tolerance
    - Integer types (int8/16/32/64): require exact equality (torch.equal)
    - Boolean types (bool): strict comparison on CPU (avoid device differences)
    """
    # Check if output data types match
    assert y_cal.dtype == y_ref.dtype, f"dtype mismatch: {y_cal.dtype} vs {y_ref.dtype}"
    tensor_dtype = y_cal.dtype

    # Move tensors to NPU (assuming tests run on NPU)
    y_cal = y_cal.npu() if not y_cal.is_npu else y_cal
    y_ref = y_ref.npu() if not y_ref.is_npu else y_ref

    try:
        # Select different comparison methods based on data type
        if tensor_dtype == torch.float16:
            # float16 has lower precision, allow slightly larger error
            torch.testing.assert_close(y_ref, y_cal, rtol=1e-3, atol=1e-3, equal_nan=True)
        elif tensor_dtype == torch.bfloat16:
            # bfloat16 has even lower precision, recommend converting to float32 for comparison
            torch.testing.assert_close(
                y_ref.to(torch.float32),
                y_cal.to(torch.float32),
                rtol=1e-3,
                atol=1e-3,
                equal_nan=True
            )
        elif tensor_dtype == torch.float32:
            # float32 has higher precision, use stricter tolerance
            torch.testing.assert_close(y_ref, y_cal, rtol=1e-4, atol=1e-4, equal_nan=True)
        elif tensor_dtype in [torch.int64, torch.int32, torch.int16, torch.int8, torch.uint32]:
            # Integer types should be exactly equal
            assert torch.equal(y_cal, y_ref), f"Integer tensors are not equal for dtype {tensor_dtype}"
        elif tensor_dtype == torch.bool:
            # Boolean types recommend comparison on CPU to avoid device-specific boolean representation differences
            assert torch.equal(y_cal.cpu(), y_ref.cpu()), "Boolean tensors are not equal"
        else:
            raise ValueError(f'Invalid or unsupported tensor dtype: {tensor_dtype}')

        print(f"✓ {op_name} ({dtype_name}): PASSED - Max diff: {torch.max(torch.abs(y_cal - y_ref))}")
        return True
    except Exception as e:
        print(f"✗ {op_name} ({dtype_name}): FAILED - {str(e)}")
        return False


def test_vector_add():
    """Test VectorAdd operator"""
    print("\n=== Testing VectorAdd ===")
    torch.manual_seed(0)
    size = 98432

    # Test different data types
    test_cases = [
        (torch.float32, "float32"),
        (torch.float16, "float16"),
        (torch.bfloat16, "bfloat16"),
    ]

    for dtype, dtype_name in test_cases:
        x = torch.rand(size, device='npu', dtype=dtype)
        y = torch.rand(size, device='npu', dtype=dtype)
        output_torch = x + y
        output_triton = triton_add(x, y)
        accuracy_comparison(output_triton, output_torch, "VectorAdd", dtype_name)


def test_softmax():
    """Test Softmax operator"""
    print("\n=== Testing Softmax ===")
    torch.manual_seed(0)
    device = torch.npu.current_device()
    stream = torch.npu.current_stream(device).npu_stream

    # Test different data types
    test_cases = [
        (torch.float32, "float32"),
        (torch.float16, "float16"),
        (torch.bfloat16, "bfloat16"),
    ]

    for dtype, dtype_name in test_cases:
        x = torch.randn(1823, 781, device='npu', dtype=dtype)
        y_triton = triton_softmax(x, stream)
        y_torch = torch.softmax(x, axis=1)
        accuracy_comparison(y_triton, y_torch, "Softmax", dtype_name)


def test_layer_norm():
    """Test LayerNorm operator"""
    print("\n=== Testing LayerNorm ===")

    # Test different data types
    test_cases = [
        (128, 128, torch.float16, "float16"),
        (128, 128, torch.bfloat16, "bfloat16"),
        (128, 128, torch.float32, "float32"),
    ]

    for M, N, dtype, dtype_name in test_cases:
        x_shape = (M, N)
        w_shape = (x_shape[-1], )
        weight = torch.rand(w_shape, dtype=dtype, device='npu')
        bias = torch.rand(w_shape, dtype=dtype, device='npu')
        x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device='npu')

        y_tri = triton_layer_norm(x, w_shape, weight, bias, eps=1e-5)
        y_ref = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps=1e-5).to(dtype)
        accuracy_comparison(y_tri, y_ref, "LayerNorm", dtype_name)


def test_flash_attention():
    """Test FlashAttention operator"""
    print("\n=== Testing FlashAttention ===")

    # Test configurations
    test_configs = [
        (1, 1, 128, 128, torch.float16, 32, 128, "float16"),
        (1, 1, 128, 128, torch.bfloat16, 64, 128, "bfloat16"),
        (1, 2, 128, 128, torch.float16, 32, 128, "float16"),
        (1, 2, 256, 256, torch.bfloat16, 32, 256, "bfloat16"),
    ]

    for Z, H, N_CTX, HEAD_DIM, dtype, BM, BN, dtype_name in test_configs:
        # Skip non-divisible cases
        if N_CTX % BM != 0 or N_CTX % BN != 0 or HEAD_DIM % 16 != 0:
            continue

        torch.manual_seed(20)
        q = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device='npu').normal_(mean=0.0, std=0.5)
        k = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device='npu').normal_(mean=0.0, std=0.5)
        v = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device='npu').normal_(mean=0.0, std=0.5)

        sm_scale = 0.5

        tri_out = triton_attention(q, k, v, causal=False, sm_scale=sm_scale, BM=BM, BN=BN)
        ref_out = torch_npu.npu_fusion_attention(
            q, k, v, H,
            padding_mask=None,
            atten_mask=None,
            scale=sm_scale,
            keep_prob=1.0,
            input_layout="BNSD",
            pre_tockens=65535,
            next_tockens=65535,
            sparse_mode=0,
        )[0]

        try:
            torch.testing.assert_close(ref_out, tri_out, atol=1e-2, rtol=1e-2, equal_nan=True)
            print(f"✓ FlashAttention ({dtype_name}): PASSED - Config: ({Z},{H},{N_CTX},{HEAD_DIM})")
        except Exception as e:
            print(f"✗ FlashAttention ({dtype_name}): FAILED - Config: ({Z},{H},{N_CTX},{HEAD_DIM}) - {str(e)}")


def test_matmul():
    """Test Matmul operator"""
    print("\n=== Testing Matmul ===")
    torch.manual_seed(0)

    # Test different data types
    test_cases = [
        (512, 512, 512, torch.float16, "float16"),
        (512, 512, 512, torch.bfloat16, "bfloat16"),
        (256, 256, 256, torch.float32, "float32"),
    ]

    for M, K, N, dtype, dtype_name in test_cases:
        a = torch.randn((M, K), device='npu', dtype=dtype)
        b = torch.randn((K, N), device='npu', dtype=dtype)

        triton_output = triton_matmul(a, b, activation="leaky_relu_custom")

        # PyTorch reference
        torch_output = torch.matmul(a, b)
        torch_output = torch.where(torch_output >= 0, torch_output, 0.01 * torch_output) + 1.0

        accuracy_comparison(triton_output, torch_output, "Matmul", dtype_name)


def run_all_tests():
    """Run all accuracy comparison tests"""
    print("=" * 60)
    print("Triton-Ascend Operator Accuracy Comparison Tests")
    print("=" * 60)

    results = {}

    try:
        test_vector_add()
        results['VectorAdd'] = True
    except Exception as e:
        print(f"VectorAdd test failed with error: {e}")
        results['VectorAdd'] = False

    try:
        test_softmax()
        results['Softmax'] = True
    except Exception as e:
        print(f"Softmax test failed with error: {e}")
        results['Softmax'] = False

    try:
        test_layer_norm()
        results['LayerNorm'] = True
    except Exception as e:
        print(f"LayerNorm test failed with error: {e}")
        results['LayerNorm'] = False

    try:
        test_flash_attention()
        results['FlashAttention'] = True
    except Exception as e:
        print(f"FlashAttention test failed with error: {e}")
        results['FlashAttention'] = False

    try:
        test_matmul()
        results['Matmul'] = True
    except Exception as e:
        print(f"Matmul test failed with error: {e}")
        results['Matmul'] = False

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for op_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{op_name}: {status}")

    all_passed = all(results.values())
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
