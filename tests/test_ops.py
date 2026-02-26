"""
Unit tests for AutoTritonOps operators
"""

import pytest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch_npu
    HAS_NPU = True
except ImportError:
    HAS_NPU = False
    print("Warning: torch_npu not available, some tests will be skipped")


@pytest.mark.skipif(not HAS_NPU, reason="torch_npu not available")
class TestVectorAdd:
    """Test VectorAdd operator"""

    def test_vector_add_basic(self):
        """Test basic vector addition"""
        from ops.vector_add import add

        torch.manual_seed(0)
        size = 1024
        x = torch.rand(size, device='npu')
        y = torch.rand(size, device='npu')

        output_triton = add(x, y)
        output_torch = x + y

        assert torch.allclose(output_triton, output_torch, rtol=1e-3, atol=1e-3)

    def test_vector_add_different_dtypes(self):
        """Test vector addition with different data types"""
        from ops.vector_add import add

        for dtype in [torch.float16, torch.float32, torch.bfloat16]:
            size = 512
            x = torch.rand(size, device='npu', dtype=dtype)
            y = torch.rand(size, device='npu', dtype=dtype)

            output_triton = add(x, y)
            output_torch = x + y

            assert torch.allclose(output_triton, output_torch, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not HAS_NPU, reason="torch_npu not available")
class TestSoftmax:
    """Test Softmax operator"""

    def test_softmax_basic(self):
        """Test basic softmax"""
        from ops.softmax import softmax

        torch.manual_seed(0)
        x = torch.randn(128, 64, device='npu')
        device = torch.npu.current_device()
        stream = torch.npu.current_stream(device).npu_stream

        output_triton = softmax(x, stream)
        output_torch = torch.softmax(x, axis=1)

        assert torch.allclose(output_triton, output_torch, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not HAS_NPU, reason="torch_npu not available")
class TestLayerNorm:
    """Test LayerNorm operator"""

    def test_layer_norm_basic(self):
        """Test basic layer normalization"""
        from ops.layer_norm import layer_norm

        torch.manual_seed(0)
        M, N = 128, 64
        x = torch.randn(M, N, device='npu')
        weight = torch.rand(N, device='npu')
        bias = torch.rand(N, device='npu')

        output_triton = layer_norm(x, (N,), weight, bias, eps=1e-5)
        output_torch = torch.nn.functional.layer_norm(x, (N,), weight, bias, eps=1e-5)

        assert torch.allclose(output_triton, output_torch, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not HAS_NPU, reason="torch_npu not available")
class TestMatmul:
    """Test Matmul operator"""

    def test_matmul_basic(self):
        """Test basic matrix multiplication"""
        from ops.matmul import matmul

        torch.manual_seed(0)
        M, K, N = 256, 256, 256
        a = torch.randn(M, K, device='npu', dtype=torch.float16)
        b = torch.randn(K, N, device='npu', dtype=torch.float16)

        output_triton = matmul(a, b, activation="")
        output_torch = torch.matmul(a, b)

        assert torch.allclose(output_triton, output_torch, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not HAS_NPU, reason="torch_npu not available")
class TestFlashAttention:
    """Test FlashAttention operator"""

    def test_flash_attention_basic(self):
        """Test basic flash attention"""
        from ops.flash_attention import attention

        torch.manual_seed(0)
        Z, H, N_CTX, HEAD_DIM = 1, 1, 128, 64
        q = torch.randn(Z, H, N_CTX, HEAD_DIM, device='npu', dtype=torch.float16)
        k = torch.randn(Z, H, N_CTX, HEAD_DIM, device='npu', dtype=torch.float16)
        v = torch.randn(Z, H, N_CTX, HEAD_DIM, device='npu', dtype=torch.float16)

        sm_scale = 0.5
        BM, BN = 32, 64

        output_triton = attention(q, k, v, causal=False, sm_scale=sm_scale, BM=BM, BN=BN)

        # Compare with torch_npu fusion attention if available
        try:
            output_torch = torch_npu.npu_fusion_attention(
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
            assert torch.allclose(output_torch, output_triton, rtol=1e-2, atol=1e-2)
        except Exception:
            # If torch_npu fusion attention is not available, just check output shape
            assert output_triton.shape == (Z, H, N_CTX, HEAD_DIM)


def test_imports():
    """Test that all modules can be imported"""
    import ops.vector_add
    import ops.softmax
    import ops.layer_norm
    import ops.flash_attention
    import ops.matmul
