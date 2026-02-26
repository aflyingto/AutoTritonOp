"""Triton-Ascend 算子示例。

该文件提供了一个可直接复用的 Ascend NPU Triton-Ascend 算子模板：
1) 向量加法 (add)
2) SiLU 激活 (silu)

注意：运行前需要安装支持昇腾后端的 Triton-Ascend 发行版，并确保张量在 NPU 上。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def _silu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # SiLU: x * sigmoid(x)
    out = x * tl.sigmoid(x)
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_ascend_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """使用 Triton-Ascend 实现向量加法。

    Args:
        x: 输入张量，要求与 y 同形状、同 dtype，且位于 NPU。
        y: 输入张量，要求与 x 同形状、同 dtype，且位于 NPU。

    Returns:
        输出张量 out = x + y。
    """

    if x.shape != y.shape:
        raise ValueError(f"x.shape ({x.shape}) 必须与 y.shape ({y.shape}) 一致")
    if x.dtype != y.dtype:
        raise ValueError(f"x.dtype ({x.dtype}) 必须与 y.dtype ({y.dtype}) 一致")
    if not x.is_contiguous() or not y.is_contiguous():
        raise ValueError("x 与 y 必须是 contiguous 张量")

    out = torch.empty_like(x)
    n_elements = out.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)
    return out


def triton_ascend_silu(x: torch.Tensor) -> torch.Tensor:
    """使用 Triton-Ascend 实现 SiLU 激活。"""

    if not x.is_contiguous():
        raise ValueError("x 必须是 contiguous 张量")

    out = torch.empty_like(x)
    n_elements = out.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _silu_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)
    return out


def _demo():
    """简单示例：在 NPU 上调用 Triton-Ascend 算子。"""

    device = "npu"
    x = torch.randn(4096, device=device, dtype=torch.float16)
    y = torch.randn(4096, device=device, dtype=torch.float16)

    add_out = triton_ascend_add(x, y)
    silu_out = triton_ascend_silu(x)

    # 与 PyTorch 结果做快速一致性检查
    torch.testing.assert_close(add_out, x + y, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(silu_out, torch.nn.functional.silu(x), rtol=1e-2, atol=1e-2)
    print("Triton-Ascend demo passed.")


if __name__ == "__main__":
    _demo()
