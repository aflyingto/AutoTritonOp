# AutoTritonOp

该仓库包含一个昇腾 NPU 的 Triton-Ascend 算子示例：

- `ascend_triton_ops.py`
  - `triton_ascend_add`：向量加法算子
  - `triton_ascend_silu`：SiLU 激活算子

## 使用方式

```bash
python ascend_triton_ops.py
```

> 运行前请确保环境已安装支持 Ascend 后端的 Triton 版本，并且 PyTorch 可以访问 `npu` 设备。
