# AutoTritonOp - Triton-Ascend 算子实现

本项目包含基于 Triton-Ascend 实现的高性能算子代码，包括 VectorAdd、Softmax、LayerNorm、FlashAttention 和 Matmul 算子。

## 项目结构

```
AutoTritonOp/
├── ops/                    # 算子实现目录
│   ├── vector_add.py        # 向量加法算子
│   ├── softmax.py           # Softmax 算子
│   ├── layer_norm.py        # LayerNorm 算子
│   ├── flash_attention.py    # FlashAttention 算子
│   └── matmul.py           # 矩阵乘法算子
├── tests/                  # 单元测试目录
│   └── test_ops.py         # 单元测试
├── .github/                # GitHub Actions 配置
│   └── workflows/           # CI/CD 工作流
│       ├── build-wheel.yml      # 构建 wheel 包
│       ├── pre-commit-check.yml  # 代码规范检查
│       └── archive-artifacts.yml # 归档制品
├── test_accuracy.py        # 精度验证测试脚本
├── build.sh               # 构建和测试脚本
├── pyproject.toml          # 项目配置
├── setup.py               # 安装脚本
├── requirements.txt        # 依赖列表
├── .pre-commit-config.yaml  # Pre-commit 配置
└── README.md              # 本文件
```

## 算子说明

### 1. VectorAdd (向量加法)
- **文件**: `ops/vector_add.py`
- **功能**: 实现元素级向量加法运算
- **输入**: 两个形状相同的向量
- **输出**: 元素级相加的结果向量
- **特点**: 使用 Triton-Ascend kernel 实现高性能并行计算

### 2. Softmax (归一化指数函数)
- **文件**: `ops/softmax.py`
- **功能**: 实现行维度的 Softmax 归一化
- **输入**: 形状为 (n_rows, n_cols) 的矩阵
- **输出**: Softmax 归一化后的矩阵
- **特点**: 使用减去最大值的方法提高数值稳定性

### 3. LayerNorm (层归一化)
- **文件**: `ops/layer_norm.py`
- **功能**: 实现层归一化操作
- **输入**: 输入张量、权重、偏置
- **输出**: 归一化后的张量
- **特点**: 融合了均值、方差计算和仿射变换

### 4. FlashAttention (Flash Attention v2)
- **文件**: `ops/flash_attention.py`
- **功能**: 实现 Flash Attention v2 算法
- **输入**: Query、Key、Value 张量
- **输出**: 注意力输出
- **特点**: 支持 Causal Attention，使用分块计算减少内存访问

### 5. Matmul (矩阵乘法)
- **文件**: `ops/matmul.py`
- **功能**: 实现矩阵乘法运算
- **输入**: 矩阵 A (M, K) 和矩阵 B (K, N)
- **输出**: 矩阵 C (M, N)
- **特点**: 支持激活函数融合，使用自动调优配置

## 环境要求

- Python 3.8+
- PyTorch
- torch_npu
- Triton-Ascend

## 安装依赖

```bash
# 安装 torch_npu (根据您的环境选择合适的版本)
pip install torch_npu

# 安装 triton-ascend
cd /home/wpf/triton-ascend
pip install -e .
```

## 使用方法

### 1. 单独测试每个算子

```bash
# 测试 VectorAdd
python3 ops/vector_add.py

# 测试 Softmax
python3 ops/softmax.py

# 测试 LayerNorm
python3 ops/layer_norm.py

# 测试 FlashAttention
python3 ops/flash_attention.py

# 测试 Matmul
python3 ops/matmul.py
```

### 2. 运行精度验证测试

```bash
# 运行所有算子的精度验证测试
python3 test_accuracy.py
```

### 3. 使用构建脚本

```bash
# 赋予执行权限
chmod +x build.sh

# 运行构建脚本（会自动运行测试）
./build.sh
```

## 精度验证

精度验证测试脚本 (`test_accuracy.py`) 会将 Triton-Ascend 实现的结果与 PyTorch 参考实现进行比对：

- **float32**: 使用 rtol=1e-4, atol=1e-4
- **float16**: 使用 rtol=1e-3, atol=1e-3
- **bfloat16**: 转换为 float32 后使用 rtol=1e-3, atol=1e-3
- **整数类型**: 要求完全相等

## 算子接口示例

### VectorAdd

```python
from ops.vector_add import add

x = torch.rand(1024, device='npu')
y = torch.rand(1024, device='npu')
result = add(x, y)
```

### Softmax

```python
from ops.softmax import softmax

x = torch.randn(1823, 781, device='npu')
stream = torch.npu.current_stream(device).npu_stream
result = softmax(x, stream)
```

### LayerNorm

```python
from ops.layer_norm import layer_norm

x = torch.randn(128, 128, device='npu')
weight = torch.rand(128, device='npu')
bias = torch.rand(128, device='npu')
result = layer_norm(x, (128,), weight, bias, eps=1e-5)
```

### FlashAttention

```python
from ops.flash_attention import attention

q = torch.randn(1, 1, 128, 128, device='npu', dtype=torch.float16)
k = torch.randn(1, 1, 128, 128, device='npu', dtype=torch.float16)
v = torch.randn(1, 1, 128, 128, device='npu', dtype=torch.float16)
result = attention(q, k, v, causal=False, sm_scale=0.5, BM=32, BN=128)
```

### Matmul

```python
from ops.matmul import matmul

a = torch.randn(512, 512, device='npu', dtype=torch.float16)
b = torch.randn(512, 512, device='npu', dtype=torch.float16)
result = matmul(a, b, activation="leaky_relu_custom")
```

## 性能优化

所有算子都经过以下优化：

1. **并行计算**: 使用 Triton 的并行编程模型
2. **内存访问优化**: 使用合适的 block size 和 stride
3. **数值稳定性**: 在 Softmax 和 LayerNorm 中使用数值稳定算法
4. **自动调优**: Matmul 使用自动调优配置选择最佳参数

## CI/CD 工作流

本项目使用 GitHub Actions 实现自动化 CI/CD：

### 1. 构建 Wheel 包 (`.github/workflows/build-wheel.yml`)
- 自动构建多 Python 版本的 wheel 包
- 支持发布到 PyPI 和 Test PyPI
- 在推送标签或合并到 main/develop 分支时触发

### 2. 代码规范检查 (`.github/workflows/pre-commit-check.yml`)
- 运行 pre-commit 钩子检查代码质量
- 使用 Black、isort、Flake8、MyPy、Bandit 等工具
- 检查文档字符串风格和代码规范

### 3. 归档制品 (`.github/workflows/archive-artifacts.yml`)
- 自动创建源码归档（tar.gz 和 zip）
- 打包 wheel 文件
- 生成文档归档
- 计算校验和（SHA256 和 MD5）
- 归档到 GitHub 制品厂（保留 90 天）

### Pre-commit 配置 (`.pre-commit-config.yaml`)
- 自动格式化代码（Black、isort）
- 检查代码质量（Flake8、MyPy、Bandit）
- 检查文档字符串风格（pydocstyle）
- 检查文件格式和编码

## 触发 GitHub Actions Workflows

### 自动触发（推荐）

以下操作会自动触发所有 workflows：

```bash
cd /home/wpf/AutoTritonOp

# 1. 修改代码后提交
git add .
git commit -m "更新文档和配置"

# 2. 推送到 main 或 develop 分支（自动触发所有 workflows）
git push origin main
```

**自动触发的 workflows：**
- ✅ `build-wheel.yml` - 构建 wheel 包
- ✅ `pre-commit-check.yml` - 代码规范检查
- ✅ `archive-artifacts.yml` - 归档制品

### 手动触发

通过 GitHub 网页手动触发 workflows：

1. 访问：`https://gitcode.com/aflyingto/AutoTritonOps/actions`
2. 点击左侧的 "Workflows" 菜单
3. 选择要运行的 workflow
4. 点击 "Run workflow" 按钮
5. 选择分支和参数（如果有）
6. 点击绿色的 "Run workflow" 按钮

### 查看运行状态

访问：`https://gitcode.com/aflyingto/AutoTritonOps/actions`

### 下载制品

1. 访问：`https://gitcode.com/aflyingto/AutoTritonOps/actions`
2. 点击运行记录
3. 滚动到页面底部的 "Artifacts" 部分
4. 点击下载所需的制品

## 安装 Pre-commit

```bash
# 安装 pre-commit
pip install pre-commit

# 安装项目配置的 pre-commit 钩子
pre-commit install

# 手动运行所有 pre-commit 钩子
pre-commit run --all-files
```

## 注意事项

1. 确保在 NPU 设备上运行这些算子
2. FlashAttention 要求输入形状满足特定整除条件
3. 某些算子需要特定的数据类型支持
4. 建议在实际使用前先运行精度验证测试
5. 提交代码前会自动运行 pre-commit 检查

## 参考资料

- [Triton-Ascend 官方文档](https://gitee.com/ascend/triton-ascend)
- [Flash Attention v2 论文](https://tridao.me/publications/flash2/flash2.pdf)
- [Triton 编程指南](https://triton-lang.org/)

## 许可证

本项目遵循 Triton-Ascend 项目的许可证。
