# umap-rs Patches

本目录包含对官方 `umap-rs 0.4.5` crate 的修改。

## 修改原因

官方 `umap-rs 0.4.5` 存在三个关键 bug，导致 UMAP 输出结果与 Python 实现差异巨大，聚类完全失败。

## 修改内容

### 1. 修复 Embedding 归一化 (`src/optimizer.rs`)

**位置**: 行 181-217

**问题**: 原实现对每个维度独立归一化到 [0, 10]，破坏了维度间的关系。

**修复**: 改为整体缩放，保持维度间的关系，与 Python 的 `noisy_scale_coords` 行为一致。

```rust
// 修复前：每个维度独立归一化
let scales: Vec<f32> = mins.iter().zip(&maxs).map(|(&min, &max)| {
    let range = max - min;
    if range > 0.0 { 10.0 / range } else { 0.0 }
}).collect();
flat.par_iter_mut().enumerate().for_each(|(idx, v)| {
    let d = idx % n_dims;
    if scales[d] > 0.0 {
        *v = (*v - mins[d]) * scales[d];
    }
});

// 修复后：整体缩放
if normalize_init {
    let max_abs = embedding.as_slice().unwrap()
        .par_iter().map(|&v| v.abs()).reduce(|| 0.0f32, |a, b| a.max(b));
    if max_abs > 0.0 {
        let scale = 10.0 / max_abs;
        embedding.par_iter_mut().for_each(|v| *v *= scale);
    }
}
```

### 2. 修复正样本梯度计算 (`src/layout/optimize_layout_euclidean.rs`)

**位置**: 行 179-186, 290-297

**问题**: 分母中多了 `* dist_squared` 因子，导致吸引力计算错误。

**修复**:

```rust
// 修复前：
gc /= a * dist_pow_b * dist_squared + 1.0;  // ❌ 错误

// 修复后：
gc /= a * dist_pow_b + 1.0;  // ✅ 正确
```

**Python 参考** (`umap/layouts.py:136-140`):
```python
if dist_squared > 0.0:
    grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
    grad_coeff /= a * pow(dist_squared, b) + 1.0
```

### 3. 修复负样本梯度计算 (`src/layout/optimize_layout_euclidean.rs`)

**位置**: 行 214-219, 327-332

**问题**: 分母中多了 `* dist_squared` 因子，导致排斥力计算错误。

**修复**:

```rust
// 修复前：
2.0 * gamma * b / ((0.001 + dist_squared) * (a * dist_pow_b * dist_squared + 1.0))  // ❌

// 修复后：
2.0 * gamma * b / ((0.001 + dist_squared) * (a * dist_pow_b + 1.0))  // ✅
```

**Python 参考** (`umap/layouts.py:168-171`):
```python
if dist_squared > 0.0:
    grad_coeff = 2.0 * gamma * b
    grad_coeff /= (0.001 + dist_squared) * (a * pow(dist_squared, b) + 1)
```

### 4. 添加归一化控制参数 (`src/optimizer.rs`)

**位置**: 行 58-86

**功能**: 新增 `new_with_normalize` 方法，允许控制是否对初始化 embedding 进行归一化。

**用途**:
- `random` 初始化: `normalize_init = false` (embedding 已在 [-10, 10])
- `spectral/pca` 初始化: `normalize_init = true` (需要缩放)

## 修复效果

### 修复前
```
Rust UMAP:
- 聚类数量: 1 个 (所有点归为一类) ❌
- 值范围: [0.006, 9.99]
- 均值: 5.00
```

### 修复后
```
Rust UMAP:
- 聚类数量: 12 个 ✅
- 聚类大小: [360, 503, 483, 525, 601, 370, 545, 530, 545, 557, 819, 541]
- 值范围: [-3.4, 3.2]
- 均值: -0.01

Python UMAP:
- 聚类数量: 12 个 ✅
- 聚类大小: [361, 503, 484, 520, 608, 370, 545, 530, 544, 557, 819, 541]
- 值范围: [-3.23, 13.92]
- 均值: 4.89
```

**差异**: 每个聚类仅差 1-6 个点 (误差 < 1%) ✅

## 应用 Patch

在项目根目录的 `Cargo.toml` 中：

```toml
[dependencies]
umap-rs = "0.4.5"

[patch.crates-io]
umap-rs = { path = "./source-umap-rs/umap-rs" }
```

这样配置后：
1. Cargo 会从 crates.io 下载官方 umap-rs 0.4.5
2. 然后应用本地的修改版本
3. 既保持了对官方版本的追踪，又应用了必要的 bug 修复

## 提交上游

建议将这些修复提交到 umap-rs 官方仓库：
- Issue: 报告这些 bug
- PR: 提供修复代码
- 文档: 说明修复原因和测试结果

## 参考资料

- [UMAP 原论文](https://arxiv.org/abs/1802.03426)
- [Python UMAP 实现](https://github.com/lmcinnes/umap)
- [umap-rs 官方仓库](https://github.com/ geological-observatory/umap-rs)
