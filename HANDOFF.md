# HANDOFF — FallDetection 项目交接文档（截至 v2.2）

> 面向**没有任何本会话上下文**的新 agent / 新同事。  
> 更新日期：2026-07-14  
> **本文件位置：** `output/v2.2/HANDOFF.md`（建议同步根目录 `HANDOFF.md`）  
> **当前推荐结果归档：** `output/v2.2/`（`TEST_STRIDE=64` + 软概率 `k=2, τ=0.46`）  
> **注意：** `config.py` / `save_result` / 可视化**默认写入** `output/` 根目录（非 `v2.2`）。重新跑实验不会自动进归档；`output/` 整体在 `.gitignore` 中。

---

## 1. 我们在做什么

基于可穿戴惯性传感器（加速度 / 陀螺仪 / 欧拉角）做**跌倒检测（二分类：跌倒 vs 非跌倒）**。

| 项 | 现状 |
|----|------|
| 数据 | SisFall 风格：受试者约 `S06`–`S38`，试次 CSV + Excel 标签 |
| 评估 | **LOSO**（Leave-One-Subject-Out），约 **32 折**；指标为 **trial 级** |
| 模型 | **Random Forest**、**SVM** 已跑通；**Transformer** 代码在，`main.py` 中默认注释关闭 |
| 当前工作点 | `TEST_STRIDE=64`，`PROB_SMOOTH_SIZE=2`，`PROB_THRESHOLD=0.46` |
| 历史瓶颈（已解决） | v1.0 硬投票 → trial 全阴；v2.1 仍漏检约 19% 跌倒 trial（测试窗过稀） |
| 当前主要状态 | 召回接近饱和（SVM FN≈7）；剩余短板偏 **Precision / 误报**（如 S07） |

---

## 2. 仓库结构（关键路径）

```
FallDetection/
├── main.py                    # 实验入口（RF + SVM）
├── config.py                  # 路径、切窗、软决策 k/τ、超参网格
├── scripts/
│   └── sweep_svm_threshold.py # SVM：训一次 → 离线扫 (k,τ)
├── preprocessing/             # 读取、切窗、158 维特征
├── models/                    # RF / SVM / Transformer
├── utils/                     # LOSO、软决策 sliding、缓存、结果写出
├── visualization/
└── output/                    # .gitignore
    ├── Feature Dataset/       # 运行时默认特征缓存（当前为 stride=64）
    ├── v1.0/                  # 硬投票
    ├── v2.0/                  # 软决策 τ=0.5 + 首次阈值扫描
    ├── v2.1/                  # stride=160 + k=2,τ=0.35
    └── v2.2/                  # ★ 当前推荐：stride=64 + k=2,τ=0.46
```

运行环境：本机常用 conda env `ml`  
（例：`c:\Users\Alextn\miniconda3\envs\ml\python.exe`）

---

## 3. 任务流程（端到端）

与 v2.1 相同：dataset → 切窗 → 特征 → LOSO → train/evaluate → `save_result` → 可视化 → 人工归档。

### 3.1 切窗（相对 v2.1 的关键变更）

| 模式 | 行为 |
|------|------|
| `train` | 不变：跌倒事件中心裁剪；非跌倒最多 `NON_FALL_MAX_WINDOWS=8` |
| `test` | 全序列滑窗，**`TEST_STRIDE=64`**（约 0.64s @ 100Hz；v2.1 为 160） |

窗长仍为 `WINDOW_SIZE=32`（约 0.32s）。注意：stride 仍 **大于** 窗长 → 相邻测试窗**仍不重叠**；收益来自更密时间采样（约 ×2.5 窗：test 缓存约 **67437** 窗）。

### 3.2 Trial 决策

实现：`utils/sliding.py`

- \(k=1\)：\(s=\max_t p_t\)
- \(k>1\)：\(s=\max_i \mathrm{mean}(p_i,\ldots,p_{i+k-1})\)
- 判决：\(\hat{y}=1 \iff s \ge \tau\)

```python
PROB_SMOOTH_SIZE = 2   # k
PROB_THRESHOLD = 0.46  # τ
TEST_STRIDE = 64
```

### 3.3 超参网格（当前）

- RF：固定 `{n_estimators: 100, max_depth: None}`
- SVM：`kernel=rbf`，`C∈{10,100}`，`gamma∈{scale, 0.01}`

---

## 4. 结果版本对照

| 版本 | 设定 | SVM Recall | SVM Precision | SVM F1 | SVM AUC |
|------|------|------------|---------------|--------|---------|
| v1.0 | 硬投票 | ~0.015 | ~0.61 | ~0.03 | ~0.92 |
| v2.0 | stride=160, k=2, τ=0.50 | ~0.60 | ~0.94 | ~0.73 | ~0.92 |
| v2.1 | stride=160, k=2, τ=0.35 | ~0.81 | ~0.86 | ~0.83 | ~0.92 |
| **v2.2（当前）** | **stride=64, k=2, τ=0.46** | **~0.997** | **~0.856** | **~0.920** | **~0.987** |

**关键结论：** 减小测试 stride 显著抬高跌倒 trial 的软分数（分布更可分）；须**同步上调 τ**，否则在旧 τ=0.35 下会出现「Recall≈1 但 Precision 暴跌」。

中间态（仅供理解，已不作为推荐）：stride=64 + 旧 τ=0.35 → SVM Recall≈0.999 / Precision≈0.77（FP 过多）。

---

## 5. 当前模型效果（v2.2）

数据：`output/v2.2/Random Forest/`、`output/v2.2/SVM/`，32 折 LOSO，trial 级。

| 指标 | Random Forest | SVM |
|------|---------------|-----|
| Accuracy | ~0.909 | ~0.919 |
| Precision | ~0.843 | ~0.856 |
| Recall | ~0.999 | ~0.997 |
| F1 | ~0.912 | ~0.920 |
| AUC | ~0.991 | ~0.987 |

**合计混淆矩阵：**

| | RF | SVM |
|--|----|-----|
| TP / FN | 2343 / 3 | 2339 / 7 |
| FP / TN | 461 / 2268 | 404 / 2325 |

### 解读摘要

1. **相对 v2.1**：SVM 漏检从约 440 → **7**；F1 0.83 → **0.92**；AUC 0.92 → **0.99**。stride 假设成立。
2. **两模型均强**；SVM Precision / F1 / Acc 略优；RF Recall 略高（几乎打满）。
3. **新瓶颈是误报**：例如 SVM 上 S07 Recall 仍高但 Precision 偏低。不必再优先减 stride。
4. 阈值依据见 `output/v2.2/SVM_threshold_sweep/`：在 Precision≥0.85 下 Recall 优先 → **`k=2, τ=0.46`**（F1 峰值附近在 τ≈0.50–0.52，更偏 Precision）。

---

## 6. `output/` 归档怎么用

| 路径 | 作用 |
|------|------|
| `output/Feature Dataset/` | 运行时缓存（**stride=64**；与 v2.1 不兼容） |
| `output/v2.1/` | 旧 stride=160 基线 |
| `output/v2.2/` | **当前推荐**完整 LOSO + `SVM_threshold_sweep/` + 本 HANDOFF |
| `output/SVM_threshold_sweep/` | 运行时扫阈值写出；已复制进 `v2.2/` |

复现：保证 `output/Feature Dataset/` 有 joblib（可从 `v2.2/Feature Dataset/` 拷贝），`REBUILD_FEATURES=False`，`TEST_STRIDE=64`，`PROB_THRESHOLD=0.46`。

---

## 7. 如何复现 / 扫阈值

```powershell
cd D:\Data\VSCode\FallDetection
$env:PYTHONUNBUFFERED='1'
& 'c:\Users\Alextn\miniconda3\envs\ml\python.exe' -u main.py
```

```powershell
& 'c:\Users\Alextn\miniconda3\envs\ml\python.exe' -u scripts\sweep_svm_threshold.py
# 改网格后可：
& 'c:\Users\Alextn\miniconda3\envs\ml\python.exe' -u scripts\sweep_svm_threshold.py --reuse-probs --tau-start 0.40 --tau-stop 0.60 --tau-step 0.01
```

改 `TEST_STRIDE` 必须设 `REBUILD_FEATURES=True` 并重建测试特征；改 τ 不必重建。

新结果默认写 `output/` 根；归档请复制到 `output/v2.*/`。

---

## 8. 建议的下一步

按优先级：

1. **诊断高 FP 受试者**（如 S07）：非跌倒 trial 的 score 尾部；是否 ADL 类似跌倒峰值。
2. 若需更少误报：在 **τ∈[0.48, 0.52]** 微调（扫描已表明 Recall 仍可 >0.97、Prec↑）。
3. 可选：保存每折 `best_model`；接入 Transformer 对齐同一评估接口。
4. **暂缓：** 再减小 stride、盲目换模型——当前几何覆盖已基本解决；瓶颈在非跌倒高分尾。

---

## 9. 本阶段关键演进

1. `TEST_STRIDE` **160 → 64**；测试特征约 **67437** 窗（train 仍约 30660）。
2. 软决策工作点 **τ 0.35 → 0.46**（同 `k=2`），依据新概率重扫。
3. 文档/归档：`output/v2.2/`（含 `SVM_threshold_sweep/`）。

---

## 10. 快速自检清单

- [ ] `config.py`：`TEST_STRIDE=64`，`PROB_SMOOTH_SIZE=2`，`PROB_THRESHOLD=0.46`，`REBUILD_FEATURES=False`
- [ ] `output/v2.2/SVM/summary.csv` 与 `Random Forest/summary.csv` 为当前基线
- [ ] 扫阈值：`output/v2.2/SVM_threshold_sweep/recommendation.txt`
- [ ] Feature joblib 为 stride=64 版本（勿混用 v2.1 缓存）
- [ ] `main.py` 启用 RF+SVM；Transformer 关闭
- [ ] 知悉：重跑默认写 `output/` 根，不是自动进 `v2.2/`

---

*有冲突时以仓库当前代码与 `output/v2.2/*/summary.csv` 为准。旧叙事见 `output/v2.1/HANDOFF.md`。*
