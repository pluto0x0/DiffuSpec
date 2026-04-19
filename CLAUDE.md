# DiffuSpec — Project Context

## 项目概述

复现论文 **DiffuSpec: Unlocking Diffusion Language Models for Speculative Decoding**（arXiv 2510.02358v1）。使用预训练 Diffusion LM（Dream-7B）作为 drafter 替代自回归 drafter，通过 CPS + ADL 两个模块提升投机解码效率，无需额外训练。

论文原文：`2510.02358v1.pdf`

---

## 环境

| 项 | 值 |
|---|---|
| Python | 3.10.20 |
| PyTorch | 2.6.0+cu124 |
| transformers | 5.5.4 |
| accelerate | 1.13.0 |
| 运行时 venv | `/venv/diffuspec` |
| Python 解释器 | `/venv/diffuspec/bin/python` |
| HuggingFace 缓存 | `HF_HOME=/workspace/.hf_home` |
| GPU | NVIDIA GeForce RTX 3090 (24 GB VRAM) |

运行任何脚本须用 `/venv/diffuspec/bin/python`，或先执行：
```bash
export PATH="/venv/diffuspec/bin:$PATH"
export HF_HOME="/workspace/.hf_home"
```

---

## 本地已缓存模型

| 角色 | Model ID | vocab_size | hidden_size | 备注 |
|---|---|---|---|---|
| DLM Drafter | `dream-org/dream-v0-instruct-7b` | 152064 | 3584 | `mask_token_id=151666` |
| Target LM | `Qwen/Qwen3-8B` | 151936 | 4096 | 论文用 Qwen2.5-32B，本地只有 8B |

> **注意**：论文目标模型是 Qwen2.5-32B，本地只缓存了 Qwen3-8B。二者 vocab_size 不同（151936 vs 152064），投机解码时需要对齐 tokenizer 或处理词表差异。

---

## 代码结构

```
diffuspec/
├── config.py                  # DiffuSpecConfig + 子配置（ADL/CPS/Drafting）
├── engine.py                  # 主引擎：Algorithm 1 四阶段流水线
├── drafting/
│   └── dlm_drafter.py         # Stage 1: Dream-7B 封装，draft() + compute_l2r_logprobs()
├── search/
│   └── cps.py                 # Stage 2: Causal-Consistency Path Search
├── control/
│   └── adl.py                 # Stage 4: Adaptive Draft-Length 控制器
├── verification/
│   └── verifier.py            # Stage 3: AR 并行验证（greedy + stochastic）
└── proxy/
    └── ngram_proxy.py         # 因果代理：NgramProxy / UniformProxy / KenLMProxy
```

---

## 论文核心超参数（来自 Appendix A）

| 参数 | 值 | 位置 |
|---|---|---|
| `k_min` | 20 | ADLConfig |
| `k_max` | 30 | ADLConfig |
| `delta` (δ) | 10 | ADLConfig |
| `rho` (ρ) | 0.5 | ADLConfig |
| `M_max` | 15 | CPSConfig |
| `tau` (τ) | 0.8 | CPSConfig |
| `beam_size` (B) | 3 | CPSConfig |
| `lam` (λ) | 0.5 | CPSConfig |
| `num_refinement_steps` (S) | 1 | DraftingConfig |

---

## 已知问题与修复

### transformers 5.x RoPE 兼容问题
- **现象**：`KeyError: 'default'` 出现在 `DreamRotaryEmbedding.__init__`
- **原因**：transformers 5.x 从 `ROPE_INIT_FUNCTIONS` 移除了 `'default'` key，Dream 模型代码基于旧版本编写
- **修复**：`diffuspec/drafting/dlm_drafter.py` 模块顶层调用 `_patch_rope_init_functions()`，在 key 缺失时注入标准 RoPE 实现（`inv_freq = 1/base^(2i/dim)`, `scale=1.0`）

### 磁盘空间
- conda create 会填满磁盘，清理命令：
  ```bash
  /opt/miniforge3/bin/conda clean --all -y
  /venv/diffuspec/bin/pip cache purge
  ```

---

## 测试

```bash
/venv/diffuspec/bin/python -m pytest tests/test_components.py -v
# 12 tests: ADLController(5) + CPS(4) + NgramProxy(3) — 全部通过
```

---

## 快速运行

```bash
export HF_HOME=/workspace/.hf_home

/venv/diffuspec/bin/python scripts/generate.py \
    --prompt "Explain speculative decoding." \
    --target-model Qwen/Qwen3-8B \
    --drafter-model dream-org/dream-v0-instruct-7b \
    --max-new-tokens 128 \
    --device cuda \
    --verbose
```

---

## 待办 / 后续工作

- [ ] 处理 Qwen3-8B 与 Dream-7B 词表不对齐问题（vocab_size 151936 vs 152064）
- [ ] 验证端到端 generate() 流程（加载双模型后实际运行）
- [ ] 集成 3-gram KenLM proxy（论文实验使用，当前用 UniformProxy 占位）
- [ ] 实现 Spec-Bench 评测流程（MAT / Speedup 指标）
- [ ] 对 S、B、M_max、τ 做超参数消融实验（对应论文 Fig. 7）
