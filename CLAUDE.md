# DiffuSpec — Project Context

## 项目概述

复现论文 **DiffuSpec: Unlocking Diffusion Language Models for Speculative Decoding**（arXiv 2510.02358v1）。使用预训练 Diffusion LM（Dream-7B）作为 drafter 替代自回归 drafter，通过 CPS + ADL 两个模块提升投机解码效率，无需额外训练。

论文原文：`2510.02358v1.pdf`

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
