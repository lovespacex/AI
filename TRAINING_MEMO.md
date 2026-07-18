# 200M 中文聊天模型训练备忘录

## 固定方案

200M 模型采用从零预训练路线，不从现有 25M 检查点扩展：

1. 继续复用现有 32K tokenizer。
2. 复用 FineWeb 2 中文数据的下载和处理代码。
3. 随机初始化约 200M 参数模型。
4. 完成约 40 亿 token 的通用中文预训练。
5. 加载 200M 预训练检查点，使用 COIG 进行对话监督微调。

现有 25M 模型只用于验证 tokenizer、数据处理、训练、断点恢复和聊天界面的完整流程，
不作为 200M 模型的初始化权重。

## 预训练

模型配置：`configs/model_200m_no_ckpt.yaml`

训练配置：`configs/pretrain.yaml`

当前配置的总训练量为：

```text
61000 steps × 64 gradient accumulation × 1 micro batch × 1024 tokens
= 3,997,696,000 tokens
≈ 40 亿 tokens
```

准备预训练数据：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_pretrain.py `
  --input "data/raw/fineweb2/data/cmn_Hani/train/*.parquet" `
  --tokenizer tokenizer/tokenizer.json `
  --output data/processed/pretrain/train.bin `
  --max-tokens 4000000000
```

启动或恢复预训练：

```powershell
.\.venv\Scripts\python.exe scripts\train.py `
  --config configs/pretrain.yaml
```

检查点目录：`checkpoints/pretrain-200m`

## 对话微调

预训练完成后再执行 COIG 微调，不使用随机初始化模型直接做 SFT。

准备 COIG 数据：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_sft.py `
  --input data/raw/coig `
  --tokenizer tokenizer/tokenizer.json `
  --output data/processed/sft `
  --validation-output data/processed/sft-validation `
  --seq-len 1024
```

启动或恢复微调：

```powershell
.\.venv\Scripts\python.exe scripts\train.py `
  --config configs/sft.yaml
```

`configs/sft.yaml` 默认从以下预训练检查点初始化：

```text
checkpoints/pretrain-200m/latest.pt
```

微调检查点目录：`checkpoints/sft-200m`

## 硬件时间预估

- RTX 5060 Ti 8GB：约 9～12 天完成 40 亿 token。
- 单张 A800 40GB/80GB：调大 micro batch 后约 0.6～1.2 天。

A800 需要保持每次更新的有效批量不变。40GB 可以从
`micro_batch_size: 16`、`gradient_accumulation_steps: 4` 开始；80GB 可以从
`micro_batch_size: 32`、`gradient_accumulation_steps: 2` 开始，并根据实测显存调整。

## 完成标准

1. 预训练达到约 40 亿 token，训练和验证 loss 无发散。
2. `latest.pt` 可以在新进程中恢复模型、优化器和步数。
3. COIG 微调必须从 200M 预训练检查点开始。
4. 使用未出现在训练脚本中的中文问题进行真实多轮生成测试。
5. 模型权重通过 Git LFS 或独立模型存储发布，不提交原始训练语料。

