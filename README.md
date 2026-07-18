# MiniChat-200M

这是一个完全随机初始化、从零训练的约 200M 参数中文聊天模型。项目不依赖
Transformers，也不下载任何预训练模型权重。PyTorch 仅用于张量计算和自动求导。

## 本机环境

- Python 3.12.10：`.runtime/Python312`
- 虚拟环境：`.venv`
- PyTorch 2.13.0 + CUDA 13.0
- GPU：NVIDIA GeForce RTX 5060 Ti 8 GB

在 PowerShell 中激活环境：

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/check_environment.py
```

## 数据来源

- 预训练：[HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)，
  使用 `cmn_Hani` 配置，许可为 ODC-BY。
- 对话训练：[BAAI/COIG](https://huggingface.co/datasets/BAAI/COIG)，许可为
  Apache-2.0；部分组成数据有各自的兼容许可，详见上游数据卡。

原始数据保存在 `data/raw`，处理结果保存在 `data/processed`。模型训练不会向
Hugging Face 上传任何内容。

## 数据准备

训练 32K BPE 词表：

```powershell
python scripts/train_tokenizer.py `
  --input "data/raw/fineweb2/data/cmn_Hani/test/*.parquet" `
  --output tokenizer --vocab-size 32000
```

打包验证集和正式预训练集：

```powershell
python scripts/prepare_pretrain.py `
  --input "data/raw/fineweb2/data/cmn_Hani/test/*.parquet" `
  --tokenizer tokenizer/tokenizer.json `
  --output data/processed/pretrain/validation.bin

python scripts/prepare_pretrain.py `
  --input "data/raw/fineweb2/data/cmn_Hani/train/*.parquet" `
  --tokenizer tokenizer/tokenizer.json `
  --output data/processed/pretrain/train.bin `
  --max-tokens 4000000000
```

打包 COIG 对话数据：

```powershell
python scripts/prepare_sft.py --seq-len 1024
```

## 训练

预训练配置对应约 40 亿 token。每一步是 `64 x 1024 = 65,536` token：

```powershell
python scripts/train.py --config configs/pretrain.yaml
```

预训练完成后进行监督微调：

```powershell
python scripts/train.py --config configs/sft.yaml
```

训练会写入 `latest.pt`，中断后执行同一命令即可自动恢复。默认只保留最近两个
步骤检查点，避免训练状态占满磁盘。

本机实测正式配置约为 4,970 token/s、峰值显存 4.41 GiB。40 亿 token 的纯训练
时间约 9.3 天，考虑验证、保存和系统波动应按 9～12 天估算。训练期间应关闭占用
显卡的浏览器和聊天软件，并避免系统休眠或自动重启。

## 聊天

```powershell
python scripts/chat.py --checkpoint checkpoints/sft-200m/latest.pt
```

### Web 界面

启动本地 Web 服务：

```powershell
.\.venv\Scripts\python.exe web\server.py `
  --checkpoint checkpoints\quick-chat-25m\latest.pt
```

浏览器访问 `http://127.0.0.1:8000`。服务只监听本机地址；聊天记录和生成参数保存在
当前浏览器的本地存储中，点击“新对话”可清空。

一小时快速版本使用约 25M 参数、31.7M 预训练 token，再进行短时 COIG 微调：

```powershell
python scripts/prepare_sft.py --seq-len 512 `
  --output data/processed/sft-512 `
  --validation-output data/processed/sft-512-validation
python scripts/train.py --config configs/quick_pretrain.yaml
python scripts/train.py --config configs/quick_sft.yaml
python scripts/chat.py --checkpoint checkpoints/quick-sft-25m/latest.pt
```

对于小模型，纯多轮聊天子集通常比混合考试任务更稳定：

```powershell
python scripts/prepare_sft.py --chat-only --seq-len 512 `
  --output data/processed/chat-sft-512 `
  --validation-output data/processed/chat-sft-512-validation
python scripts/train.py --config configs/quick_chat_sft.yaml
python scripts/chat.py --checkpoint checkpoints/quick-chat-25m/latest.pt
```

200M 模型容量有限，最终聊天质量主要取决于预训练 token 数、数据清洗和训练是否
完整收敛。刚跑通冒烟测试的检查点不能正常聊天，必须完成预训练与 SFT。
