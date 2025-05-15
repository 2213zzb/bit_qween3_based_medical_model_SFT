# 基于Qwen3的医疗问答模型微调报告
# 文件构成
- data.py：数据集下载
- train.py：微调模型训练
- test.py：推理
- checkpoint-360：微调后模型
# 实验环境与命令
此代码使用 accelerate launch 启动多卡训练，通过 swanlab 实时监控代码运行情况。
## 环境安装
- swanlab

- modelscope==1.22.0

- transformers>=4.50.0

- datasets==3.2.0

- accelerate

- pandas

- addict

一键安装命令：
```
pip install swanlab modelscope==1.22.0 "transformers>=4.50.0" datasets==3.2.0 accelerate pandas addict
```
## 数据集下载
```
python3 ./data.py 
```

## 训练
由于作者在docker中运行代码，且主机GPU驱动版本过低：
```
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 PYTORCH_DISABLE_CHECK_KERNEL_VERSION=1 accelerate launch --num_processes=3 --mixed_precision=bf16 --dynamo_backend=no train.py
```
## 推理
```
python3 ./test.py
```