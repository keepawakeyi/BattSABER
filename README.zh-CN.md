# BattSABER

**Segmentation-Aware Basis Learning for Robust Battery Degradation Forecasting Under Heterogeneous Operating Conditions**

主方法代码将在论文接收后公开。

当前仓库提供跨工况电池容量预测的对比方法、处理后数据、实验配置和运行入口。

## 模型

默认运行 MLP、CNN、DLinear、LDNet、OmniTIEFormer、ModernTCN，另提供可选 GRU。

## 安装

建议 Python 3.10，根据设备安装对应版本的 PyTorch。

```bash
python -m pip install -r requirements.txt
```

## 数据

- 55块电池，36个工程特征加历史Capacity，共37个输入通道。
- Source：2C、3C、R2.5、R3；Target：RW、Satellite。
- 训练／验证／测试分别为30／9／16块电池。
- 使用64周期历史数据，预测未来20周期容量，单位Ah。

数据位于 `data/xjtu_cccv36/`，特征定义和固定划分分别见
`data/feature_manifest.csv`、`configs/split.json`。

## 运行

```bash
# 检查数据和模型接口
python run.py --action audit --device cpu

# 小规模运行测试
python run.py --action smoke --device cpu

# 使用选定配置运行全部默认模型
python run.py --action train --device cuda --resume

# 只运行指定模型和种子
python run.py --action train --methods MLP ModernTCN --seeds 1 --device cuda --output outputs/subset
```

无GPU时使用 `--device cpu`。默认batch size为256，最多训练200轮，早停patience为10，
使用1–5五个种子。模型配置保存在 `configs/selected.json`。

训练输入采用50% clean与50% mixed-noise，验证输入和预测标签保持clean；
测试噪声等级为0、0.2、0.4。结果先对两个目标域等权平均，再计算种子间均值和标准差。

结果写入 `outputs/`。添加 `--save-predictions` 可保存逐窗口预测，`--resume` 用于续跑。
更改实验设置时，请使用新的输出目录。

## 调参

```bash
python run.py --action tune --device cuda --output outputs/search --resume
python run.py --action train --config outputs/search/tuned_selected.json --device cuda --output outputs/retuned --resume
```

每个模型搜索5组候选参数，按Source验证集MAE选择。若只搜索部分模型，两条命令使用
相同的 `--methods` 列表即可。
