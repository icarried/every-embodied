# ETPNav 复现与训练指南

本指南提供了 ETPNav (Evolving Topological Planning for Vision-Language Navigation in Continuous Environments) 的完整复现流程，包含环境配置、数据集下载、权重准备以及模型训练与评估的具体步骤。

## 1. 环境配置

本项目推荐使用 Python 3.8 环境。请按照以下步骤配置虚拟环境及相关依赖：

### 1.1 创建虚拟环境与安装 PyTorch

```bash
# 创建并激活 conda 虚拟环境
conda create -n vlnce38 python=3.8
conda activate vlnce38

# 安装 PyTorch (1.9.1+cu111)
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f [https://download.pytorch.org/whl/torch_stable.html](https://download.pytorch.org/whl/torch_stable.html)
```

### 1.2 安装项目依赖
下载requirements.txt：[百度网盘](https://pan.baidu.com/s/14HF1YRknfD6V70iR5dJKNQ) (提取码: `nrjs`) 
```bash
python -m pip install "pip<24.1"
python -m pip install -r requirements.txt
```

### 1.3 安装 Habitat 仿真器

首先下载无头版本 Habitat-sim v0.1.7 预编译包：[点击下载](https://anaconda.org/aihabitat/habitat-sim/0.1.7/download/linux-64/habitat-sim-0.1.7-py3.8_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7.tar.bz2)

```bash
# 安装 habitat-sim
conda install habitat-sim-0.1.7-py3.8_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7.tar.bz2

# 克隆并安装 habitat-lab
git clone --branch v0.1.7 https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab-0.1.7
pip install -e .
```

## 2. 数据集下载

### 2.1 场景数据 (Scenes): Matterport3D (MP3D)

需要下载 Matterport3D 场景重建数据，共有 90 个场景。最终的存放路径应为：`data/scene_datasets/mp3d/{scene}/{scene}.glb`

* **方式一：官方脚本申请下载** (需要 Python 2.7)
请访问 Matterport3D 的[官方项目主页](https://niessner.github.io/Matterport/)，按照网页上的说明获取官方的下载脚本 (`download_mp.py`)。
  ```bash
  python download_mp.py --task habitat -o data/scene_datasets/mp3d/
  ```

* **方式二：网盘快捷下载**
  * 链接: [百度网盘](https://pan.baidu.com/s/1XRXDsRhg4j09nHxXBe9boA)
  * 提取码: `4kz6`

### 2.2 任务数据 (Episodes): R2R & RxR

请将以下下载的 Episode 数据放置在 `data/datasets` 目录下。

| 数据集 | 下载链接 | 存放路径 |
| --- | --- | --- |
| **R2R_VLNCE_v1-2_preprocessed** | [Google Drive](https://drive.google.com/file/d/1j9sQ0w4wFYSafh42U8VCuKTwMrnrsV6z/view) | `data/datasets` |
| **R2R_VLNCE_v1-2_preprocessed_BERTidx** | [Google Drive](https://drive.google.com/file/d/1j9sQ0w4wFYSafh42U8VCuKTwMrnrsV6z/view) | `data/datasets` |
| **RxR** | [百度网盘](https://pan.baidu.com/s/1bcgUSQ4WDawxkrpj9FFU9w) (提取码: `eqph`) | `data/datasets` |

### 2.3 连通图 (Connectivity Graphs)

用于可视化的连通图文件：
* **下载链接**: [connectivity_graphs.pkl](https://github.com/jacobkrantz/VLN-CE/blob/master/data/connectivity_graphs.pkl)
* **存放路径**: `data/connectivity_graphs.pkl`

最终数据集存储结构如下：

```text
ETPNav/
└── data/
    ├── datasets/
    │   ├── R2R_VLNCE_v1-2_preprocessed/
    │   ├── R2R_VLNCE_v1-2_preprocessed_BERTidx/
    │   └── RxR_VLNCE_v0_enc_xlmr/
    ├── ddppo-models/
    │   └── gibson-2plus-resnet50.pth
    ├── scene_datasets/
    │   └── mp3d/
    ├── wp_pred/
    │   ├── check_cwp_bestdist_hfov79
    │   └── check_cwp_bestdist_hfov90
    └── connectivity_graphs.pkl
```
    
## 3. 模型权重与预训练数据

请按要求下载相应的编码器权重、预测器权重及预训练特征文件，并放置在指定目录下。

### 3.1 编码器与组件权重

| 模型组件 | 下载链接 | 目标存放路径 |
| --- | --- | --- |
| **Waypoint Predictor (R2R-CE)** | [[原项目链接](https://drive.google.com/file/d/1goXbgLP2om9LsEQZ5XvB0UpGK4A5SGJC/view)] | `data/wp_pred/check_cwp_bestdist_hfov90` |
| **Waypoint Predictor (RxR-CE)** | [[原项目链接](https://drive.google.com/file/d/1LxhXkise-H96yMMrTPIT6b2AGjSjqqg0/view)] | `data/wp_pred/check_cwp_bestdist_hfov63` |
| **BERT 权重** | [Huggingface](https://huggingface.co/google-bert/bert-base-uncased/tree/main) | `bert_config/bert-base-uncased` |
| **RGB 编码器 (ViT-B32)** | [Huggingface](https://huggingface.co/jinaai/clip-models/blob/main/ViT-B-32.pt) | `.cache/clip/ViT-B-32.pt` |
| **Depth 编码器 (ResNet50)** | [Gibson Pretrained](https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ddppo/ddppo-models/gibson-2plus-resnet50.pth) | `data/pretrained_models/ddppo-models/gibson-2plus-resnet50.pth` |

### 3.2 预训练数据 (Pretraining Data)

* **R2R 预训练数据**: [下载链接](https://www.dropbox.com/scl/fo/4iaw2ii2z2iupu0yn4tqh/AP2waOdlwdbJE5sUti2557U/R2R?dl=0&rlkey=88khaszmvhybxleyv0a9bulyn&subfolder_nav_tracking=1) -> 存至 `pretrain_src/datasets/R2R`
* **预计算视觉特征**: [下载链接](https://drive.google.com/file/d/1D3Gd9jqRfF-NjlxDAQG_qwxTIakZlrWd/view) -> 存至 `pretrain_src/datasets/img_features`
* **LXMERT 预训练权重**:

  ```bash
  cd pretrain_src
  mkdir -p datasets/pretrained/LXMERT
  wget https://nlp.cs.unc.edu/data/model_LXRT.pth -P datasets/pretrained/LXMERT
  ```

### 3.3 最终预训练权重 (Pretrained Weights)

如果你希望跳过预训练阶段，可直接下载已提供的预训练权重：
* **下载链接**: [百度网盘](https://pan.baidu.com/s/1oTmRkuj6syTmI6kE78k0JQ) (提取码: `vfsh`)
* **存放路径**: `pretrained/ETP/model_step_82500.pt`

## 4. 代码运行

### 4.1 预训练 (Pretraining)

1. **修改引用路径**：确保相关数据加载路径与本地配置一致。
2. **修改 GPU 数量**：根据实际硬件条件按需调整分布式训练的 GPU 数量。
3. **启动预训练**：该步骤主要执行 MLM (Masked Language Modeling) 和 SAP 两个预训练任务。

   ```bash
   CUDA_VISIBLE_DEVICES=0 bash pretrain_src/run_pt/run_r2r.bash 233
   ```

### 4.2 微调 (Finetuning)

1. **配置预训练权重路径**：将脚本中的加载路径指向 `pretrained/ETP/model_step_82500.pt`。
2. **配置 GPU 数量**：通过修改脚本中的 `nproc_per_node` 参数来指定 GPU 数量。
3. **启动微调**：
   **注**：单张 RTX 4090 显卡完成微调大约需要 1.5 天。

   ```bash
   CUDA_VISIBLE_DEVICES=0 bash pretrain_src/run_pt/run_r2r.bash 2333
   ```

### 4.3 测试与评估 (Testing)

*(在此处补充测试相关的执行命令与参数说明)*
