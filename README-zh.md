# 基于 LSTM 的行为识别

## 依赖

Python 3.6.5 (Anaconda 5.2.0):

[https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/)

pytorch 0.4.1 (CUDA 9.0):

[https://pytorch.org/](https://pytorch.org/)

## 数据处理

本项目的默认数据结构如下

```
~/
  data/
    train_data/
      .../ (directories of class names)
        .../ (directories of video names)
          ... (jpg files)
    valid_data/
      .../ (directories of class names)
        .../ (directories of video names)
          ... (jpg files)
    save_model/
      save_100.pth
```

因为项目使用的是视频帧对应的图片，所以需要在训练和测试前进行数据处理。本项目转换前的视频训练集数据结构如下：

```
train_data/
      .../ (directories of class names)
        ... (avi files)
```

使用 `utils` 文件夹下的 `video2jpg.py` 进行转换：

```
python video2jpg.py video_root_directory image_root_directory
```

转换时考虑到训练集的训练时间和可能会有的 out of memory 问题，增加了处理函数的输入参量，可以指定最大文件尺寸跳过了部分较大的文件：

```python
# skip large files
if os.path.getsize(video_file_path) > maxSize * 1000:
    continue
```

虽然数据集经过处理已经可以作为神经网络的输入，但是还需要将训练集进行划分，分为训练集和交叉验证集，所以编写了 `split_data.py`：

```
python split_data.py train_root_directory valid_root_directory
```

## 神经网络的设计

本项目基于 CNN 和 LSTM，其中 CNN 部分使用了 PyTorch repo 的预训练网络，默认使用 AlexNet 的微调网络。

整个神经网络的结构可以认为是 CNN 特征提取器 和 LSTMs 的组合，默认情况下设置 LSTMs 的层数为一层。 

![](imgs/lstm.jpg)

## 神经网络实现

TODO

## 训练过程

### step.1 训练代码验证

正式进行训练前，需要编写正确的训练代码。

该阶段从原数据集中抽取了五个分类，每个分类选择两个较小的训练数据，所以数据集的大小为 10。

因为代码参考了部分结构简单的 LSTMs，所以修改后的代码必须保证：

1. 训练、验证、预测以及模型的保存和加载都可以正常进行，各阶段不会出现报错
2. 随着训练的进行，在训练集上的 loss 要有明显降低，也就是训练要收敛

### step.2 开始正式训练

这个阶段出现了一些第一阶段没有出现的问题：

1. 训练集中存在一些较大的数据，在网络训练时会使显存溢出
2. 训练的速度太慢，没法短期内对训练的效果进行评估

因为以前打游戏用的 GPU 刚好卖掉，实验室的计算平台还没有采购，硬件方面较为短缺。

为了解决这些问题，对网络进行了优化并删除了中间量，使用了一些训练框架的新特性，但是收效甚微，最后只能采取临时升级硬件的方法：

1. 在云主机平台上进行训练
2. 购置新的 GPU

在收购到新的 GPU 之前，使用 P40 的 GPU 云主机进行了一段时间的训练，但是发现收敛困难。

### step.3 网络的优化和继续训练

TODO
