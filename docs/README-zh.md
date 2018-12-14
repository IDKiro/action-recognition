# 基于 LSTM 的行为识别

[训练细节](/docs/log.md)

## 依赖

Python 3.6.5 (Anaconda 5.2.0):

[https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/)

pytorch 0.4.1 (CUDA 9.0):

[https://pytorch.org/](https://pytorch.org/)

## 数据结构

输入指令下载示例数据:

```
python utils/download.py
```

或者直接从 [Google Drive](https://drive.google.com/file/d/1SI4mAeupeYQXbRN0zHqtfttULGHpXmw2/view?usp=sharing) 上下载并解压.

本项目的训练数据结构如下：

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

测试数据结构如下：

```
~/
  test_data/
    .../ (directories of class names)
      .../ (directories of video names)
        ... (jpg files)
```

此时训练集的路径为 data，测试集的路径为 test_data，在输入指令时必须指定数据集的路径。

因为项目使用的是视频帧对应的图片，所以需要在训练和测试前进行数据处理。本项目转换前的视频训练集数据结构如下：

```
~/
  video_data/
      .../ (directories of class names)
        ... (avi files)
```

使用 `utils` 文件夹下的 `video2jpg.py` 进行转换：

```
python video2jpg.py video_root_directory image_root_directory
```

虽然数据集经过处理已经可以作为神经网络的输入，但是还需要将训练集进行划分，分为训练集和交叉验证集，所以编写了 `split_data.py`：

```
python split_data.py train_root_directory valid_root_directory
```

## 开始训练

输入指令以默认参数开始训练：

```
python train.py train_data_directory
```

默认可直接使用:

```
python train.py data
```

可选参数：

```
--model DIR           path to model
--arch ARCH           model architecture (default: alexnet)
--lstm-layers LSTM    number of lstm layers (default: 1)
--hidden-size HIDDEN  output size of LSTM hidden layers (default: 512)
--fc-size FC_SIZE     size of fully connected layer before LSTM (default:
                      1024)
--epochs N            manual epoch number (default: 150)
--lr LR               initial learning rate (default: 0.01)
--optim OPTIM         optimizer (default: sgd)
--momentum M          momentum (default: 0.9)
--lr-step LR_STEP     learning rate decay frequency (default: 50)
--batch-size N        mini-batch size (default: 1)
--weight-decay W      weight decay (default: 1e-4)
--workers N           number of data loading workers (default: 8)
```

## 测试

输入指令开始测试：

```
python test.py model_directory test_data_directory
```

默认可直接使用:

```
python test.py data/save_model/model_best.pth.tar data/valid_data
```

## 预测

输入指令开始测试：

```
python predict.py model_directory predict_data_directory
```

默认可直接使用:

```
python test.py data/save_model/model_best.pth.tar data/predict_data
```

## 神经网络的设计

本项目基于 CNN 和 LSTM，其中 CNN 部分使用了 PyTorch 的预训练网络，默认使用 AlexNet。

整个神经网络的结构可以认为是 CNN 特征提取器 和 LSTMs 的组合，默认情况下设置 LSTMs 的层数为一层。 

<div align="center">
  <img src="/imgs/lstm.jpg">
</div>

## 参考

[https://github.com/chaoyuaw/pytorch-coviar.git](https://github.com/chaoyuaw/pytorch-coviar.git)
