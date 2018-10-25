# 基于 LSTM 的行为识别

## 依赖

Python 3.6.5 (Anaconda 5.2.0):

[https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/)

pytorch 0.4.1 (CUDA 9.0):

[https://pytorch.org/](https://pytorch.org/)

## 数据结构

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
video_data/
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

## 开始训练

输入指令以默认参数开始训练：

```
python train.py data
```

可选参数：

```
--model DIR           path to model
--epochs N            manual epoch number (default: 90)
--lr LR               initial learning rate (default: 0.01)
--optim OPTIM         optimizer (default: rmsprop)
--momentum M          momentum (default: 0.9)
--lr_step LR_STEP     learning rate decay frequency (default: 30)
--arch ARCH           model architecture (default: alexnet)
--workers N           number of data loading workers (default: 8)
--batch-size N        mini-batch size (default: 1)
--weight-decay W      weight decay (default: 1e-4)
--lstm-layers LSTM    number of lstm layers (default: 1)
--hidden-size HIDDEN  output size of LSTM hidden layers (default: 512)
--fc-size FC_SIZE     size of fully connected layer before LSTM (default:
                      1024)
```

输入指令开始测试：

```
python test.py data/save_model/model_best.pth.tar test_data
```

## 神经网络的设计

本项目基于 CNN 和 LSTM，其中 CNN 部分使用了 PyTorch repo 的预训练网络，默认使用 AlexNet 的微调网络。

整个神经网络的结构可以认为是 CNN 特征提取器 和 LSTMs 的组合，默认情况下设置 LSTMs 的层数为一层。 

<center>

![](/imgs/lstm.jpg)

</center>

## 参考

[https://github.com/siqinli/GestureRecognition-PyTorch.git](https://github.com/siqinli/GestureRecognition-PyTorch.git)

[https://github.com/chaoyuaw/pytorch-coviar.git](https://github.com/chaoyuaw/pytorch-coviar.git)
