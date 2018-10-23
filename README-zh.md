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

本项目基于 CNN 和 LSTM
