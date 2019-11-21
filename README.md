# Action Recognition

**Three steps** to train your own model for action recognition based on CNN and LSTM.

## Environment

Python 3.7.5 (Anaconda 5.3.0):

[https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/)

PyTorch 1.3.0 (CUDA 10.1):

[https://pytorch.org/](https://pytorch.org/)

## Data

1. Download the dataset ([HMDB51](http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar)):

```
python data/download.py
```

2. Convert videos to images:

```
python data/video2jpg.py
```

## Train

Use the following command to train the model:

```
python train.py
```

## Predict

Use the following command to predict:

```
python predict.py
```

## Network

This project is based on CNNs and LSTMs. 

<div align="center">
  <img src="imgs/lstm.jpg">
</div>


## Tips

> If you download the dataset by yourself, you need to move the `rar` file to `data` folder firstly.
