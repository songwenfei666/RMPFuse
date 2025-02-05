# RPFuse

**1. Data Preparation**
Download the MSRS dataset from [this link](https://github.com/Linfeng-Tang/MSRS) and place it in the folder ``'./Datasets/train/'``.

**2. Pre-Processing**

Run 
```
python preprocessing.py
``` 
and the processed training dataset is in ``'./data/MSRS_train_imgsize_128_stride_200.h5'``.

**3. RMPFuse Training**

Run 
```
python Train.py
``` 
and the trained model is available in ``'./model/'``.

