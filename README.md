# COSONet: Compact Second-Order Network for Video Face Recognition 

This paper has been accepted as ACCV 2018 Oral [[paper]](http://vipl.ict.ac.cn/uploadfile/upload/2018111616133187.pdf)

## Requirements

PyTorch 0.4.1. 

Python packages: pickle, matplotlib, h5py

## Datasets

The training and testing data, trained model are released. You can download from [[BaiduYun]](https://pan.baidu.com/s/1a1VXZ6sBEibLMp88cU8tnQ).


## Test
After unzipping the data in to ./data, run test_ijba.py, and you will get the result on IJB-A of our COSONet (corresponding to the ResNet_34_COSO in Table 2)

TAR = [0.6832107  0.81946075 0.85863197 0.93930644 0.9631115 ] @FAR[0.001, 0.005, 0.01, 0.05, 0.1]


## Train
To train the COSONet by yourself, you can just run train_webface_resnet34_coso.py.


## Contact

If you have any question, be free to contact me. My email is yirong.mao@vipl.ict.ac.cn
