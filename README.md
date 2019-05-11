## Implementation of Attention Modules in ResNet50, ResNet34
### To train a network
```sh
$python  train.py [-h] [-a ARCH] [-lr LR] [-e N] [-b N]
ResNet Training
optional arguments:
  -h, --help                    show this help message and exit
  -a ARCH, --arch ARCH          network architecture 
                                (resnetX, se_resnetX, bam_resnetX_c, bam_resnetX_c, 
                                bam_resnetX_s, cbam_resnetX_c, cbam_resnetX_c, 
                                cbam_resnetX_s) with X = 34 or 50
  -lr LR, --learning-rate LR    initial learning rate
  -e N, --epoch N               number of epoches
  -b N, --batch-size N          mini-batch size. default: 128
```
### Examples
Train the SE-ResNet50 with 100 epochs, initial learning rate 0.1:
```sh
$python train.py --arch se_resnet50 --learning-rate 0.1 --epoch 100 
```
Train the CBAM-ResNet34 with 150 epochs, initial learning rate 0.1:
```sh
$python train.py --arch cbam_resnet34 --learning-rate 0.1 --epoch 150 
```
# Results obtained by Google Colab
The folder **results** contains the runtime of my training processes using Google Colab.  