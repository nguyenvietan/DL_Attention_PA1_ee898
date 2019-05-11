# run the models with ResNet50 backbone
python train.py --arch resnet50 --epoch 100
python train.py --arch se_resnet50 --epoch 100
python train.py --arch bam_resnet50 --epoch 100
python train.py --arch cbam_resnet50 --epoch 100