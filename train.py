import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

import dataLoader 
import models

def parser():
    parser = argparse.ArgumentParser(description='ResNet Training')
    parser.add_argument('-a', '--arch', default='resnet50', metavar='ARCH', help='network architecture')
    parser.add_argument('-lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('-e', '--epoch', default=100, type=int, metavar='N', help='number of epoches')    
    parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size. default: 128')
    return parser.parse_args()


def eval(model, dataset, size, device, batch_size=dataLoader.BATCH_SIZE):
    """ double-check this function!!!"""
    model.eval()
    correct_1 = 0.0
    correct_5 = 0.0
    with torch.no_grad():
        for data in dataset:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            _, pred = outputs.topk(5, 1, largest=True, sorted=True)
            labels = labels.view(labels.size(0), -1).expand_as(pred)
            correct = (pred == labels).float()
            correct_5 += correct[:, :5].sum()
            correct_1 += correct[:, :1].sum()
    return (1-correct_5/size).item(), (1-correct_1/size).item()

def select_model(model_name):
    # Baseline
    if model_name == 'resnet50':
        return models.resnet.resnet50(num_classes=100)
    if model_name == 'resnet34':
        return models.resnet.resnet34(num_classes=100)
    
    # SE
    if model_name == 'se_resnet50':
        return models.se_resnet.se_resnet50(num_classes=100)
    if model_name == 'se_resnet34':
        return models.se_resnet.se_resnet34(num_classes=100)
    
    # BAM
    if model_name == 'bam_resnet50':
        return models.bam_resnet.bam_resnet50(num_classes=100)
    if model_name == 'bam_resnet50_c':
        return models.bam_resnet_c.bam_resnet50_c(num_classes=100)
    if model_name == 'bam_resnet50_s':
        return models.bam_resnet_s.bam_resnet50_s(num_classes=100)

    if model_name == 'bam_resnet34':
        return models.bam_resnet.bam_resnet34(num_classes=100)
    if model_name == 'bam_resnet34_c':
        return models.bam_resnet_c.bam_resnet34_c(num_classes=100)
    if model_name == 'bam_resnet34_s':
        return models.bam_resnet_s.bam_resnet34_s(num_classes=100)

    # CBAM
    if model_name == 'cbam_resnet50':
        return models.cbam_resnet.cbam_resnet50(num_classes=100)
    if model_name == 'cbam_resnet50_c':
        return models.cbam_resnet_c.cbam_resnet50_c(num_classes=100)
    if model_name == 'cbam_resnet50_s':
        return models.cbam_resnet_s.cbam_resnet50_s(num_classes=100)

    if model_name == 'cbam_resnet34':
        return models.cbam_resnet.cbam_resnet34(num_classes=100)
    if model_name == 'cbam_resnet34_c':
        return models.cbam_resnet_c.cbam_resnet34_c(num_classes=100)
    if model_name == 'cbam_resnet34_s':
        return models.cbam_resnet_s.cbam_resnet34_s(num_classes=100)

def save_checkpoint():
	pass

if not os.path.exists('./checkpoints'):
    os.mkdir('./checkpoints')

if __name__ == '__main__':
        
    args = parser()
    print args
    print "Model: ", args.arch

    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')
    save_path = './checkpoints/' + args.arch
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    trainloader, size_train = dataLoader.dataLoader()
    testloader, size_test = dataLoader.dataLoader(is_train=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = select_model(args.arch) 
#   if torch.cuda.device_count() > 1:
#       print("Let's us e", torch.cuda.device_count(), "GPUs!")
#       model = nn.DataParallel(model)
    model.to(device) # model-> GPU before declaring optimizer
    
    lr = args.learning_rate
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    e1_train = []
    e5_train = []

    e1_test = []
    e5_test = []
    
    e1_best = 1.0
    e5_best = 1.0

    start_time = time.time()

    for epoch in range(args.epoch+1):
        model.train() 
        if ((epoch+1) % 30 == 0):
            lr /= 10
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        running_loss = 0.0
        cnt = 0

        for data in trainloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            cnt += 1
        avg_loss = running_loss/cnt
       
        e5_tr, e1_tr = eval(model, trainloader, size_train, device) 
        e5_te, e1_te = eval(model, testloader, size_test, device)

        if e1_te < e1_best:
            e1_best = e1_te
        if e5_te < e5_best:
            e5_best = e5_te

        e1_train.append(e1_tr)
        e5_train.append(e5_tr)
        e1_test.append(e1_te)
        e5_test.append(e5_te)

        print('[Epoch %3d], lr: %.5f, Loss: %.3f | top1-e-train: %.3f, top5-e-train: %.3f | top1-e-test: %.3f, top5-e-test: %.3f' % \
                (epoch, lr, avg_loss, e1_tr, e5_tr, e1_te, e5_te))
        
        if epoch % 10 == 0:
            torch.save({
                        'model_state_dict': model.state_dict(),
                        'e1_train': e1_train, 
                        'e5_train': e5_train,
                        'e1_test': e1_test,
                        'e5_test': e5_test
                        }, os.path.join(save_path, args.arch + '_epoch{}.pth'.format(epoch)))
    
    print "--------- Finish Training --------"
    print "Model: ", args.arch
    print "Top1E: ", e1_best
    print "Top5E: ", e5_best
    print '\n'
    print "e1_train: ", e1_train
    print "e5_train: ", e5_train
    print '\n'
    print "e1_test: ", e1_test
    print "e5_test: ", e5_test
    print "\n"
    print 'TIME: ', time.time() - start_time

    print '--------------- DONE!!! ----------------------------\n\n'

