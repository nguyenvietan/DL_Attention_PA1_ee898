import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as td
import torch

BATCH_SIZE = 128

def dataLoader(is_train=True, batch_size=BATCH_SIZE, shuffle=True):
        if is_train:
            trans = [transforms.RandomHorizontalFlip(),
                     transforms.RandomCrop(32, padding=4),
                     transforms.ToTensor(),
                     transforms.Normalize(mean=[n/255.
                        for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]])]
            trans = transforms.Compose(trans)
            train_set = td.CIFAR100('data2', train=True, download=True, transform=trans)
            size = len(train_set)
            train_loader = torch.utils.data.DataLoader(
                            train_set, batch_size=batch_size, shuffle=shuffle)
        else:
            trans = [transforms.ToTensor(),
                     transforms.Normalize(mean=[n/255.
                        for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]])]
            trans = transforms.Compose(trans)
            test_set = td.CIFAR100('data2', train=False, download=True, transform=trans)
            size = len(test_set)
            train_loader = torch.utils.data.DataLoader(
                            test_set, batch_size=batch_size, shuffle=shuffle)

        return train_loader, size

if __name__ == '__main__':
    trainloader, size_train = dataLoader()
    testloader, size_test = dataLoader(is_train=False)

    print (size_train)
    print (size_test)
