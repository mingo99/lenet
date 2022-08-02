from copy import deepcopy
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.quantization as tq
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from LeNet import LeNet
from quant import Quant

def mkdirs(directorys):
    """Make directory"""
    for dir in directorys:
        if not os.path.exists(dir):
            os.makedirs(dir)

def model_save(epoch, model_state_dict, optimizer_state_dict, PATH):
    """To save checkpoint"""
    print("Start saving model...")
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save({
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            }, PATH)
    print("Finish saving model...")

def model_load(model, optimizer, PATH):
    """To load checkpoint"""
    print("Start loading checkpoint!!!\n...")
    if os.path.isdir(PATH):
        try:
            if len(os.listdir(PATH)) > 0:
                checkpoint = torch.load(PATH + os.listdir(PATH)[-1])
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                print('Load checkpoint successfully!!!')
            else:
                print('Can\'t found the .pth file in checkpoint and start from scratch!')
                start_epoch = 0
        except FileNotFoundError:
            print('Can\'t found the .pth file in checkpoint and start from scratch!')
            start_epoch = 0
    else:
        start_epoch = 0
        print("Can\'t find folder'checkpoint' and Start from scratch")
    return start_epoch

def load_data(PATH: str, batch_size):
    """To load MINIST dataset"""
    # 定义数据预处理方式
    transform = transforms.ToTensor()
    # 定义训练数据集
    trainset = datasets.MNIST(root=PATH,train=True,download=True,transform=transform)
    # 定义测试数据集
    testset = datasets.MNIST(root=PATH,train=False,download=True,transform=transform)
    # 定义训练批处理数据
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True)
    # 定义测试批处理数据
    testloader = torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle=False)
    # 返回数据
    return trainloader, testloader

def showdata(data_loader):
    """Show the dataset example"""
    for batch_idx, (data, target) in enumerate(data_loader):
        if batch_idx == 6:
            break
        print(batch_idx)
        print(data.shape)
        print(target.shape)
        print(data.dtype)
        plt.subplot(2, 3, batch_idx + 1)
        plt.tight_layout()
        data= data.reshape(-1,28)
        plt.imshow(data, cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(target))
        plt.xticks([])
        plt.yticks([])
        plt.show()

def epoch_train(net, device, optimizer, criterion, trainloader, epoch):
    """Start LeNet training with MINIST"""
    # 训练
    print("Start Training, LeNet-5!")
    with open(f"./log/log_net{epoch+1:02d}.log", "w")as f:
        print(f'Epoch:{epoch+1}')
        net.train()
        sum_loss = 0.0
        correct = 0
        total = 0
        # 数据读取
        for i, data in enumerate(trainloader):
            length = len(trainloader)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # 梯度清零
            optimizer.zero_grad()
            # forward + backward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # 每训练100个batch打印一次平均loss
            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            print(f"LR:{(optimizer.state_dict()['param_groups'][0]['lr']):.4f} | total_iter:{(i+1+epoch*length)} [iter:{i+1}/469 in epoch:{epoch+1}] | Loss: {(sum_loss/(i + 1)):.03f} | Acc: {(100.*correct/total):.3f}%")
            f.write(f"LR:{(optimizer.state_dict()['param_groups'][0]['lr']):.4f} | total_iter:{(i+1+epoch*length):04d} [iter:{(i+1):03d}/469 in epoch:{(epoch+1):d}] | Loss: {(sum_loss/(i + 1)):.03f} | Acc: {(100.*correct/total):.3f}%")
            f.write('\n')
            f.flush()
    print("Finish Training!!!")

def epoch_test(net, device, testloader, epoch):
    """Test the accurancy"""
    net.eval()
    correct = 0
    total = 0
    with open("./result/acc.txt", "w") as f:
        with torch.no_grad():
            print("Waiting Test!")
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                # 取得分最高的那个类
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print(f"Test Acc:{(100.*correct/total):.2f}")
            f.write(f"EPOCH={epoch+1:02d},TestAcc={(100.*correct/total):.2f}")
            f.write('\n')
            f.flush()

def quant_test(net, device, testloader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        print("Waiting Test!")
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            # 取得分最高的那个类
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print(f"Quant test Acc:{(100.*correct/total):.2f}")

def run():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCH = 10          #遍历数据集次数
    BATCH_SIZE = 128    #批处理尺寸(batch_size)
    LR = 0.01          #学习率

    net = LeNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，通常用于多分类问题上
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    path = '../datasets'
    trainloader, testloader = load_data(path, BATCH_SIZE)

    print(f"Neural Network Module: LeNet \nDatasets: MINIST \nNumber of Epoch: {EPOCH} \nBacth Size: {BATCH_SIZE}")
    mkdirs(['./log', './result', './checkpoint'])
    start_epoch = model_load(net, optimizer, './checkpoint/') 
    # if(start_epoch==EPOCH):
    #     print("The model has trained completely!!!")
    #     sys.exit(0)

    for epoch in range(start_epoch, EPOCH):
        epoch_train(net, DEVICE, optimizer, criterion, trainloader, epoch)
        epoch_test(net, DEVICE, testloader, epoch)
        # 调整学习率
        scheduler.step()
        model_save(epoch, net.state_dict(), optimizer.state_dict(), f'./checkpoint/ckp_net{(epoch+1):02d}.pth')

    quant_net = deepcopy(net)
    print(quant_net)
    quant_net.quantize()
    quant_test(quant_net,DEVICE,testloader)

if __name__ == "__main__":
    run()
