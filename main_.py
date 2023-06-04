import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import timm
import random
import numpy as np
from tqdm import tqdm

from torchvision.models import resnet18, ResNet18_Weights

from sklearn.metrics import classification_report
from matplotlib import pyplot as plt

# What to do to maximize accuracy
# transform adjust
# lr schduler tuning
#  

model_num = 1 
total_epoch = 100 
lr = 0.0001
best_acc = 0.0
log_train_loss = []
log_train_acc = []
log_test_loss = []
log_test_acc = []

for s in range(model_num):
    # fix random seed
    seed_number = s
    random.seed(seed_number)
    np.random.seed(seed_number)
    torch.manual_seed(seed_number)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Check if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Define the data transforms
    transform_train = transforms.Compose([
        transforms.Resize(256),
        # transforms.Resize(224),
        # transforms.RandomCrop(224),
        # transforms.AutoAugment(policy = transforms.AutoAugmentPolicy.IMAGENET, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.AutoAugment(policy = transforms.AutoAugmentPolicy.CIFAR10, interpolation=transforms.InterpolationMode.BILINEAR),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        # transforms.Resize(224),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=16)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=16)

    # Define the ResNet-18 model with pre-trained weights
    model = timm.create_model('resnet18', pretrained=True, num_classes=10)
    # model = resnet18(ResNet18_Weights.IMAGENET1K_V1, zero_init_residual=True)
    # model.fc = nn.Linear(512, 128)
    # model.fc2 = nn.Linear(128, 10)
    # print(model)
    # exit()
    model = model.to(device)  # Move the model to the GPU

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # optimizer = optim.AdamW(model.parameters(), lr=lr)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Define the learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    def train():
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        # print("==========================================")
        for i, data in enumerate(tqdm(trainloader)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # Move the input data to the GPU
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # print(loss)
            loss.backward()
            optimizer.step()

            train_result = classification_report(np.array(labels.cpu()), np.array(torch.argmax(outputs, dim=1).cpu()), output_dict=True, zero_division= 0)
            train_acc += train_result['accuracy']*100
            train_loss += loss.item()

        print("epoch: ", epoch)
        # print('[%d, %5d] loss: %.6f' % (epoch + 1, i + 1, train_loss / 100))
        print("epoch: {} train_loss: {:.4f} train_accuracy: {:.4f}%".format(epoch, train_loss/len(trainloader), train_acc/len(trainloader)))
        log_train_loss.append(train_loss/len(trainloader))
        log_train_acc.append(train_acc/len(trainloader))
    
                
    def test():
        model.eval()
        
        # Test the model
        correct = 0
        total = 0
        test_acc = 0.0
        test_loss = 0.0

        with torch.no_grad():
            for data in tqdm(testloader):
                images, labels = data
                images, labels = images.to(device), labels.to(device)  # Move the input data to the GPU
                outputs = model(images)
                tloss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                test_result = classification_report(np.array(labels.cpu()), np.array(torch.argmax(outputs, dim=1).cpu()), output_dict=True, zero_division= 0)
                test_acc += test_result['accuracy']*100
                test_loss += tloss.item()

        print("epoch: ", epoch)
        # print('[%d, %5d] loss: %.6f' % (epoch + 1, i + 1, train_loss / 100))
        print("epoch: {} test_loss: {:.4f} test_accuracy: {:.4f}%".format(epoch, test_loss/len(testloader), test_acc/len(testloader)))
        
        log_test_loss.append(test_loss/len(testloader))
        log_test_acc.append(test_acc/len(testloader))

        # print('Accuracy of the network on the 10000 test images: %f %%' % (100 * correct / total))
        global best_acc
        if best_acc < (100 * correct / total):
            best_acc = (100 * correct / total)
        print('Best Accuracy: %f %%' % (best_acc))

    # Train the model
    for epoch in range(total_epoch):
        train()
        test()
        scheduler.step()

    print('Finished Training')

    # Save the checkpoint of the last model
    PATH = './resnet18_cifar10_{:.4f}.pth'.format(best_acc) 
    torch.save(model.state_dict(), PATH)
    print("model saved ",PATH)

epochs = range(1, total_epoch+1)
plt.figure(1)
plt.plot(epochs, log_train_loss, label='train_loss', color='green')
plt.plot(epochs, log_test_loss, label='test_loss', color='blue')
plt.title('resnet18_loss per epoch')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train_loss', 'test_loss'])
plt.figure(1).savefig('acc_resnet18_cifar10_{:.4f}.png'.format(best_acc))

print("log_train_acc: ", log_train_acc)
print("log_train_loss: ", log_train_loss)

plt.figure(2)
plt.plot(epochs, log_train_acc, label='train_acc', color='green')
plt.plot(epochs, log_test_acc, label='test_acc', color='blue')
plt.title('resnet18_accuracy per epoch')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train_acc', 'test_acc'])
plt.figure(2).savefig('loss_resnet18_cifar10_{:.4f}.png'.format(best_acc))

print("log_test_acc: ", log_test_acc)
print("log_test_loss: ", log_test_loss)