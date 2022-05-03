from ctypes.wintypes import HACCEL
from xml.dom import WrongDocumentErr
from tensorboard import summary
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

NUM_EPOCHS = 25
BATCH_SIZE = 100
LEARNING_RATE = 0.001
NUM_CLASSES=27

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_dataset = torchvision.datasets.EMNIST(root='./data/',
                                           train=True, 
                                           split='letters',
                                           transform=transforms.ToTensor(),
                                           download=True)


test_dataset = torchvision.datasets.EMNIST(root='./data/',
                                          split='letters',
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=BATCH_SIZE, 
                                          shuffle=False)




def conv3x3(in_planes, out_planes, stride=1):
    #3x3 convolutions and padding
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale):
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # because EMNIST is already 1x1 here:
        # disable avg pooling
        #x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



def resnet18(num_classes):
    #creates resnet model
    model = ResNet(block=BasicBlock, 
                   layers=[2, 2, 2, 2],
                   num_classes=num_classes,
                   grayscale=True)
    return model

model = resnet18(NUM_CLASSES).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)



def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
# Train the model_conv
total_step = len(train_loader)
train_acc_list, test_acc_list = [], []
train_loss_list, test_loss_list = [], []
curr_lr = LEARNING_RATE
wrong_dict = dict()
for x in range(NUM_CLASSES):
    wrong_dict[x]=0
for epoch in range(NUM_EPOCHS):
    train_loss=0.0
    test_loss=0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if (i+1) % 100 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                   .format(epoch+1, NUM_EPOCHS, i+1, total_step, loss.item()))


    # Decay learning rate
    if (epoch+1) % 5 == 0:
        curr_lr = curr_lr*(2/3)
        update_lr(optimizer, curr_lr)

    # Test the model
    
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            test_loss += loss.item()
            total += labels.size(0)
            wrong_idx = (predicted != labels).nonzero()[:, 0]
            if(epoch>20): #finds wrong predictions
                for idx in wrong_idx:
                    wrong_dict[labels[idx].item()]+=1

            correct += (predicted == labels).sum().item()
    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
    test_acc_list.append(100 * correct / total)
    train_loss_list.append(train_loss/len(train_loader))
    test_loss_list.append(test_loss/len(test_loader))
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the train images: {} %'.format(100 * correct / total))
    train_acc_list.append(100 * correct / total)

    model.train()
    

def getLetter(index ): #returns letter from index
    for letter, idx in train_dataset.class_to_idx.items():
        if idx == index:
            return letter
    return 'N/A'

print(wrong_dict)

ROW_IMG = 10
N_ROWS = 5
fig = plt.figure()
for index in range(1, ROW_IMG * N_ROWS + 1): #shows sample predictions
    
    plt.subplot(N_ROWS, ROW_IMG, index)
    plt.axis('off')
    plt.imshow(train_dataset.data[index], cmap='gray_r')
    
    with torch.no_grad():
        model.eval()
        abc = train_dataset[index][0].cuda()
        probs = model(abc.unsqueeze(0))
        letter = getLetter(torch.argmax(probs))


    title = f'{letter}'
    
    plt.title(title, fontsize=12)
fig.suptitle('Resnet-18 - predictions');
fig.show()
plt.show()


plt.plot(train_acc_list, '-b', label='train acc')
plt.plot(test_acc_list, '-r', label='test acc')
plt.legend()
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.xticks(rotation=60)
plt.title('Accuracy ~ Epoch')
# plt.savefig('assets/accr_{}.png'.format(cfg_idx))
plt.show()

plt.plot(train_loss_list, '-b', label='train loss')
plt.plot(test_loss_list, '-r', label='test loss')
plt.legend()
plt.ylabel('Loss Function')
plt.xlabel('Epoch')
plt.xticks(rotation=60)
plt.title('Loss Function ~ Epoch')
#plt.savefig('assets/loss_{}.png'.format(cfg_idx))
plt.show()



# Save the model_conv checkpoint
torch.save(model.state_dict(), 'model_resnet18.ckpt')

