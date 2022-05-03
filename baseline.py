from ctypes.wintypes import HACCEL
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

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
print(torchvision.datasets.EMNIST.classes_split_dict)
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

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
  def __init__(self):
    super(ConvNet, self).__init__()

    self.conv1 = nn.Conv2d(1, 16, 3)
    self.pool1 = nn.MaxPool2d(2,2)

    self.conv2 = nn.Conv2d(16, 32 ,3)
    self.pool2 = nn.MaxPool2d(2,2)

    self.fc1 = nn.Linear(800, 256)
    self.fc2 = nn.Linear(256, 27)

  def forward(self, x):
    x = self.pool1(F.relu(self.conv1(x)))
    x = self.pool2(F.relu(self.conv2(x)))
    x = x.view(-1, 800)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x


        

    
    
model_conv = ConvNet().to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_conv.parameters(), lr=LEARNING_RATE)

# Train the model_conv
total_step = len(train_loader)
test_acc_list, train_acc_list = [], []
train_loss_list, test_loss_list = [], []
curr_lr = LEARNING_RATE

def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
for epoch in range(NUM_EPOCHS):
    train_loss=0.0
    test_loss=0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model_conv(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, NUM_EPOCHS, i+1, total_step, loss.item()))
    if (epoch+1) % 5 == 0:
        curr_lr = curr_lr*(2/3)
        update_lr(optimizer, curr_lr)
    # Test the model_conv
    model_conv.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model_conv(images)
            _, predicted = torch.max(outputs.data, 1)
            test_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model_conv on the 10000 test images: {} %'.format(100 * correct / total))
        test_acc_list.append(100 * correct / total)
        train_loss_list.append(train_loss/len(train_loader))
        test_loss_list.append(test_loss/len(test_loader))
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model_conv(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model_conv on the train images: {} %'.format(100 * correct / total))
        train_acc_list.append(100 * correct / total)

        
        
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
torch.save(model_conv.state_dict(), 'model_conv.ckpt')