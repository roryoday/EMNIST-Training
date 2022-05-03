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
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_dataset = torchvision.datasets.EMNIST(root='./data/',
                                           train=True, 
                                           split='letters',
                                           transform=transform,
                                           download=True)


test_dataset = torchvision.datasets.EMNIST(root='./data/',
                                          split='letters',
                                          train=False, 
                                          transform=transform)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=BATCH_SIZE, 
                                          shuffle=False)

VGG_architectures = {
    'VGG16': [64, 64, 'pool', 128, 128, 'pool', 256, 256, 256, 'pool', 512, 512, 512, 'pool', 512, 512, 512, 'pool']
}
class VGG(nn.Module):
  def __init__(self, num_classes, in_planes = 1, ):
    super(VGG, self).__init__()
    self.in_planes = in_planes
    self.convs = self.stack_layers(VGG_architectures['VGG16'])

    self.fully_connected = nn.Sequential(
        nn.Linear(512, 128),
        nn.ReLU(inplace = True),
        nn.Dropout(0.2),
        nn.Linear(128, 128),
        nn.ReLU(inplace = True),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes)
    )

  def stack_layers(self, architecture):
    in_planes = self.in_planes
    stack = []

    for layer in architecture:
      if type(layer) == int:
        out = layer
        stack += [nn.Conv2d(in_planes, out, kernel_size = 3, stride = 1, padding = 1),
                  nn.BatchNorm2d(layer),
                  nn.ReLU(inplace = True)]
        in_planes = layer
      else:
        stack += [nn.MaxPool2d(kernel_size = 2)]
      
    return nn.Sequential(*stack)

  def forward(self, x):
    x = self.convs(x)
    x = x.reshape(x.shape[0], -1)
    x = self.fully_connected(x)

    return x

        

    
    
model_conv = VGG(27).to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_conv.parameters(), lr=LEARNING_RATE)
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
# Train the model_conv
total_step = len(train_loader)
test_acc_list, train_acc_list = [], []
train_loss_list, test_loss_list = [], []
curr_lr = LEARNING_RATE

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
torch.save(model_conv.state_dict(), 'model_lenet.ckpt')