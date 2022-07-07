import torch.optim as optim
import torch
import time
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.autograd import Variable
from Probe_Dataset import MyDataset
from torchvision.models import densenet201
import matplotlib.pyplot as plt
import skorch
# Set the training parameters
modellr = 1e-3
BATCH_SIZE = 16
EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# The path of micro-robots images
img_dir = '../preprocessed_dataset/trainset/frames/'
data_list = '../preprocessed_dataset/trainset/data.csv'

# The pre-processing of the dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize((0.485), (0.229))

])

# Split the dataset into train_set and val_set
data_set = MyDataset(data_list,img_dir,transform=transform)
train_size = int(len(data_set) * 0.7)
val_size = len(data_set) - train_size
train_set, val_set = torch.utils.data.random_split(data_set, [train_size, val_size])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

# Load the model as densenet and put it into gpu device
model = densenet201()
# Re-fit the model from classfier to regression
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, 1)

# Print the densenet
print(model)
model.to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=modellr)

Train_loss = []

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
    modellrnew = modellr * (0.1 ** (epoch // 5))
    print("lr:", modellrnew)
    for param_group in optimizer.param_groups:
        param_group['lr'] = modellrnew
 
 
# Define the train process
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    sum_loss = 0
    total_num = len(train_loader.dataset)
    print(total_num, len(train_loader))
    in_epoch = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).to(device), Variable(target).to(device) # The problem of the lost of the accuaracy
        output = model(data)
        target = torch.tensor(target,dtype=torch.float)
        target = target.reshape(output.shape)
        train_loss = criterion(output, target)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        print_loss = train_loss.data.item()
        sum_loss += print_loss
        Train_loss.append(train_loss.item())
        if (batch_idx + 1) % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), train_loss.item()))
    ave_loss = sum_loss / len(train_loader)
    print('epoch:{},loss:{}'.format(epoch, ave_loss))
    out_epoch = time.time()
    print(f"use {(out_epoch-in_epoch)//60}min{(out_epoch-in_epoch)%60}s")
 
 
# Define the validation process
def val(model, device, test_loader):
    model.eval()
    test_loss = 0
    total_num = len(test_loader.dataset)
    print(total_num, len(test_loader))
    in_epoch = time.time()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = Variable(data).to(device), Variable(target).to(device)
            output = model(data)
            target = target.reshape(output.shape)
            val_loss = criterion(output, target)
            _, pred = torch.max(output.data, 1)
            print_loss = val_loss.data.item()
            test_loss += print_loss
        avgloss = test_loss / len(test_loader)
        print('\nVal set: Average loss: {:.4f}\n'.format(avgloss))
    out_epoch = time.time()
    print(f"use {(out_epoch-in_epoch)//60}min{(out_epoch-in_epoch)%60}s")
 
 
# Train and valid in each epoch
for epoch in range(1, EPOCHS + 1):
    adjust_learning_rate(optimizer, epoch)
    train(model, DEVICE, train_loader, optimizer, epoch)
    val(model, DEVICE, val_loader)
torch.save(model, './models/Vision_Transformer_model.pth')

# Plot the train loss with training iterations
figure_path = './training_loss/Vision_Transformer_train_loss.png'
plt.figure(figsize=(10,5))
plt.title("The training loss")
plt.plot(Train_loss,label="Training loss")
plt.xlabel("iterations")
plt.ylabel("loss")
plt.legend()
plt.savefig(figure_path)