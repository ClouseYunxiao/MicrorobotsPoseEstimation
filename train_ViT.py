import torch
import torch.nn as nn
from torchvision import transforms
import time
from Probe_Dataset import MyDataset
from matplotlib import pyplot as plt
from ViTNET import ViTNET

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

num_epochs = 50
batch_size = 16
learning_rate = 1e-3
global X_liner# Define the feature collector


# The path of both images and labels
img_dir = '../preprocessed_dataset/trainset/frames/'
data_list = '../preprocessed_dataset/trainset/data.csv'

# The pre-processing of the dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize((0.485), (0.229))

])

# Split the dataset into train_set and val_set
data_set = MyDataset(data_list,img_dir,transform=transform)
train_size = int(len(data_set) * 0.8)
val_size = len(data_set) - train_size
train_set, val_set = torch.utils.data.random_split(data_set, [train_size, val_size])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)


model =  ViTNET()
model.to(device)
print(model)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Set a learning rate decreased function 
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Define the test function as the val part
def test(model,val_loader):
    
    model.eval()
    with torch.no_grad():
        test_loss = 0
        total = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            #++++++++++++++++++++++++++++++++++++++++++++++++++++
            # Empty the feature collector
            global X_liner
            X_liner=torch.empty((batch_size,0),device=device)
            #++++++++++++++++++++++++++++++++++++++++++++++++++++
            outputs = model(images)
            labels = torch.tensor(labels,dtype=torch.float)
            labels = labels.reshape(outputs.shape)
            val_loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            print_loss = val_loss.data.item()
            test_loss += print_loss
            total += labels.size(0)
        avgloss = test_loss / len(val_loader)
        print('\nVal set: Average loss: {:.4f}\n'.format(avgloss))

Train_loss = []
# Training 
total_step = len(train_loader)
curr_lr = learning_rate
for epoch in range(num_epochs):
    in_epoch = time.time()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        #++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Empty the feature collector
        X_liner=torch.empty((batch_size,0),device=device)
        #++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Forward pass
        outputs = model(images)
        labels = torch.tensor(labels,dtype=torch.float)
        labels = labels.reshape(outputs.shape)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        Train_loss.append(loss.item())
        if (i + 1) % 100 == 0:
            print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
    test(model,val_loader)
    out_epoch = time.time()
    print(f"use {(out_epoch-in_epoch)//60}min{(out_epoch-in_epoch)%60}s")
    if (epoch + 1) % 3 == 0:
        curr_lr = curr_lr * 0.5
        update_lr(optimizer, curr_lr)

test(model,train_loader)
torch.save(model, './models/ViTnet.pth')

# Plot the train loss with training iterations
figure_path = './training_loss/ViT_train_loss.png'
plt.figure(figsize=(10,5))
plt.title("The training loss of Vision Transformer")
plt.plot(Train_loss,label="Training loss")
plt.xlabel("iterations")
plt.ylabel("loss")
plt.legend()
plt.savefig(figure_path)