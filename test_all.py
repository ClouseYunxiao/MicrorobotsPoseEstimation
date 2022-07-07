from cProfile import label
import torch
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
from Probe_Dataset import MyDataset


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


for i in range(4,5):
    # The path of test set 
    img_dir = '../preprocessed_dataset/set' + str(i) + '/frames/'
    data_list = '../preprocessed_dataset/set' + str(i) + '/data.csv'

    # Resize the images and transit it into dataloader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize((0.485), (0.229))
    ])
    test_set = MyDataset(data_list,img_dir,transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    # Load the trained model
    model1 = torch.load('./models/resnet152_model.pth')
    model2 = torch.load('./models/Densenet121_model.pth')
    torch.no_grad()
    target_all = []
    prediction1 = []
    prediction2 = []
    # Define the process of testing data
    def test(model1,model2, device, test_loader):
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = Variable(data).to(device), Variable(target).to(device) # The problem of the lost of the accuaracy
            label_tmp = target.cpu().detach().numpy()
            label = label_tmp[0]
            target_all.append(label)
            output1 = model1(data)
            output2 = model2(data)
            predict_tmp1 = output1[0].cpu().detach().numpy()
            predict_tmp2 = output2[0].cpu().detach().numpy()
            predict1 = predict_tmp1[0] + 0.5
            predict2 = predict_tmp2[0] + 0.5
            prediction1.append(predict1)
            prediction2.append(predict2)
        return target_all, prediction1, prediction2
     
    Labels, Predictions1, Predictions2=test(model1, model2, DEVICE, test_loader)
    figure_path = './results/set_' + str(i) + '_comparsion_GT_Predict.png'
    plt.figure(figsize=(10,6))
    #plt.title("The Ground Truth and the Predictions")
    plt.plot(Labels,label="Ground Truth")
    plt.plot(Predictions2,label='Model with Bayesian Optimization', color='#FF0000')
    plt.plot(Predictions1,label='Model without Bayesian Optimization')
    plt.legend(loc='upper left', fontsize=12)
    plt.xlabel("Number of Frames", fontsize=20)
    plt.ylabel("z[um]", fontsize=20)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.savefig(figure_path)