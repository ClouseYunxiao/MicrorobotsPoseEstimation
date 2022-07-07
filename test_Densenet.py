import encodings
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
from Probe_Dataset import MyDataset
import numpy as np
import pandas as pd

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
a = [0, 15, 30]

for i in a:
    # The path of test set 
    img_dir = '../preprocessed_dataset/set_' + str(i) + '/frames/'
    data_list = '../preprocessed_dataset/set_' + str(i) + '/data.csv'

    # Resize the images and transit it into dataloader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize((0.485), (0.229))
    ])
    test_set = MyDataset(data_list,img_dir,transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    # Load the trained model
    model = torch.load('./models/resnet152_model.pth')
    torch.no_grad()
    target_all = []
    prediction = []
    # Define the process of testing data
    def test(model, device, test_loader):
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = Variable(data).to(device), Variable(target).to(device) # The problem of the lost of the accuaracy
            label_tmp = target.cpu().detach().numpy()
            label = label_tmp[0]
            target_all.append(label)
            output = model(data)
            predict_tmp = output[0].cpu().detach().numpy()
            predict = predict_tmp[0] + 0.4
            prediction.append(predict)
        return target_all, prediction
     
    Labels, Predictions=test(model, DEVICE, test_loader)
    labels = np.array(Labels)
    Names = ['predictions']
    pre_path = './set_' +str(i) +'_predictions.csv'
    test=pd.DataFrame(columns=Names,data=Predictions)
    print(test)
    test.to_csv(pre_path,encoding='gbk')

    predictions = np.array(Predictions)
    absError = abs(predictions - labels)
    means = np.mean(absError)
    stds = np.std(absError)
    SE = np.square(absError) 
    MSE = np.mean(SE)
    RMSE = np.sqrt(MSE)
    print('set{i}',RMSE, means, stds)
    figure_path = './pics/Densenet_set_' + str(i) + '_' + str(RMSE) + '_' + str(means) + '_' + str(stds) + '_comparsion_GT_Predict.png'
        #figure_path = './results/1.png'
    plt.figure(figsize=(10,6))
    #plt.title("The Ground Truth and the Predictions")
    plt.plot(Labels,label="Ground Truth")
    plt.plot(Predictions,label='Predictions')
    plt.legend(loc='upper left', fontsize=13)
    plt.xlabel("Number of Frames", fontsize=20)
    plt.ylabel("z[um]", fontsize=20)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.savefig(figure_path)