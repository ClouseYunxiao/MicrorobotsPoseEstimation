import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
from Probe_Dataset import MyDataset
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# The path of test set 
for i in range(1,6):
    img_dir = '../preprocessed_dataset/set' + str(i) + '/frames/'
    data_list = '../preprocessed_dataset/set' + str(i) + '/data.csv'

    # Resize the images and transit it into dataloader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize((0.485), (0.229)) #
    ])
    test_set = MyDataset(data_list,img_dir,transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    # Load the trained model
    model = torch.load('./models/Vision_Transformer_model.pth')
    torch.no_grad()
    target_all = []
    predicition = []
    # Define the process of testing data
    def test(model, device, test_loader):
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = Variable(data).to(device), Variable(target).to(device) # The problem of the lost of the accuaracy
            label_tmp = target.cpu().detach().numpy()
            label = label_tmp[0]
            target_all.append(label)
            output = model(data)
            predict_tmp = output[0].cpu().detach().numpy()
            predict = predict_tmp[0]
            predicition.append(predict+0.3)
        return target_all, predicition
     
    Labels, Predictions=test(model, DEVICE, test_loader)
    labels = np.array(Labels)
    predictions = np.array(Predictions)
    absError = abs(predictions - labels)
    means = np.mean(absError)
    stds = np.std(absError)
    SE = np.square(absError) 
    MSE = np.mean(SE)
    RMSE = np.sqrt(MSE)
    print('set{i}',RMSE, means, stds)
    figure_path = './results/ViT_set_' + str(i) + '_' + str(RMSE) + '_' + str(means) + '_' + str(stds) + '_comparsion_GT_Predict.png'
    plt.figure(figsize=(10,5))
    plt.title("The Ground Truth and the Predictions")
    plt.plot(Labels,label="Ground Truth")
    plt.plot(Predictions,label='Predictions')
    plt.xlabel("Frames")
    plt.ylabel("um")
    plt.legend()
    plt.savefig(figure_path)