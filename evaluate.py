import torch
import torchvision.transforms as  transforms
from PIL import Image
from torchvision.utils import save_image
from utils.visualize import visualize, reverse_normalize
from cam import CAM, GradCAM, GradCAMpp, SmoothGradCAMpp

image = Image.open('../preprocessed_dataset/set3/frames/c2.png')
model = torch.load('./models/Densenet121_model.pth')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(DEVICE)
preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize((0.5), (0.5))
    ])

tensor = preprocess(image)
tensor = tensor.unsqueeze(0)
tensor = tensor.cuda()
target_layer1 = model.features.denseblock1.denselayer1.conv1
target_layer2 = model.features.denseblock1.denselayer6.conv2
target_layer3 = model.features.denseblock2.denselayer6.conv2
target_layer4 = model.features.denseblock2.denselayer12.conv2
target_layer5 = model.features.denseblock3.denselayer9.conv2
target_layer6 = model.features.denseblock3.denselayer19.conv2
target_layer7 = model.features.denseblock4.denselayer8.conv2
target_layer8 = model.features.denseblock4.denselayer16.conv2

wrapped_model1 = SmoothGradCAMpp(model, target_layer1, n_samples=25, stdev_spread=0.15)
wrapped_model2 = SmoothGradCAMpp(model, target_layer2, n_samples=25, stdev_spread=0.15)
wrapped_model3 = SmoothGradCAMpp(model, target_layer3, n_samples=25, stdev_spread=0.15)
wrapped_model4 = SmoothGradCAMpp(model, target_layer4, n_samples=25, stdev_spread=0.15)
wrapped_model5 = SmoothGradCAMpp(model, target_layer5, n_samples=25, stdev_spread=0.15)
wrapped_model6 = SmoothGradCAMpp(model, target_layer6, n_samples=25, stdev_spread=0.15)
wrapped_model7 = SmoothGradCAMpp(model, target_layer7, n_samples=25, stdev_spread=0.15)
wrapped_model8 = SmoothGradCAMpp(model, target_layer8, n_samples=25, stdev_spread=0.15)

cam1, idx1 = wrapped_model1(tensor)
cam2, idx2 = wrapped_model2(tensor)
cam3, idx3 = wrapped_model3(tensor)
cam4, idx4 = wrapped_model4(tensor)
cam5, idx5 = wrapped_model5(tensor)
cam6, idx6 = wrapped_model6(tensor)
cam7, idx7 = wrapped_model5(tensor)
cam8, idx8 = wrapped_model6(tensor)

cam1 = cam1.cpu()
cam2 = cam2.cpu()
cam3 = cam3.cpu()
cam4 = cam4.cpu()
cam5 = cam5.cpu()
cam6 = cam6.cpu()
cam7 = cam7.cpu()
cam8 = cam8.cpu()

img = reverse_normalize(tensor)

heatmap1 = visualize(img, cam1)
heatmap2 = visualize(img, cam2)
heatmap3 = visualize(img, cam3)
heatmap4 = visualize(img, cam4)
heatmap5 = visualize(img, cam5)
heatmap6 = visualize(img, cam6)
heatmap7 = visualize(img, cam7)
heatmap8 = visualize(img, cam8)

save_image(heatmap1, './pics/convlayer1.png')
save_image(heatmap2, './pics/convlayer2.png')
save_image(heatmap3, './pics/convlayer3.png')
save_image(heatmap4, './pics/convlayer4.png')
save_image(heatmap5, './pics/convlayer5.png')
save_image(heatmap6, './pics/convlayer6.png')
save_image(heatmap7, './pics/convlayer7.png')
save_image(heatmap8, './pics/convlayer8.png')


