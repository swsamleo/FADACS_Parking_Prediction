import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
import tqdm
import os
import time

DEVICE = torch.device('cuda:{}'.format(0)) if torch.cuda.is_available() else torch.device('cpu')

def _checkAndCreatDir(dirPath):
    try:
        os.makedirs(dirPath)
        print("Directory ", dirPath, " Created ")
    except FileExistsError:
        print("Directory ", dirPath, " already exists")


class ReverseLayerF(Function):
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class Discriminator(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dis1 = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dis2 = nn.Linear(hidden_dim, 1)


    def forward(self, x):
        x = F.relu(self.dis1(x))
        x = self.dis2(self.bn(x))
        x = torch.sigmoid(x)
        return x


class FeatureExtractorMLP(nn.Module):
    def __init__(self,featureSize,outFeatureSize):
        super(FeatureExtractorMLP, self).__init__()
        feature = nn.Sequential()
        feature = nn.Sequential()
        feature.add_module('c_fc1', nn.Linear(featureSize, outFeatureSize))
        feature.add_module('c_bn1', nn.BatchNorm1d(outFeatureSize))
        feature.add_module('c_relu1', nn.ReLU(True))
        feature.add_module('c_drop1', nn.Dropout2d())
        feature.add_module('c_fc2', nn.Linear(outFeatureSize, outFeatureSize))
        feature.add_module('c_bn2', nn.BatchNorm1d(outFeatureSize))
        feature.add_module('c_relu2', nn.ReLU(True))
        self.feature = feature

    def forward(self, x):
        return self.feature(x)


class RegressorMLP(nn.Module):
    def __init__(self,outFeatureSize):
        super(RegressorMLP, self).__init__()
        self.regressor = nn.Sequential()
        self.regressor.add_module('c_fc1', nn.Linear(outFeatureSize, outFeatureSize))
        self.regressor.add_module('c_bn1', nn.BatchNorm1d(outFeatureSize))
        self.regressor.add_module('c_relu1', nn.ReLU(True))
        self.regressor.add_module('c_drop1', nn.Dropout2d())
        self.regressor.add_module('c_fc2', nn.Linear(outFeatureSize, 50))
        self.regressor.add_module('c_bn2', nn.BatchNorm1d(50))
        self.regressor.add_module('c_relu2', nn.ReLU(True))
        self.regressor.add_module('c_fc3', nn.Linear(50, 1))

    def forward(self, x):
        return self.regressor(x)

class DANNMLP(nn.Module):

    def __init__(self, device,featureSize = 510 ,outFeatureSize = 200, input_dim=200, hidden_dim=50):
        super(DANNMLP, self).__init__()
        self.device = device
        self.feature = FeatureExtractorMLP(featureSize,outFeatureSize)
        self.regressor = RegressorMLP(outFeatureSize)
        self.domain_classifier = Discriminator( input_dim = input_dim, hidden_dim=hidden_dim)

    def forward(self, input_data, alpha=1, source=True):
        input_data = input_data.view(input_data.size(0), -1)
        feature = self.feature(input_data)
        class_output = self.regressor(feature)
        domain_output = self.get_adversarial_result(
            feature, source, alpha)
        return class_output, domain_output

    def get_adversarial_result(self, x, source=True, alpha=1):
        loss_fn = nn.BCELoss()
        if source:
            domain_label = torch.ones(len(x)).long().to(self.device)
        else:
            domain_label = torch.zeros(len(x)).long().to(self.device)
        x = ReverseLayerF.apply(x, alpha)
        domain_pred = self.domain_classifier(x)
        loss_adv = loss_fn(domain_pred, domain_label.float())
        return loss_adv



def test(model, dataloader, epoch):
    criterion = torch.nn.MSELoss()
    mae_crit = torch.nn.L1Loss()
    alpha = 0
    # dataloader = data_loader.load_test_data(dataset_name)
    model.eval()
    mse = 0
    mae = 0
    with torch.no_grad():
        for _, (t_data, t_label) in enumerate(dataloader):
            t_data, t_label = t_data.to(DEVICE), t_label.to(DEVICE)
            pred_output, _ = model(input_data=t_data, alpha=alpha)
            mse += criterion(pred_output, t_label).item()
            mae += mae_crit(pred_output, t_label).item()
    return mae,(mse / len(dataloader)) ** (0.5)


def train(data):
    modelFile = '{}dann_'.format(data["filepath"])+data["col"]+'.pth'
    start = time.time()
    _checkAndCreatDir(data["filepath"])

    dataloader_src = data["dataloaders"]["srcTrain"]
    dataloader_tar = data["dataloaders"]["tarTrain"]
    testloader_src = data["dataloaders"]["srcTest"]
    testloader_tar = data["dataloaders"]["tarTest"]
    
    torch.random.manual_seed(10)

    nepoch = 2
    learningRate = 1e-2
    gamma = .5
    input_dim = 200
    hidden_dim = 50
    featureSize = 510
    outFeatureSize = 200
    
    if data["parameters"] is not None:
        if "gamma" in data["parameters"]: gamma = data["parameters"]["gamma"]
        if "nepoch" in data["parameters"]: nepoch = data["parameters"]["nepoch"]
        if "learningRate" in data["parameters"]: learningRate = data["parameters"]["learningRate"]
        if "input_dim" in data["parameters"]: input_dim = data["parameters"]["input_dim"]
        if "hidden_dim" in data["parameters"]: hidden_dim = data["parameters"]["hidden_dim"]
        if "featureSize" in data["parameters"]: featureSize = data["parameters"]["featureSize"]
        if "outFeatureSize" in data["parameters"]: outFeatureSize = data["parameters"]["outFeatureSize"]

    
    model = DANNMLP(DEVICE,featureSize = featureSize ,
                    outFeatureSize = outFeatureSize, 
                    input_dim = input_dim, 
                    hidden_dim = hidden_dim).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=learningRate)
    
    criterion = torch.nn.MSELoss()
    best_mae = float('inf')
    len_dataloader = min(len(dataloader_src), len(dataloader_tar))
        
    if os.path.isfile(modelFile) and data["reTrain"] == False:
        print (data["col"] + "DANN Training Model File exist, skip training and loading it")
        model = torch.load(modelFile)
        model.eval()
    else:
        print("DANN Training [" + data["col"] + "] start!")

        for epoch in range(nepoch):
            model.train()
            i = 1
            for (data_src, data_tar) in tqdm.tqdm(zip(enumerate(dataloader_src), enumerate(dataloader_tar)), total=len_dataloader, leave=False):
                _, (x_src, y_src) = data_src
                _, (x_tar, _) = data_tar
                x_src, y_src, x_tar = x_src.to(DEVICE), y_src.to(DEVICE), x_tar.to(DEVICE)
                p = float(i + epoch * len_dataloader) / nepoch / len_dataloader
                alpha = 2. / (1. + np.exp(-10 * p)) - 1

                optimizer.zero_grad()

                pred_output, err_s_domain = model(input_data=x_src, alpha=alpha)
                err_s_label = criterion(pred_output, y_src)
                _, err_t_domain = model(input_data=x_tar, alpha=alpha, source=False)
                err_domain = err_t_domain + err_s_domain
                err = err_s_label + gamma * err_domain
                
                err.backward()
                optimizer.step()
                i += 1
                
            item_pr = 'Epoch: [{}/{}], classify_loss: {:.4f}, domain_loss_s: {:.4f}, domain_loss_t: {:.4f}, domain_loss: {:.4f},total_loss: {:.4f}'.format(
                epoch, nepoch, err_s_label.item(), err_s_domain.item(), err_t_domain.item(), err_domain.item(), err.item())
            print(item_pr)
            fp = open(data["filepath"]+"result.csv", 'a')
            fp.write(item_pr + '\n')

            torch.save(model, modelFile)
            
        # test
        mae_src,rmse_src = test(model, testloader_src, epoch)
        mae_tar,rmse_tar = test(model, testloader_tar, epoch)
        test_info = 'Source mae: {:.4f}, target mae: {:.4f}, Source rmse: {:.4f}, target rmse: {:.4f}'.format(mae_src, mae_tar,rmse_src, rmse_tar)
        fp.write(test_info + '\n')
        print(test_info)
        fp.close()
        
        if best_mae > mae_tar:
            best_mae = mae_tar
        end = time.time()
        print('DANN [{}] Train Done, Test best mae: {:.4f}'.format(data["col"],best_mae)+", spent: %.2fs" % (end - start))
        
        return {data["col"]:{"mae_src":mae_src,"rmse_src":rmse_src,"mae_tar":mae_tar,"rmse_tar":rmse_tar}}



# if __name__ == '__main__':
    
#     loader_src = get_melb(train = True)
#     loader_tar = get_morn(train = True)
    
#     testloader_src = get_melb(train = False)
#     testloader_tar = get_morn(train = False)
#     train(loader_src, loader_tar, testloader_src, testloader_tar)