import torch
import torchvision 
import torch.nn as nn


class FeatureExtractorResNeXt(nn.Module):
    
    def __init__(self):
        
        super(FeatureExtractorResNeXt, self).__init__()
        
        resnext = torchvision.models.resnext50_32x4d(weights='DEFAULT')
        
        self.conv1 = nn.Sequential(
            resnext.conv1,
            resnext.bn1,
            resnext.relu,
            resnext.maxpool,
        )
        self.conv2 = resnext.layer1
        self.conv3 = resnext.layer2
        self.conv4 = resnext.layer3
        self.conv5 = resnext.layer4
        
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        return x
        

class ClassifierResNeXt(nn.Module):
    
    def __init__(self, num_targets=1):

        super(ClassifierResNeXt, self).__init__()  
        
        self.num_targets = num_targets

        self.fc1 = nn.Sequential(
            nn.BatchNorm1d(4096),  # 2048 (avg) + 2048 (max)
            nn.Dropout(p = 0.5),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True)
        )

        self.fc2 = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Dropout(p = 0.5),
            nn.Linear(512, num_targets) 
        )
        
        
    def forward(self, x):
        
        x = self.fc1(x)
        x = self.fc2(x)
    
        # nn.Linear siempre devuelve una salida 2D
        # Aplanamos la salida si solo hay una variable target
        return x.squeeze(-1) if self.num_targets==1 else x
        

#
class ResNeXtRegressor(nn.Module):
    
    def __init__(self, num_targets=1):
        
        super(ResNeXtRegressor, self).__init__()
        
        self.feature_extractor = FeatureExtractorResNeXt()
        
        # Nueva head
        self.pool_avg = nn.AdaptiveAvgPool2d((1,1))
        self.pool_max = nn.AdaptiveMaxPool2d((1,1))
        self.flatten = nn.Flatten()

        self.classifier = ClassifierResNeXt(num_targets)


    def forward(self, x):
        x = self.feature_extractor(x)
        avg = self.pool_avg(x)
        max = self.pool_max(x)
        x = torch.cat([avg, max], dim=1) 
        x = self.flatten(x)
        x = self.classifier(x)
        return x
    

