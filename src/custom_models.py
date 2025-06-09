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


    def forward(self, x, return_features=False):
        x = self.feature_extractor(x)
        avg = self.pool_avg(x)
        max = self.pool_max(x)
        x = torch.cat([avg, max], dim=1) 
        x = self.flatten(x)

        if return_features:
            return x
        
        x = self.classifier(x)
        return x
    

    def get_layer_groups(self):
        layer_groups = []
        layer_groups.append(list(self.feature_extractor.conv1.parameters()))
        layer_groups.append(list(self.feature_extractor.conv2.parameters()))
        layer_groups.append(list(self.feature_extractor.conv3.parameters()))
        layer_groups.append(list(self.feature_extractor.conv4.parameters()))
        layer_groups.append(list(self.feature_extractor.conv5.parameters()))
        layer_groups.append(list(self.classifier.fc1.parameters()))
        layer_groups.append(list(self.classifier.fc2.parameters()))

        return layer_groups
    

# Clase de pérdida para la regresión cuantílica
# Esta pérdida mide cuán bien predice un modelo los cuantiles de una distribución
class QuantileLoss(nn.Module):
    
    def __init__(self, quantiles):
        
        super().__init__()
        self.quantiles = quantiles
        
        
    def forward(self, preds, targets):
        # Asegura que los targets no estén marcados para el cálculo de gradientes (esto es importante porque 
        # solo las predicciones deben participar en el backpropagation, no las etiquetas reales)
        assert not targets.requires_grad
        # Asegura que el batch size de las predicciones y los targets coincide
        assert preds.size(0) == targets.size(0)
        # Asegura que el número de columnas en preds coincida con el número de cuantiles que se quieren 
        # predecir
        assert preds.size(1) == len(self.quantiles)
        
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = targets - preds[:,i]
            losses.append(torch.max((q-1)*errors, q*errors).unsqueeze(1))
            
        #
        all_losses = torch.cat(losses, dim=1)
        loss = torch.mean(torch.sum(all_losses, dim=1))
        return loss