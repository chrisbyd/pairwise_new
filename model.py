import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power
    
    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1./self.power)
        out = x.div(norm)
        return out

# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        # init.normal_(m.weight.data, 0, 0.001)
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        init.zeros_(m.bias.data)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class FeatureBlock(nn.Module):
    def __init__(self, input_dim, low_dim, dropout=0.5, relu=True):
        super(FeatureBlock, self).__init__()
        feat_block = []
        feat_block += [nn.Linear(input_dim, low_dim)] 
        feat_block += [nn.BatchNorm1d(low_dim)]
        
        feat_block = nn.Sequential(*feat_block)
        feat_block.apply(weights_init_kaiming)
        self.feat_block = feat_block
    def forward(self, x):
        x = self.feat_block(x)
        return x
        
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=0.5, relu=True):
        super(ClassBlock, self).__init__()
        classifier = []       
        if relu:
            classifier += [nn.LeakyReLU(0.1)]
        if dropout:
            classifier += [nn.Dropout(p=dropout)]
        
        classifier += [nn.Linear(input_dim, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.classifier = classifier
    def forward(self, x):
        x = self.classifier(x)
        return x       

# Define the ResNet18-based Model
class visible_net_resnet(nn.Module):
    def __init__(self, arch ='resnet18'):
        super(visible_net_resnet, self).__init__()
        if arch =='resnet18':
            model_ft = models.resnet18(pretrained=True)
        elif arch =='resnet50':
            model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.visible = model_ft
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        x = self.visible.layer1(x)
        x = self.visible.layer2(x)
        x = self.visible.layer3(x)
        x = self.visible.layer4(x)
        x = self.visible.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        # x = self.dropout(x)
        return x
        
class thermal_net_resnet(nn.Module):
    def __init__(self, arch ='resnet18'):
        super(thermal_net_resnet, self).__init__()
        if arch =='resnet18':
            model_ft = models.resnet18(pretrained=True)
        elif arch =='resnet50':
            model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.thermal = model_ft
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        x = self.thermal.layer1(x)
        x = self.thermal.layer2(x)
        x = self.thermal.layer3(x)
        x = self.thermal.layer4(x)
        x = self.thermal.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        # x = self.dropout(x)
        return x
        
class embed_net(nn.Module):
    def __init__(self, low_dim, class_num, drop = 0.5, arch ='resnet50'):
        super(embed_net, self).__init__()
        if arch =='resnet18':
            self.visible_net = visible_net_resnet(arch = arch)
            self.thermal_net = thermal_net_resnet(arch = arch)
            pool_dim = 512
        elif arch =='resnet50':
            self.visible_net = visible_net_resnet(arch = arch)
            self.thermal_net = thermal_net_resnet(arch = arch)
            pool_dim = 2048

        self.feature = FeatureBlock(pool_dim, low_dim, dropout = drop)
        self.classifier = ClassBlock(low_dim, class_num, dropout = drop)
        self.l2norm = Normalize(2)
        
    def forward(self, x1, x2, modal = 0 ):
        if modal==0:
            x1 = self.visible_net(x1)
            x2 = self.thermal_net(x2)
            x = torch.cat((x1,x2), 0)
        elif modal ==1:
            x = self.visible_net(x1)
        elif modal ==2:
            x = self.thermal_net(x2)
        
        y = self.feature(x) 
        out = self.classifier(y)
        if self.training:
            return out, self.l2norm(y)
        else:
            return self.l2norm(x), self.l2norm(y)

class ReconstructNet(nn.Module):
    def __init__(self,feature_dim,norm_layer=nn.BatchNorm2d,ngf=64,output_nc=3):
        super(ReconstructNet,self).__init__()
        self.pool_dim = 512
        self.fc = nn.Linear(feature_dim,self.pool_dim*9*5)

        model_2 = [
            norm_layer(self.pool_dim),
            nn.ConvTranspose2d(self.pool_dim, 256, stride=2, padding=1, kernel_size=3, output_padding=(1, 0)),
            norm_layer(256),
            nn.ReLU(True)
        ]
        model_2 += [
            nn.ConvTranspose2d(256, 128, stride=2, padding=1, kernel_size=3, output_padding=1),
            norm_layer(128),
            nn.ReLU(True)
        ]
        model_2 += [
            nn.ConvTranspose2d(128, 64, stride=2, padding=1, kernel_size=3, output_padding=1),
            norm_layer(64),
            nn.ReLU(True)
        ]
        model_2 += [
            nn.Upsample(scale_factor=2)
        ]
        model_2 += [
            nn.ConvTranspose2d(64, ngf, stride=2, padding=1, kernel_size=3, output_padding=1),
            norm_layer(ngf),
            nn.ReLU(True)
        ]
        model_2 += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.model_2 = nn.Sequential(*model_2)

    def forward(self, input):
        output_1 = self.fc(input)

        resize = output_1.view(output_1.shape[0], self.pool_dim, 9, 5)

        fake_img = self.model_2(resize)

        return fake_img



class Reconstruct(nn.Module):
    def __init__(self,batch_size):
        super(Reconstruct,self).__init__()
        self.reconstruct_visible_net = ReconstructNet(feature_dim=512)
        self.reconstruct_thermal_net = ReconstructNet(feature_dim=512)
        self.batch_size= batch_size

    def forward(self, input):
        visible_feature, thermal_feature = torch.split(input,self.batch_size,dim= 0)
        re_visible_img = self.reconstruct_visible_net(visible_feature)
        re_therm_img = self.reconstruct_thermal_net(thermal_feature)
        return re_visible_img,re_therm_img
# debug model structure

# net = embed_net(512, 319)
# net.train()
# input = Variable(torch.FloatTensor(8, 3, 224, 224))
# x, y  = net(input, input)