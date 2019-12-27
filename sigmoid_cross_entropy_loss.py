import torch
from torch.autograd import Variable
import torch.nn as nn

def all_diffs(a,b):
    result = torch.unsqueeze(a,1) -torch.unsqueeze(b,0)
    return result

def cdist(a,b,metric='euclidean'):
    diffs =all_diffs(a,b)
    if metric =='euclidean':
        dis = torch.sqrt(torch.sum(torch.mul(diffs,diffs),-1)+1e-12)
    elif metric =='cosine':
        vector_product = torch.matmul(a,b.t())
        l2_norm_a = torch.unsqueeze(torch.norm(a,p=2,dim=1),1)
        l2_norm_b = torch.unsqueeze(torch.norm(b,p=2,dim=1),0)
        l2_norm_product = l2_norm_a*l2_norm_b
        dis = torch.div(vector_product,l2_norm_product)
    else:
        raise NotImplementedError(
            "The distance metric has not been implemented"
        )
    return  dis




class SigmoidCrossEntropyLoss(nn.Module):

    def __init__(self,lamda,batch_size):
        super(SigmoidCrossEntropyLoss,self).__init__()
        self.lamda = lamda
        self.batch_size =batch_size


    def forward(self,feat,label1,label2):
        feature1, feature2 = torch.split(feat, self.batch_size, 0)

        same_identity_mask = torch.unsqueeze(label1, 1) == torch.unsqueeze(label2, 0)
        negative_mask = torch.logical_not(same_identity_mask)

        # the weight matrix
        sum_all_elements = self.batch_size * self.batch_size
        sum_same_identities = torch.sum(same_identity_mask.int())
        sum_negative_identities = torch.sum(negative_mask.int())

        # derive the balanced_param matrix
        s = same_identity_mask.int().cuda()
        s_ = negative_mask.int().cuda()
        balance_param = torch.mul(torch.div(sum_all_elements, sum_same_identities), s) + \
                        torch.mul(torch.div(sum_all_elements, sum_negative_identities), s_)

        dis = cdist(feature1, feature2,'cosine')
        dis = dis
        dis = dis.cuda()


        Loss_cross = torch.log(1 + torch.exp(dis)) -s * dis
        weighted_loss_cross =torch.mean(Loss_cross * balance_param)

        # calculate the intra modal loss for thermal images
        # ----------------------------------------------------------------------------------
        same_identity_mask_thermal = torch.unsqueeze(label2, 1) == torch.unsqueeze(label2, 0)
        negative_mask_thermal = torch.logical_not(same_identity_mask_thermal)

        s_all_thermal = self.batch_size * self.batch_size
        s_same_thermal = torch.sum(same_identity_mask_thermal.int())
        s_negative_thermal = torch.sum(negative_mask_thermal.int())

        s_thermal = same_identity_mask_thermal.int().cuda()
        s_thermal_negative = negative_mask_thermal.int().cuda()
        balance_param_thermal = torch.mul(torch.div(s_all_thermal, s_same_thermal), s_thermal) + \
                                torch.mul(torch.div(s_all_thermal, s_negative_thermal), s_thermal_negative)
        dis_thermal = cdist(feature2, feature2,'cosine')
        dis_thermal = dis_thermal
        Loss_thermal = torch.log(1 + torch.exp(dis_thermal)) - s * dis_thermal
        weighted_loss_thermal = torch.mean(Loss_thermal * balance_param_thermal)

        # calculate the intra modal loss for visible images
        # ----------------------------------------------------------------------------------
        same_identity_mask_visible = torch.unsqueeze(label1, 1) == torch.unsqueeze(label1, 0)
        negative_mask_visible = torch.logical_not(same_identity_mask_visible)

        s_all_visible = self.batch_size * self.batch_size
        s_same_visible = torch.sum(same_identity_mask_visible.int())
        s_negative_visible = torch.sum(negative_mask_visible.int())

        s_visible = same_identity_mask_visible.int().cuda()
        s_visible_negative = negative_mask_visible.int().cuda()
        balance_param_visible = torch.mul(torch.div(s_all_visible, s_same_visible), s_visible) + \
                                torch.mul(torch.div(s_all_visible, s_negative_visible), s_visible_negative)
        dis_visible = cdist(feature1, feature1,'cosine')
        dis_visible = dis_visible
        Loss_visible = torch.log(1 + torch.exp(dis_visible)) - s * dis_visible
        weighted_loss_visible = torch.mean(Loss_visible * balance_param_visible)

        # Derive the total loss
        # ----------------------------------------------------------------------------------
        Loss_total = weighted_loss_cross + self.lamda * (weighted_loss_thermal + weighted_loss_visible)

        return Loss_total


class CrossAutoEncoderLoss(nn.Module):
    def __init__(self):
        super(CrossAutoEncoderLoss,self).__init__()
        self.mseloss = nn.MSELoss()

    def forward(self, real_visible,rec_visible,real_thermal,rec_thermal):
        loss1 = self.mseloss(real_visible,rec_thermal)
        loss2 = self.mseloss(real_thermal,rec_visible)
        return loss1+loss2

class IntraModalAELoss(nn.Module):
    def __init__(self):
        super(IntraModalAELoss,self).__init__()
        self.mseloss =nn.MSELoss()

    def forward(self, real_visible,rec_visible,real_visible1,rec_visible1):
        loss1 = self.mseloss(real_visible,rec_visible1)
        loss2 = self.mseloss(real_visible1,rec_visible)
        return loss1+loss2



