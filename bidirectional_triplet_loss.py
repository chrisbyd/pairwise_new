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


def gather_2d(params,indices):
    indices = indices.t().long()
    ndim = indices.size(0)
    idx = torch.zeros_like(indices[0]).long()
    m=1
    for i in range(ndim)[::-1]:
        idx +=indices[i]*m
        m *= params.size(i)

    return torch.take(params,idx)


class BiDirectionalLoss(nn.Module):

    def __init__(self,lamda,batch_size,margin = 0.5):
        super(BiDirectionalLoss,self).__init__()
        self.lamda = lamda
        self.batch_size =batch_size
        self.margin = margin


    def forward(self,feat,label1,label2):
        feature1, feature2 = torch.split(feat, self.batch_size, 0)

        same_identity_mask = torch.unsqueeze(label1, 1) == torch.unsqueeze(label2, 0)
        negative_mask = torch.logical_not(same_identity_mask)

        # the weight matrix
        # derive the balanced_param matrix
        s = same_identity_mask.int().cuda()
        s_ = negative_mask.int().cuda()


        dis = cdist(feature1, feature2)
        furthest_positive = torch.max(dis*s,1).values
        pos_positive = torch.argmax(dis*s,1)
        #get the closet negative
        #-------------------------------
        closest_negative = torch.min(dis + 1e5*s ,1).values
        pos_negative = torch.argmin(dis + 1e5*s ,1)

        #compute the cross modality loss
        cross_diff1 = furthest_positive - closest_negative
        cross_diff1 = torch.clamp(cross_diff1+self.margin,min= 0.0)

        loss_cross_mean1 =  torch.mean(cross_diff1)


        # calculate the intra modal loss for thermal images
        # ----------------------------------------------------------------------------------

        dis_thermal = cdist(feature2, feature2)
        idx1 = torch.cat((pos_positive.view(self.batch_size,1),pos_negative.view(self.batch_size,1)),1)
        intra_diff1 = gather_2d(dis_thermal,idx1)

        intra_diff1 = torch.clamp(0.1 -intra_diff1,min=0.0)
        intra_loss1 = torch.mean(intra_diff1)



        # calculate the intra modal loss for visible images
        # ----------------------------------------------------------------------------------
        same_identity_mask = torch.unsqueeze(label2, 1) == torch.unsqueeze(label1, 0)
        s = same_identity_mask.int().cuda()
        negative_mask = torch.logical_not(same_identity_mask)
        dis_VT = cdist(feature2, feature1)
        furthest_positive = torch.max(dis_VT * s, 1).values
        pos_positive = torch.argmax(dis_VT * s, 1)

        # get the closet negative
        # -------------------------------
        closest_negative = torch.min(dis_VT + 1e5 * s, 1).values
        pos_negative = torch.argmin(dis_VT + 1e5 * s, 1)

        # compute the cross modality loss
        cross_diff2 = furthest_positive - closest_negative
        cross_diff2 = torch.clamp(cross_diff2+self.margin, min=0.0)
        loss_cross_mean2 = torch.mean(cross_diff2)

        # calculate the intra modal loss for visible images
        # ----------------------------------------------------------------------------------

        dis_visible = cdist(feature1, feature1)
        idx1 = torch.cat((pos_positive.view(self.batch_size, 1), pos_negative.view(self.batch_size, 1)), 1)
        intra_diff2 = gather_2d(dis_visible, idx1)
        intra_diff2 = torch.clamp(0.1 - intra_diff2, 0.0)
        intra_loss2 = torch.mean(intra_diff2)


        inter_loss = loss_cross_mean1+loss_cross_mean2
        intra_loss = intra_loss1 +intra_loss2

        # Derive the total loss
        # ----------------------------------------------------------------------------------
        Loss_total = inter_loss + self.lamda*intra_loss

        return Loss_total