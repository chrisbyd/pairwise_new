from __future__ import print_function
import argparse
import sys
import time 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from model import embed_net,Reconstruct
from utils import *
from sigmoid_cross_entropy_loss import *
from cauchy_cross_entropy_loss import  *
from pairwise_loss import *
import matplotlib.pyplot as plt
from bidirectional_triplet_loss import *
from sigmoid_triplet import  *
from util.visualizer import *




parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu',  help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str, 
                    help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', '-r', default='', type=str, 
                    help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only') 
parser.add_argument('--model_path', default='save_model/', type=str, 
                    help='model save path')
parser.add_argument('--save_epoch', default=20, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str, 
                    help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--low-dim', default=2048, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=16, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--method', default='id', type=str,
                    metavar='m', help='method type')
parser.add_argument('--drop', default=0.0, type=float,
                    metavar='drop', help='dropout ratio')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--gpu', default='0', type=str,
                      help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')

parser.add_argument('--lamda1', default=0.5, type=float,
                    metavar='t', help='the weight for the modal-specific loss')
parser.add_argument('--lamda2',default=0.6,type=float,help='the weight for the identity loss')

parser.add_argument('--lamda3',default=0.5,type=float,help='the weight auto encoder loss')

#---- argument for displaying the images on
parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
parser.add_argument('--display_ncols', type=int, default=4,
                    help='if positive, display all images in a single visdom web panel with certain number of images per row.')
parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
parser.add_argument('--display_env', type=str, default='main',
                    help='visdom display environment name (default is "main")')
parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
np.random.seed(0)

dataset = args.dataset
if dataset == 'sysu':
    data_path = './Dataset/SYSU-MM01/'
    log_path = args.log_path + 'sysu_log/'
    test_mode = [1, 2] # thermal to visible
elif dataset =='regdb':
    data_path = './Dataset/RegDB/'
    log_path = args.log_path + 'regdb_log/'
    test_mode = [2, 1] # visible to thermal

checkpoint_path = args.model_path

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
 
if args.method =='id':
    suffix = dataset + '_id_bn_relu'
elif args.method == 'bd':
    suffix = dataset + '_bidiretional_bn_relu_'
elif args.method == 'st':
    suffix = dataset + '_sigmoidtriplet_bn_relu_'
elif args.method == 'sc':
    suffix = dataset + '_sigmoidcrossentropy_bn_relu'

suffix = suffix + '_drop_{}'.format(args.drop) 
suffix = suffix + '_lr_{:1.1e}'.format(args.lr) 
suffix = suffix + '_dim_{}'.format(args.low_dim)
suffix = suffix + '_lamda1_{}'.format(args.lamda1)
suffix = suffix + '_lamda2_{}'.format(args.lamda2)
suffix = suffix + '_lamda3_{}'.format(args.lamda3)
if not args.optim == 'sgd':
    suffix = suffix + '_' + args.optim
suffix = suffix + '_' + args.arch
if dataset =='regdb':
    suffix = suffix + '_trial_{}'.format(args.trial)

if dataset =='sysu':
    suffix = suffix + '__mode__{}'.format(args.mode)

test_log_file = open(log_path + suffix + '.txt', "w")
sys.stdout = Logger(log_path  + suffix + '_os.txt')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0 
feature_dim = args.low_dim

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
#normalize = transforms.Normalize(mean=[0.5],std=[0.5])

transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((args.img_h,args.img_w)),
    transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation((30,60)),
    #transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    normalize,
])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h,args.img_w)),
    transforms.ToTensor(),
   normalize,
])

end = time.time()
if dataset =='sysu':
    # training set
    trainset = SYSUData(data_path,  transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)
    
    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode = args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode = args.mode, trial = 0)
      
elif dataset =='regdb':
    # training set
    trainset = RegDBData(data_path, args.trial, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)
    
    # testing set
    query_img, query_label = process_test_regdb(data_path, trial = args.trial, modal = 'visible')
    gall_img, gall_label  = process_test_regdb(data_path, trial = args.trial, modal = 'thermal')

gallset  = TestData(gall_img, gall_label, transform = transform_test, img_size =(args.img_w,args.img_h))
queryset = TestData(query_img, query_label, transform = transform_test, img_size =(args.img_w,args.img_h))
    
# testing data loader
gall_loader  = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
   
n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')   
print('Data Loading Time:\t {:.3f}'.format(time.time()-end))


print('==> Building model..')
net = embed_net(args.low_dim, n_class, drop = args.drop, arch=args.arch)
net.to(device)
Rec_net = Reconstruct(args.batch_size)
Rec_net.to(device)
cudnn.benchmark = True

if len(args.resume)>0:   
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

criterion = nn.CrossEntropyLoss()
criterion.to(device)
use = 1
if args.method == 'id':
    use = 0
    criterion1 = BiDirectionalLoss(args.lamda1, args.batch_size, 0.5)
if args.method =='bd':
    criterion1 = BiDirectionalLoss(args.lamda1,args.batch_size,0.5)
    criterion1.to(device)
elif args.method == 'st':
    criterion1 = SigmoidTriplet(args.lamda1,args.batch_size,0.5)
    criterion1.to(device)
elif args.method == 'sc':
    criterion1 = SigmoidCrossEntropyLoss(args.lamda1,args.batch_size)
    criterion1.to(device)


AutoEncoderLoss = AutoEncoderLoss()
AutoEncoderLoss.to(device)


#------------------visualizer for visualizing the auto encoding pictures
visualizer = Visualizer(args)

ignored_params = list(map(id, net.feature.parameters() )) + list(map(id, net.classifier.parameters())) 
base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
if args.optim == 'sgd':
    optimizer = optim.SGD([
         {'params': base_params, 'lr': 0.1*args.lr},
        {'params': Rec_net.parameters(),'lr':args.lr},
         {'params': net.feature.parameters(), 'lr': args.lr},
         {'params': net.classifier.parameters(), 'lr': args.lr}],
         weight_decay=5e-4, momentum=0.9, nesterov=True)
elif args.optim == 'adam':
    optimizer = optim.Adam([
         {'params': base_params, 'lr': 0.1*args.lr},
         {'params': net.feature.parameters(), 'lr': args.lr},

         {'params': net.classifier.parameters(), 'lr': args.lr}],weight_decay=5e-4)
         
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 40:
        lr = args.lr
    elif epoch >= 40 and epoch < 80:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01

    optimizer.param_groups[0]['lr'] = 0.1*lr
    optimizer.param_groups[1]['lr'] = lr
    optimizer.param_groups[2]['lr'] = lr
    return lr
    
def train(epoch):
    current_lr = adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0
    visualizer.reset()
    # switch to train mode
    net.train()
    end = time.time()
    for batch_idx, (input1, input2, label1, label2) in enumerate(trainloader):

        input1 = Variable(input1.cuda())
        input2 = Variable(input2.cuda())
        
        labels = torch.cat((label1,label2),0)
        labels = Variable(labels.cuda())
        data_time.update(time.time() - end)
        
        outputs, feat = net(input1, input2)
        visible_img_rec ,thermal_img_rec = Rec_net(feat)
        #debug the size of features ,labels
        # print(feat.size())
        # print(label1)
        # print(label2)
        # exit()

        loss_id = criterion(outputs, labels)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        loss_ae = AutoEncoderLoss(input1,visible_img_rec,input2,thermal_img_rec)
        #the total loss
        #-----------------------------------------
        loss_ancillary = criterion1(feat,label1,label2)
        loss=use*loss_ancillary + args.lamda2 * loss_id + args.lamda3*loss_ae
       #  loss = loss_ae

        optimizer.zero_grad()    
        loss.backward()
        optimizer.step()
        train_loss.update(loss.item(), 2*input1.size(0))

        total += labels.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx%10 ==0:
            print('Epoch: [{}][{}/{}] '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'lr:{} '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                  'Accu: {:.2f}' .format(
                  epoch, batch_idx, len(trainloader),current_lr, 
                  100.*correct/total, batch_time=batch_time, 
                  data_time=data_time, train_loss=train_loss))
        visuals = {"real_a": input1, "real_b": input2, "fake_a": visible_img_rec, "fake_b": thermal_img_rec}
        visualizer.display_current_results(visuals=visuals, epoch=epoch, save_result=True)



def test(epoch):   
    # switch to evaluation mode
    net.eval()
    print ('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, args.low_dim))
    with torch.no_grad():
        for batch_idx, (input, label ) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat_pool, feat = net(input, input, test_mode[0])
            gall_feat[ptr:ptr+batch_num,: ] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))   

    # switch to evaluation mode
    net.eval()
    print ('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, args.low_dim))
    with torch.no_grad():
        for batch_idx, (input, label ) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat_pool, feat = net(input, input, test_mode[1])
            query_feat[ptr:ptr+batch_num,: ] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num         
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    
    start = time.time()
    # compute the similarity
    distmat  = np.matmul(query_feat, np.transpose(gall_feat))
    #---------------------
    # query_feat = torch.from_numpy(query_feat)
    # gall_feat  = torch.from_numpy(gall_feat)
    # distmat = cdist(query_feat,gall_feat,'cosine')
    # evaluation
    if dataset =='regdb':
        cmc, mAP = eval_regdb(-distmat, query_label, gall_label)
    elif dataset =='sysu':
        cmc, mAP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
    print('Evaluation Time:\t {:.3f}'.format(time.time()-start))
    return cmc, mAP
    
# training
print('==> Start Training...')    
for epoch in range(start_epoch, 100-start_epoch):

    print('==> Preparing Data Loader...')
    # identity sampler
    sampler = IdentitySampler(trainset.train_color_label, \
        trainset.train_thermal_label, color_pos, thermal_pos, args.batch_size)
    trainset.cIndex = sampler.index1 # color index
    trainset.tIndex = sampler.index2 # thermal index
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size,\
        sampler = sampler, num_workers=args.workers, drop_last =True)
    
    # training
    train(epoch)

    if epoch > 0 and epoch%2 ==0:
        print ('Test Epoch: {}'.format(epoch))
        print ('Test Epoch: {}'.format(epoch),file=test_log_file)
        # testing
        cmc, mAP = test(epoch)

        print('FC:   Rank-1: {:.2%} |Rank-2: {:.2%} |Rank-3: {:.2%} |Rank-4: {:.2%} |Rank-5: {:.2%} |Rank-6: {:.2%} |Rank-7: {:.2%} |Rank-8: {:.2%} | Rank-9: {:.2%} | Rank-10: {:.2%}| mAP: {:.2%}'.format(
                cmc[0], cmc[1],cmc[2],cmc[3], cmc[4],cmc[5],cmc[6],cmc[7],cmc[8], cmc[9], mAP))
        print(
            'FC:   Rank-1: {:.2%} |Rank-2: {:.2%} |Rank-3: {:.2%} |Rank-4: {:.2%} |Rank-5: {:.2%} |Rank-6: {:.2%} |Rank-7: {:.2%} |Rank-8: {:.2%} | Rank-9: {:.2%} | Rank-10: {:.2%}| mAP: {:.2%}'.format(
                cmc[0], cmc[1], cmc[2], cmc[3], cmc[4], cmc[5], cmc[6], cmc[7], cmc[8], cmc[9], mAP),file=test_log_file)

        test_log_file.flush()
        
        # save model
        if cmc[0] > best_acc: # not the real best for sysu-mm01 
            best_acc = cmc[0]
            state = {
                'net': net.state_dict(),
                'cmc': cmc,
                'mAP': mAP,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_best.t')
        
        # save model every 20 epochs    
        if epoch > 10 and epoch%args.save_epoch ==0:
            state = {
                'net': net.state_dict(),
                'cmc': cmc,
                'mAP': mAP,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_epoch_{}.t'.format(epoch))
