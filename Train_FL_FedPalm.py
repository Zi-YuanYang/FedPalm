import os

os.environ["KMP_DUPLICATE_LIB_OK"]= "TRUE"
import argparse


### This is for Original CompNet

parser = argparse.ArgumentParser(
        description="FedPalm"
    )

parser.add_argument("--batch_size",type=int,default = 2048)
parser.add_argument("--epoch_num",type=int,default = 1)
parser.add_argument("--com",type= int,default=200)
parser.add_argument("--temp", type=float, default= 0.07)
parser.add_argument("--weight1",type=float,default = 0.8)
parser.add_argument("--weight2",type=float,default = 0.2)
parser.add_argument("--id_num",type=int, default = 600, help = "IITD: 460 KTU: 145 Tongji: 600 REST: 358 XJTU: 200 POLYU 378 Multi-Spec 500 IITD_Right 230 No_Delete_PolyU 386 Tongji_LR 300")
parser.add_argument("--gpu_id",type=str, default='0')
parser.add_argument("--lr",type=float, default=0.001)
parser.add_argument("--redstep",type=int, default=30)
parser.add_argument("--mode", type=str, default='fedpalm', help="fedavg|fedprox|fedpdf")
parser.add_argument("--client_num",type=int,default = 8)
parser.add_argument("--test_interval",type=str,default = 100)
parser.add_argument("--save_interval",type=str,default = 100)  ## 200 for Multi-spec 500 for RED

##Training Path
parser.add_argument("--train_set_file",type=str,default='./data/train_all.txt')
parser.add_argument("--test_set_file",type=str,default='./data/test.txt')

##Store Path
parser.add_argument("--des_path",type=str,default='./Test_Local_Rst/checkpoint/')
parser.add_argument("--path_rst",type=str,default='./Test_Local_Rst/rst_test/')
# parser.add_argument("--save_path",type=str,default='./cross-db-checkpoint/PolyU_1')
parser.add_argument("--seed",type=int,default=42)

parser.add_argument("--ratio",type=float,default=0.5,help='The ratio of test id')

parser.add_argument("--loss",type=str,default='trus',help='turs, sup or single')
args = parser.parse_args()

# print(args.gpu_id)
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id



import time
import sys

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms
from torchvision import models

# print(torch.cuda.is_available())
# print(os.getcwd())
# import pickle
import numpy as np
from PIL import Image
import cv2 as cv
from loss import SupConLoss

import matplotlib.pyplot as plt

from utils.util import plotLossACC, saveLossACC, saveGaborFilters, saveParameters, saveFeatureMaps

plt.switch_backend('agg')

from models import MyDataset, MyDataset_general_FL
from models.compnet_original import compnet_fedpalm
from utils import *
import random
import copy

def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def communication(args, server_model, models, client_weights):
    with torch.no_grad():
        # aggregate params
        if args.mode.lower() == 'fedbn':
            for key in server_model.state_dict().keys():
                if 'bn' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])

        elif args.mode.lower() == 'fedpdf':
            for key in server_model.state_dict().keys():
##                if 'weight_fed' not in key and 'MLP' not in key :
#                if 'keys' not in key:
                if 'MLP' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        elif args.mode.lower() == 'fedpalm':
            for key in server_model.state_dict().keys():
##                if 'weight_fed' not in key and 'MLP' not in key :
#                if 'keys' not in key:
                # if 'fc_brand' in key:
                # if 'brand' in key:
                temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                for client_idx in range(len(client_weights)):
                    temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                server_model.state_dict()[key].data.copy_(temp)
                for client_idx in range(len(client_weights)):
                    models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        else:
            for key in server_model.state_dict().keys():
                # num_batches_tracked is a non trainable LongTensor and
                # num_batches_tracked are the same for all clients for the given datasets
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    return server_model, models

def test_op(model, models, data_loader_train, data_loader_test, path_rst):

    print('Start Testing!')
    print('%s' % (time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))

    path_hard = os.path.join(path_rst, 'rank1_hard')

    # output dir
    if not os.path.exists(path_rst):
        os.makedirs(path_rst)

    if not os.path.exists(path_hard):
        os.makedirs(path_hard)

    net = model  ## Server Model

    net.cuda()
    net.eval()
    for idx in range(args.client_num):
        models[idx].cuda()
        models[idx].eval()
    # feature extraction:

    featDB_train = []
    iddb_train = []

    for batch_id, (datas, target) in enumerate(data_loader_train):
        # break

        data = datas[0]

        data = data.cuda()
        target = target.cuda()

        output, fe_origi, _ = net(data, None, None)  ### Anchor Feature
            # evis = []
        fes = []
        for idx in range(args.client_num):
            ## use hyperparameter
            _, fe_1 , gabor_out = models[idx](data, None, None)  ### Anchor Feature
            # _, fe_1, _ = net(data, None, gabor_out)
            # evis.append(evi_1)
            fes.append(fe_1)

        fes_stack = torch.stack(fes)
        sim_fe = similarity_fes_op(fe_origi, fes_stack)

        # anch_output, ancho_fe, _ = net(data, target, None)
        # codes = (sim_fe + fe_origi) / 2
        codes = sim_fe * 0.2 + fe_origi * 0.8

        codes = codes.cpu().detach().numpy()
        y = target.cpu().detach().numpy()

        if batch_id == 0:
            featDB_train = codes
            iddb_train = y
        else:
            featDB_train = np.concatenate((featDB_train, codes), axis=0)
            iddb_train = np.concatenate((iddb_train, y))

    print('completed feature extraction for training set.')
    print('featDB_train.shape: ', featDB_train.shape)

    classNumel = len(set(iddb_train))
    num_training_samples = featDB_train.shape[0]
    # assert num_training_samples % classNumel == 0
    trainNum = num_training_samples // classNumel
    print('[classNumel, imgs/class]: ', classNumel, trainNum)
    # print('\n')

    featDB_test = []
    iddb_test = []

    print('Start Test Feature Extraction.')
    for batch_id, (datas, target) in enumerate(data_loader_test):

        data = datas[0]

        data = data.cuda()
        target = target.cuda()

        output, fe_origi, _ = net(data, None, None)  ### Anchor Feature
            # evis = []
        fes = []
        for idx in range(args.client_num):
            ## use hyperparameter
            _, fe_1, gabor_out = models[idx](data, None, None)  ### Anchor Feature
            # _, fe_1, _ = net(data, None, gabor_out)
            # evis.append(evi_1)
            fes.append(fe_1)

        fes_stack = torch.stack(fes)
        sim_fe = similarity_fes_op(fe_origi,fes_stack)

        # anch_output, ancho_fe, _ = net(data, target, None)
        # codes = (sim_fe + fe_origi) / 2
        codes = sim_fe * 0.2 + fe_origi * 0.8
        codes = codes.cpu().detach().numpy()

        y = target.cpu().detach().numpy()

        if batch_id == 0:
            featDB_test = codes
            iddb_test = y
        else:
            featDB_test = np.concatenate((featDB_test, codes), axis=0)
            iddb_test = np.concatenate((iddb_test, y))

    print('featDB_test.shape: ', featDB_test.shape)

    print('start feature matching ...\n')

    print('Verification EER of the test set ...')

    # verification EER of the test set
    s = []  # matching score
    l = []  # intra-class or inter-class matching
    ntest = featDB_test.shape[0]
    ntrain = featDB_train.shape[0]

    for i in range(ntest):
        feat1 = featDB_test[i]

        for j in range(ntrain):
            feat2 = featDB_train[j]

            cosdis = np.dot(feat1, feat2)
            dis = np.arccos(np.clip(cosdis, -1, 1)) / np.pi

            s.append(dis)

            if iddb_test[i] == iddb_train[j]:  # same palm
                l.append(1)
            else:
                l.append(-1)

    if not os.path.exists(path_rst+'veriEER'):
        os.makedirs(path_rst+'veriEER')
    if not os.path.exists(path_rst+'veriEER/rank1_hard/'):
        os.makedirs(path_rst+'veriEER/rank1_hard/')

    with open(path_rst+'veriEER/scores_VeriEER.txt', 'w') as f:
        for i in range(len(s)):
            score = str(s[i])
            label = str(l[i])
            f.write(score + ' ' + label + '\n')

    sys.stdout.flush()
    os.system('python ./getGI.py' + '  ' + path_rst + 'veriEER/scores_VeriEER.txt scores_VeriEER')
    os.system('python ./getEER.py' + '  ' + path_rst + 'veriEER/scores_VeriEER.txt scores_VeriEER')

    print('\n------------------')
    print('Rank-1 acc of the test set...')
    # rank-1 acc
    cnt = 0
    corr = 0
    for i in range(ntest):
        probeID = iddb_test[i]

        dis = np.zeros((ntrain, 1))

        for j in range(ntrain):
            dis[j] = s[cnt]
            cnt += 1

        idx = np.argmin(dis[:])

        galleryID = iddb_train[idx]

        if probeID == galleryID:
            corr += 1

    rankacc = corr / ntest * 100
    print('rank-1 acc: %.3f%%' % rankacc)
    print('-----------')

    with open(path_rst + 'veriEER/rank1.txt', 'w') as f:
        f.write('rank-1 acc: %.3f%%' % rankacc)

    print('\n\nReal EER of the test set...')

def test_cl(model, models, now_id, data_loader_train, data_loader_test, path_rst):

    print('Start Testing!')
    print('%s' % (time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))

    path_hard = os.path.join(path_rst, 'rank1_hard')

    # output dir
    if not os.path.exists(path_rst):
        os.makedirs(path_rst)

    if not os.path.exists(path_hard):
        os.makedirs(path_hard)

    net = model  ## Server Model

    net.cuda()
    net.eval()
    for idx in range(args.client_num):
        models[idx].cuda()
        models[idx].eval()
    # feature extraction:

    featDB_train = []
    iddb_train = []

    for batch_id, (datas, target) in enumerate(data_loader_train):
        # break

        data = datas[0]

        data = data.cuda()
        target = target.cuda()

        output, fe_origi, _ = models[now_id](data, None, None)  ### Anchor Feature
            # evis = []
        fes = []
        for idx in range(args.client_num):
            if idx == now_id:
                continue
            ## use hyperparameter
            _, fe_1, gabor_out = models[idx](data, None, None)  ### Anchor Feature
            # _, fe_1, _ = models[now_id](data, None, gabor_out)
            # evis.append(evi_1)
            fes.append(fe_1)

        fes_stack = torch.stack(fes)
        sim_fe = similarity_fes(fe_origi,fes_stack)

        anch_output, ancho_fe, _ = net(data, None, None)
        # codes = (sim_fe + ancho_fe) / 2
        codes = sim_fe * 0.2 + ancho_fe * 0.8

        codes = codes.cpu().detach().numpy()
        y = target.cpu().detach().numpy()

        if batch_id == 0:
            featDB_train = codes
            iddb_train = y
        else:
            featDB_train = np.concatenate((featDB_train, codes), axis=0)
            iddb_train = np.concatenate((iddb_train, y))

    print('completed feature extraction for training set.')
    print('featDB_train.shape: ', featDB_train.shape)

    classNumel = len(set(iddb_train))
    num_training_samples = featDB_train.shape[0]
    # assert num_training_samples % classNumel == 0
    trainNum = num_training_samples // classNumel
    print('[classNumel, imgs/class]: ', classNumel, trainNum)
    # print('\n')

    featDB_test = []
    iddb_test = []

    print('Start Test Feature Extraction.')
    for batch_id, (datas, target) in enumerate(data_loader_test):


        data = datas[0]

        data = data.cuda()
        target = target.cuda()

        output, fe_origi, _ = models[now_id](data, None, None)  ### Anchor Feature
            # evis = []
        fes = []
        for idx in range(args.client_num):
            if idx == now_id:
                continue
            ## use hyperparameter
            _, fe_1, gabor_out = models[idx](data, None, None)  ### Anchor Feature
            # _, fe_1, _ = models[now_id](data, None, gabor_out)
            # evis.append(evi_1)
            fes.append(fe_1)

        fes_stack = torch.stack(fes)
        sim_fe = similarity_fes(fe_origi, fes_stack)

        anch_output, ancho_fe, _ = net(data, None, None)
        # codes = (sim_fe + ancho_fe) / 2
        codes = sim_fe * 0.2 + ancho_fe * 0.8

        codes = codes.cpu().detach().numpy() 
        y = target.cpu().detach().numpy()

        if batch_id == 0:
            featDB_test = codes
            iddb_test = y
        else:
            featDB_test = np.concatenate((featDB_test, codes), axis=0)
            iddb_test = np.concatenate((iddb_test, y))

    print('featDB_test.shape: ', featDB_test.shape)

    print('start feature matching ...\n')

    print('Verification EER of the test set ...')

    # verification EER of the test set
    s = []  # matching score
    l = []  # intra-class or inter-class matching
    ntest = featDB_test.shape[0]
    ntrain = featDB_train.shape[0]

    for i in range(ntest):
        feat1 = featDB_test[i]

        for j in range(ntrain):
            feat2 = featDB_train[j]

            cosdis = np.dot(feat1, feat2)
            dis = np.arccos(np.clip(cosdis, -1, 1)) / np.pi

            s.append(dis)

            if iddb_test[i] == iddb_train[j]:  # same palm
                l.append(1)
            else:
                l.append(-1)

    if not os.path.exists(path_rst+'veriEER'):
        os.makedirs(path_rst+'veriEER')
    if not os.path.exists(path_rst+'veriEER/rank1_hard/'):
        os.makedirs(path_rst+'veriEER/rank1_hard/')

    with open(path_rst+'veriEER/scores_VeriEER.txt', 'w') as f:
        for i in range(len(s)):
            score = str(s[i])
            label = str(l[i])
            f.write(score + ' ' + label + '\n')

    sys.stdout.flush()
    os.system('python ./getGI.py' + '  ' + path_rst + 'veriEER/scores_VeriEER.txt scores_VeriEER')
    os.system('python ./getEER.py' + '  ' + path_rst + 'veriEER/scores_VeriEER.txt scores_VeriEER')

    print('\n------------------')
    print('Rank-1 acc of the test set...')
    # rank-1 acc
    cnt = 0
    corr = 0
    for i in range(ntest):
        probeID = iddb_test[i]

        dis = np.zeros((ntrain, 1))

        for j in range(ntrain):
            dis[j] = s[cnt]
            cnt += 1

        idx = np.argmin(dis[:])

        galleryID = iddb_train[idx]

        if probeID == galleryID:
            corr += 1

    rankacc = corr / ntest * 100
    print('rank-1 acc: %.3f%%' % rankacc)
    print('-----------')

    with open(path_rst + 'veriEER/rank1.txt', 'w') as f:
        f.write('rank-1 acc: %.3f%%' % rankacc)

    print('\n\nReal EER of the test set...')

def similarity_fes(fe_origi,fes):

    fe = fe_origi
    fe_origi = fe_origi.unsqueeze(0)
    similarities = torch.einsum('nbi, mbi -> nb', fes, fe_origi)

    similarities_t = similarities.T
    top2_indices = torch.topk(similarities_t, args.client_num - 1, dim=-1).indices  # [380, 2]

    batch_indices = torch.arange(fes.shape[1]).unsqueeze(1)  # [380, 1]
    fe1 = fes[top2_indices[:, 0], batch_indices[:, 0], :]  # [380, 512]
    fe2 = fes[top2_indices[:, 1], batch_indices[:, 0], :]  # [380, 512]
    fe3 = fes[top2_indices[:, 2], batch_indices[:, 0], :]  # [380, 512]

    fe_back = 0.8 * fe + 0.1 * fe1 + 0.1 * fe2

    return fe_back

def similarity_fes_op(fe_origi,fes):

    fe_origi = fe_origi.unsqueeze(0)
    similarities = torch.einsum('nbi, mbi -> nb', fes, fe_origi)

    similarities_t = similarities.T
    top2_indices = torch.topk(similarities_t, args.client_num - 1, dim=-1).indices  # [380, 2]

    batch_indices = torch.arange(fes.shape[1]).unsqueeze(1)  # [380, 1]
    fe1 = fes[top2_indices[:, 0], batch_indices[:, 0], :]  # [380, 512]
    fe2 = fes[top2_indices[:, 1], batch_indices[:, 0], :]  # [380, 512]
    fe3 = fes[top2_indices[:, 2], batch_indices[:, 0], :]  # [380, 512]

    fe_back = (fe1 + fe2 + fe3) / 3

    return fe_back

def fit_fedpalm(com, epoch, model, models, anchor_model, data_loader, now_id, anch_optimize = None, optimize=None, server_model=None, mode='fedavg', phase='training'):
    mu = 1e-4

    if phase != 'training' and phase != 'testing':
        raise TypeError('input error!')

    if phase == 'training':
        model.train()
        anchor_model.train()
        for idx in range(args.client_num):
            models[idx].train()

    if phase == 'testing':
        # print('test')
        model.eval()
        anchor_model.eval()
        for idx in range(args.client_num):
            models[idx].eval()

    running_loss = 0
    running_correct = 0
    ce_anchor_loss =0
    for batch_id, (datas, target) in enumerate(data_loader):

        data = datas[0]
        data = data.cuda()

        data_con = datas[1]
        data_con = data_con.cuda()

        target = target.cuda()
        if phase == 'training':
            ### Original Feature
            output, fe_origi, _ = model(data, target, None)  ### Anchor Feature
            # evis = []
            fes = []
            for idx in range(args.client_num):
                if idx == now_id:
                    continue
                ## use hyperparameter
                _, fe_1, gabor_out = models[idx](data, target, None)  ### Anchor Feature
                # _, fe_1, _ = model(data,target,gabor_out)

                # evis.append(evi_1)
                fes.append(fe_1)

            fes_stack = torch.stack(fes)
            sim_fe = similarity_fes(fe_origi,fes_stack)

            anch_output, ancho_fe, _ = anchor_model(data, target, None)
            # print('123')
            # final_fe = (sim_fe + ancho_fe) / 2
            final_fe = sim_fe * 0.2 + ancho_fe * 0.8
            
            ### Original Feature
            output2, fe_origi2, _ = model(data_con, target, None)  ### Anchor Feature
            # evis_2 = []
            fes_2 = []
            for idx in range(args.client_num):
                if idx == now_id:
                    continue
                _, fe_2, gabor_out2 = models[idx](data_con, target, None)  ### Anchor Feature
                # _, fe_2, _ =model(data_con, target, gabor_out2)
                # evis_2.append(evi2)
                fes_2.append(fe_2)

            fes_stack_2 = torch.stack(fes_2)
            sim_fe2 = similarity_fes(fe_origi2, fes_stack_2)

            anch_output2, ancho_fe2, _ = anchor_model(data_con, target, None)
            # final_fe2 = (sim_fe2 + ancho_fe2) / 2
            final_fe2 = sim_fe2 * 0.2 + ancho_fe2 * 0.8
            # evis = evis + 1
            fe = torch.cat([final_fe.unsqueeze(1), final_fe2.unsqueeze(1)], dim=1)
        else:
            with torch.no_grad():
                output, fe_origi, _ = model(data, None, None)  ### Anchor Feature
                # evis = []
                fes = []
                for idx in range(args.client_num):
                    if idx == now_id:
                        continue
                    _, fe_1, gabor_out = models[idx](data, None, None)  ### Anchor Feature
                    # _, fe_1, _ = model(data, None, gabor_out)
                    # evis.append(evi_1)
                    fes.append(fe_1)

                fes_stack = torch.stack(fes)
                sim_fe = similarity_fes(fe_origi,fes_stack)

                anch_output, ancho_fe, _ = anchor_model(data, None, None)
                # final_fe = (sim_fe + ancho_fe) / 2
                final_fe = sim_fe * 0.2 + ancho_fe * 0.8
                
                ### Original Feature
                output2, fe_origi2, _ = model(data_con, None, None)  ### Anchor Feature
                # evis_2 = []
                fes_2 = []
                for idx in range(args.client_num):
                    if idx == now_id:
                        continue
                    _, fe_2, _ = models[idx](data_con, None, None)  ### Anchor Feature
                    # _, fe_2, _ =model(data_con, None, gabor_out2)
                    # evis_2.append(evi2)
                    fes_2.append(fe_2)

                fes_stack_2 = torch.stack(fes_2)
                sim_fe2 = similarity_fes(fe_origi2, fes_stack_2)

                anch_output2, ancho_fe2, _ = anchor_model(data_con, None, None)
                # final_fe2 = (sim_fe2 + ancho_fe2) / 2
                final_fe2 = sim_fe2 * 0.2 + ancho_fe2 * 0.8
                # evis = evis + 1
                fe = torch.cat([final_fe.unsqueeze(1), final_fe2.unsqueeze(1)], dim=1)

        ce = criterion(output, target)
        ce_anchor = criterion(anch_output,target)

        ce2 = con_criterion(fe, target)
        loss = weight1 * ce + weight2 * ce2 +  weight1 * ce_anchor
        # if com < args.com // 2:
        #     loss = weight1 * ce + weight2 * ce2 +  weight1 * ce_anchor
        # else:
        #     loss = weight2 * ce2 +  weight1 * ce_anchor
        
        
        running_loss += loss.data.cpu().numpy()
        ce_anchor_loss += ce_anchor.data.cpu().numpy()
        preds = output.data.max(dim=1, keepdim=True)[1]  # max returns (value, index)
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum().numpy()


        if phase == 'training':
            # loss.backward(retain_graph=None)  #
            optimize.zero_grad()
            anch_optimize.zero_grad()
            loss.backward()
            optimize.step()
            anch_optimize.step()     
            # if com < args.com // 2:
            #     optimize.zero_grad()
            #     anch_optimize.zero_grad()
            #     loss.backward()
            #     optimize.step()
            #     anch_optimize.step()     
            # else:
            #     anch_optimize.zero_grad()
            #     loss.backward()
            #     anch_optimize.step() 

    total = len(data_loader.dataset)
    loss = running_loss / total
    an_loss = ce_anchor_loss / total
    accuracy = (100.0 * running_correct) / total
    
    if epoch % 10 == 0:
        print('epoch %d: \t%s loss is \t%7.5f; anc loss is \t%7.5f ;\t%s accuracy is \t%d/%d \t%7.3f%%' % (
            epoch, phase, loss, an_loss, phase, running_correct, total, accuracy))

    return loss, accuracy

def read_txt_file(file_path):
    paths = []
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            path, label = line.strip().split()
            paths.append(path)
            labels.append(label)
    return paths,labels

def check_duplicate_labels_in_sublists(data):

    seen_labels = set()
    

    for sublist_index, sublist in enumerate(data):
        for label in sublist:
            if label in seen_labels:
                print(f"Label {label} is repeated across sublist {sublist_index} and previous sublist(s).")
            else:
                seen_labels.add(label)

def Set_Dataset():

    train_path, train_label = read_txt_file(args.train_set_file)
    test_path, test_label = read_txt_file(args.test_set_file)
    
    train_label_to_paths = {label: [] for label in set(train_label)}
    for path, label in zip(train_path, train_label):
        train_label_to_paths[label].append(path)

    test_label_to_paths = {label: [] for label in set(test_label)}
    for path, label in zip(test_path, test_label):
        test_label_to_paths[label].append(path)

    all_ids = list(set(train_label))
    random.shuffle(all_ids)
    num_train_ids = int(len(all_ids) * args.ratio)
    train_ids = set(all_ids[:num_train_ids])
    test_ids = set(all_ids[num_train_ids:])

    node_ids = {i: [] for i in range(args.client_num)}
    all_train_ids = list(train_ids)
    random.shuffle(all_train_ids)
    
    for i, label in enumerate(all_train_ids):
        node_ids[i % args.client_num].append(label)  

    node_train_paths = {i: [] for i in range(args.client_num)}
    node_train_labels = {i: [] for i in range(args.client_num)}
    node_train_test_paths = {i: [] for i in range(args.client_num)}
    node_train_test_labels = {i: [] for i in range(args.client_num)}

    for node_id, labels in node_ids.items():
        for label in labels:

            node_train_paths[node_id].extend(train_label_to_paths[label])
            node_train_labels[node_id].extend([label] * len(train_label_to_paths[label]))

            node_train_test_paths[node_id].extend(test_label_to_paths[label])
            node_train_test_labels[node_id].extend([label] * len(test_label_to_paths[label]))

    ###  Set the Test Set
    all_test_ids = list(test_ids)

    test_gallery_paths = []
    test_gallery_labels = []
    test_query_paths = []
    test_query_labels = []

    for label in all_test_ids:
        test_gallery_paths.extend(train_label_to_paths[label])
        test_gallery_labels.extend([label] * len(train_label_to_paths[label]))

        test_query_paths.extend(test_label_to_paths[label])
        test_query_labels.extend([label] * len(test_label_to_paths[label]))
    
    # print('123')

    return node_train_paths, node_train_labels, node_train_test_paths, node_train_test_labels, \
        test_gallery_paths, test_gallery_labels, test_query_paths, test_query_labels

if __name__== "__main__" :

    print('123')
    set_seed(args.seed)
    batch_size = args.batch_size
    epoch_num = args.epoch_num
    num_classes = args.id_num  # IITD: 460    KTU: 145    Tongji: 600    REST: 358    XJTU: 200 POLYU 378
    weight1 = args.weight1
    weight2 = args.weight2
    communications = args.com
    ##Checkpoint Path
    print('seed:',args.seed)
    print('weight of cross:', weight1)
    print('weight of contra:', weight2)
    print('tempture:', args.temp)
    des_path = args.des_path
    if not os.path.exists(des_path):
        os.makedirs(des_path)

    path_rst = args.path_rst

    if not os.path.exists(path_rst):
        os.makedirs(path_rst)

    print('%s' % (time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))

    train_set_files, train_set_labels, val_set_files, val_set_labels, \
        test_gallery_paths, test_gallery_labels, test_query_paths, test_query_labels = Set_Dataset()

    Train_DataLoaders = []
    Test_DataLoaders = []
    
    ## Close-Set 
    Train_DataLoaders = [DataLoader(MyDataset_general_FL(train_set_files[i],train_set_labels[i], transforms=None, train=True, imside=128, outchannels=1), batch_size=batch_size, num_workers=2, shuffle=True) for i in range(args.client_num)]
    Val_DataLoaders = [DataLoader(MyDataset_general_FL(val_set_files[i],val_set_labels[i], transforms=None, train=True, imside=128, outchannels=1), batch_size=batch_size, num_workers=2, shuffle=True) for i in range(args.client_num)]
    
    ## Open-Set
    Test_Gallery_DataLoader = DataLoader(MyDataset_general_FL(test_gallery_paths, test_gallery_labels, transforms=None, train=True, imside=128, outchannels=1), batch_size=batch_size, num_workers=2, shuffle=True)
    Test_Query_DataLoader = DataLoader(MyDataset_general_FL(test_query_paths, test_query_labels, transforms=None, train=True, imside=128, outchannels=1), batch_size=batch_size, num_workers=2, shuffle=True)

    ## Different Models  CompNet CO3Net SCANet CCNet
    server_model = compnet_fedpalm(num_classes=num_classes).cuda()
    local_server_model = compnet_fedpalm(num_classes=num_classes).cuda()
    best_net = compnet_fedpalm(num_classes=num_classes).cuda()
    best_nets = [copy.deepcopy(best_net) for idx in range(args.client_num)]

    models = [copy.deepcopy(server_model) for idx in range(args.client_num)]
    if args.mode.lower() == 'moon':
        pre_local_models = [copy.deepcopy(server_model) for idx in range(args.client_num)]
    if args.mode.lower() == 'fedpalm':
        anchor_local_models = [copy.deepcopy(server_model) for idx in range(args.client_num)]
        anchor_optimizers = [torch.optim.Adam(anchor_local_models[idx].parameters(), lr=args.lr) for idx in range(args.client_num)]
        anch_schedulers = [lr_scheduler.StepLR(anchor_optimizers[idx], step_size=args.redstep, gamma=0.8) for idx in range(args.client_num)]

    ## Set the optimizer
    if args.mode.lower()== 'fednova':
        optimizers = [torch.optim.Adam(models[idx].parameters(), lr=args.lr) for idx in range(args.client_num)]
    else:
        optimizers = [torch.optim.Adam(models[idx].parameters(), lr=args.lr) for idx in range(args.client_num)]

    schedulers = [lr_scheduler.StepLR(optimizers[idx], step_size=args.redstep, gamma=0.8) for idx in range(args.client_num)]
    

    client_weights = [1 / args.client_num for i in range(args.client_num)]

    criterion = nn.CrossEntropyLoss()
    if args.loss.lower() == 'trus':
        con_criterion = SupConLoss(temperature=args.temp, base_temperature=args.temp)
    else:
        con_criterion = SupConLoss(temperature=args.temp, base_temperature=args.temp)

    train_losses, train_accuracy = [], []
    val_losses, val_accuracy = [], []
    
    bestacc = 0
    bestaccs = [0.] * args.client_num
    
    for com in range(communications):
        temp_val_acc = []
        for idx in range(args.client_num):

            for epoch in range(epoch_num):
                print("---Com:",com,"Client:",idx,"Epoch:",epoch,"---")
                
                epoch_loss, epoch_accuracy = fit_fedpalm(com, epoch, models[idx], models, anchor_local_models[idx], Train_DataLoaders[idx], now_id = idx, anch_optimize=anchor_optimizers[idx], optimize=optimizers[idx],\
                    server_model= server_model, mode=args.mode.lower(), phase='training')
                val_epoch_loss, val_epoch_accuracy = fit_fedpalm(com, epoch, models[idx], models, anchor_local_models[idx], Val_DataLoaders[idx], now_id = idx, phase='testing')
                
                schedulers[idx].step()
                anch_schedulers[idx].step()

                train_losses.append(epoch_loss)
                train_accuracy.append(epoch_accuracy)

                val_losses.append(val_epoch_loss)
                val_accuracy.append(val_epoch_accuracy)

                temp_val_acc.append(val_epoch_accuracy)

            if val_epoch_accuracy >= bestaccs[idx]:
                bestaccs[idx] = val_epoch_accuracy

                torch.save(models[idx].state_dict(), des_path + 'best_' + str(idx) + '_net_params_best.pth')
                for key in models[idx].state_dict().keys():
                    best_nets[idx].state_dict()[key].data.copy_(models[idx].state_dict()[key])

        server_model, anchor_local_models = communication(args, server_model, anchor_local_models, client_weights)

        val_epoch_accuracy = sum(temp_val_acc)/temp_val_acc.__len__()
        # save the best model
        if val_epoch_accuracy >= bestacc:
            bestacc = val_epoch_accuracy
            torch.save(server_model.state_dict(), des_path + 'net_params_best.pth')
            for key in server_model.state_dict().keys():
                best_net.state_dict()[key].data.copy_(server_model.state_dict()[key])

            for iii in range(args.client_num):
                torch.save(models[iii].state_dict(), des_path + '_id_' + str(iii) + '_net_params_best.pth')
                for key in models[iii].state_dict().keys():
                    best_nets[iii].state_dict()[key].data.copy_(models[iii].state_dict()[key])

        # save the current model and log info:
        if com % args.save_interval == 0 or com == (communications - 1) and com != 0:
            torch.save(server_model.state_dict(), des_path + 'net_params.pth')
            # for source_id in range(args.client_num):
            for iii in range(args.client_num):
                torch.save(models[iii].state_dict(), des_path + 'com_' + str(com) + '_id_' + str(iii) + '_net_params.pth')

            saveLossACC(train_losses, val_losses, train_accuracy, val_accuracy, bestacc, path_rst)

        if com % args.save_interval == 0:
            torch.save(server_model.state_dict(), des_path + 'com_' + str(com) + '_net_params.pth')
            for iii in range(args.client_num):
                torch.save(models[iii].state_dict(), des_path + 'com_' + str(com) + '_id_' + str(iii) + '_net_params.pth')

        if com % args.test_interval == 0 and com != 0:
            # test_op(server_model, models, Test_Gallery_DataLoader, Test_Query_DataLoader, path_rst + 'Open-Set/LAST/')
            # for source_id in range(args.client_num):
            #     for target_id in range(4):
            #         print(names[source_id],'->',names[target_id])
            #         path_rst = args.path_rst + names[source_id] + '2' + names[target_id] + '_best/'
            for source_id in range(args.client_num):
                print("---------------YANG_TOGETHER Client:",source_id,"---------------")
                test_cl(server_model, models, source_id, Train_DataLoaders[source_id],Val_DataLoaders[source_id], path_rst + 'Close-Set/')
  
            print("--------------- OPEN SET ---------------")
            test_op(server_model, models, Test_Gallery_DataLoader, Test_Query_DataLoader, path_rst + 'Open-Set/LAST/')

    print('------------ LAST CLOSE-SET ------------\n')

    for source_id in range(args.client_num):
        print("--------------- YANG_TOGETHER Client:",source_id,"---------------")
        test_cl(server_model,models, source_id,Train_DataLoaders[source_id],Val_DataLoaders[source_id], path_rst + 'Close-Set/LAST/')

        
    print('------------ LAST OPEN-SET ------------\n')   
    test_op(server_model, models, Test_Gallery_DataLoader, Test_Query_DataLoader, path_rst + 'Open-Set/LAST/')    

    print('------------ BEST CLOSE-SET ------------\n')

    for source_id in range(args.client_num):
        print("-------------- YANG_TOGETHER Client:",source_id,"---------------")
        test_cl(best_net, best_nets, source_id, Train_DataLoaders[source_id],Val_DataLoaders[source_id], path_rst + 'Close-Set/BEST/')

        
    print('------------ BEST OPEN-SET ------------\n')   
    test_op(best_net, best_nets, Test_Gallery_DataLoader, Test_Query_DataLoader, path_rst + 'Open-Set/BEST/')