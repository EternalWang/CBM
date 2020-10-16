import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchFewShot.models.resnet12 import resnet12
from sklearn.manifold import LocallyLinearEmbedding
import pickle
import numpy as np
import os.path as osp


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.temperature = args.temperature
        self.base = resnet12()
        self.nFeat = self.base.nFeat
        self.clasifier = nn.Conv2d(self.nFeat, args.num_classes, kernel_size=1)
        self.args = args
        if(args.method in {'CBM', 'CBM_LLE'}):
            with open(osp.join(args.save_dir, 'base_proto.pickle'), 'rb') as fo:
                self.base_proto = pickle.load(fo)  # [64 512]
            if(args.method == 'CBM_LLE'):
                self.LLE = LocallyLinearEmbedding(
                    n_neighbors=args.k, n_components=args.dim)
                if(args.L2):
                    self.base_proto = F.normalize(self.base_proto, p=2, dim=-1)
                self.base_proto = torch.from_numpy(
                    self.LLE.fit_transform(self.base_proto.cpu().numpy())).cuda()
            self.base_proto = self.base_proto.unsqueeze(0)
            if(self.args.similarityOnBase == 'cosine'):
                self.base_proto = F.normalize(self.base_proto, p=2, dim=-1)

    def test(self, ftrain, ftest, batch_size, num_way, num_test):
        ftrain = ftrain.mean((-1, -2))
        ftest = ftest.mean((-1, -2))
        phi = self.calPhi(ftrain, ftest, batch_size, num_way, num_test)
        if(self.args.method in {'CBM', 'CBM_LLE'}):
            varPhi = self.calVarPhi(
                ftrain, ftest, batch_size, num_way, num_test)
            return self.args.alpha*phi+(1-self.args.alpha)*varPhi  # [4 30 5]
        else:
            return phi

    def calPhi(self, ftrain, ftest, batch_size, num_way, num_test):
        ftrain = ftrain.view(batch_size, 1, num_way, -1)
        ftest = ftest.view(batch_size, num_test, 1, -1)
        ftrain = F.normalize(ftrain, p=2, dim=-1)
        ftest = F.normalize(ftest, p=2, dim=-1)
        scores = torch.sum(ftest * ftrain, dim=-1)  # [4 30 5]
        return scores

    def calVarPhi(self, ftrain, ftest, batch_size, num_way, num_test):
        if(self.args.method == 'CBM_LLE'):
            if(self.args.L2):
                ftrain = F.normalize(ftrain, p=2, dim=-1)
                ftest = F.normalize(ftest, p=2, dim=-1)
            ftrain = torch.from_numpy(self.LLE.transform(
                ftrain.cpu().numpy())).cuda()
            ftest = torch.from_numpy(self.LLE.transform(
                ftest.cpu().numpy())).cuda()
        ftrain = ftrain.unsqueeze(1)
        ftest = ftest.unsqueeze(1)
        if(self.args.similarityOnBase == 'cosine'):
            ftrain = F.normalize(ftrain, p=2, dim=-1)
            ftrain = (ftrain*self.base_proto).sum(-1)
            ftest = F.normalize(ftest, p=2, dim=-1)
            ftest = (ftest*self.base_proto).sum(-1)
        else:  # Euclidean
            ftrain = -(ftrain-self.base_proto).norm(dim=-1)
            ftest = -(ftest-self.base_proto).norm(dim=-1)
        if(self.args.softmax):
            ftrain = F.softmax(ftrain, dim=-1)
            ftest = F.softmax(ftest, dim=-1)
        if(self.args.similarityOfDistribution == 'cosine'):
            ftrain = F.normalize(
                ftrain, p=2, dim=-1).view(batch_size, 1, num_way, -1)
            ftest = F.normalize(
                ftest, p=2, dim=-1).view(batch_size, num_test, 1, -1)
            scores = (ftrain*ftest).sum(-1)
        elif(self.args.similarityOfDistribution == 'Euclidean'):
            ftrain = F.normalize(
                ftrain, p=2, dim=-1).view(batch_size, 1, num_way, -1)
            ftest = F.normalize(
                ftest, p=2, dim=-1).view(batch_size, num_test, 1, -1)
            scores = -(ftrain-ftest).norm(dim=-1)
        else:  # KL
            ftrain = F.softmax(ftrain, dim=-1).view(batch_size, 1, num_way, -1)
            ftest = F.softmax(ftest, dim=-1).view(batch_size,
                                                  num_test, 1, -1).log()
            scores = -(ftrain*(ftrain.log()-ftest)).sum(dim=-1)
        return scores

    def forward(self, xtrain, xtest, ytrain, ytest):
        batch_size, num_train = xtrain.size(0), xtrain.size(1)
        num_test = xtest.size(1)
        num_way = ytrain.size(2)
        ytrain = ytrain.transpose(1, 2)
        xtrain = xtrain.view(-1, xtrain.size(2),
                             xtrain.size(3), xtrain.size(4))
        xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))
        x = torch.cat((xtrain, xtest), 0)
        f = self.base(x)
        ftrain = f[:batch_size * num_train]
        ftrain = ftrain.view(batch_size, num_train, -1)
        ftrain = torch.bmm(ytrain, ftrain)
        ftrain = ftrain.div(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain))
        ftrain = ftrain.view(-1, *f.size()[1:])  # [4*5 512 6 6]
        ftest = f[batch_size * num_train:]
        ftest = ftest.view(-1, *f.size()[1:])  # [4*30 512 6 6]
        if not self.training:
            score = self.test(ftrain, ftest, batch_size, num_way, num_test)
            # score = score.view(batch_size*num_test, num_way)
            return score
        else:
            ytest = self.clasifier(ftest) * self.temperature  # [4*30 64 6 6]
            return ytest
