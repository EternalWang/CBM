import pickle
import os.path as osp
from PIL import Image
from torchFewShot import transforms as T
import torch
from torchFewShot.models.net import Model
from args_mini import add_arguments as add_arguments_mini
from args_tiered import add_arguments as add_arguments_tiered
import argparse
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()
parser_mini = subparsers.add_parser('mini')
add_arguments_mini(parser_mini)
args = parser.parse_args()


def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)
    return label2inds


file = '../FewShotWithoutForgetting-master/datasets/MiniImagenet/miniImageNet_category_split_train_phase_train.pickle'
with open(file, 'rb') as fo:
    dataset = pickle.load(fo, encoding='iso-8859-1')
data = dataset['data']
labels = dataset['labels']
labels2inds = buildLabelIndex(labels)
labelIds = sorted(labels2inds.keys())  # (0~63)
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
model = Model(args)
checkpoint = torch.load(osp.join(args.resume, str(
    args.nExemplars)+'-shot', 'best_model.pth.tar'))
model.load_state_dict(checkpoint['state_dict'])
model = model.cuda()
model.eval()
feas = []
with torch.no_grad():
    for i in labelIds:
        ids = labels2inds[i]
        imgs = []
        for j in ids:
            img = data[j]
            img = Image.fromarray(img)
            img = transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs, dim=0)  # [100 3 84 84]
        imgs = imgs.cuda()
        fea = model.base(imgs)  # [100 512 6 6]
        fea = fea.mean((0, 2, 3))
        feas.append(fea)
feas = torch.stack(feas, dim=0)
print(feas.size())
file = osp.join(args.save_dir, str(args.nExemplars)+'-shot')
file = file+'/base_proto.pickle'
with open(file, 'wb') as f:
    pickle.dump(feas, f)
with open(file, 'rb') as fo:
    features = pickle.load(fo)
    print(features)
