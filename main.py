from sklearn.random_projection import SparseRandomProjection
from sampling_methods.kcenter_greedy import kCenterGreedy
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
from torchvision import transforms
import pytorch_lightning as pl
from PIL import Image
import numpy as np
import argparse
import shutil
import faiss
import torch
import glob
import cv2
import os
import torch.optim as optim
from tqdm import tqdm
import utils
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix
import pickle
from sampling_methods.kcenter_greedy import kCenterGreedy
from sklearn.random_projection import SparseRandomProjection
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import gaussian_filter


def load_weights(filepath):
    path = os.path.join(filepath)
    model = torch.load(path + '_model.pt')
    state = torch.load(path + '_model_state_dict.pt')
    model.load_state_dict(state)
    print('Loading weights from {}'.format(filepath))
    return model

def contrastive_loss(out_1, out_2):
    out_1 = F.normalize(out_1, dim=-1)
    out_2 = F.normalize(out_2, dim=-1)
    bs = out_1.size(0)
    temp = 0.25
    # [2*B, D]
    out = torch.cat([out_1, out_2], dim=0)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temp)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * bs, device=sim_matrix.device)).bool()
    # [2B, 2B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * bs, -1)

    # compute loss
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temp)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss


def train_model(model, train_loader, test_loader, train_loader_1, device, args):
    model.eval()
    if args.dataset == 'cifar10':
        auc, feature_space = get_score(model, device, train_loader, test_loader)
        print('Epoch: {}, AUROC is: {}'.format(0, auc))
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.00005)
        center = torch.FloatTensor(feature_space).mean(dim=0)
        if args.angular:
            center = F.normalize(center, dim=-1)
        center = center.to(device)
        for epoch in range(args.epochs):
            running_loss = run_epoch(model, train_loader_1, optimizer, center, device, args.angular)
            print('Epoch: {}, Loss: {}'.format(epoch + 1, running_loss))
            auc, _ = get_score(model, device, train_loader, test_loader)
            print('Epoch: {}, AUROC is: {}'.format(epoch + 1, auc))
        torch.save(model, args.filepath + '_model.pt')
        torch.save(model.state_dict(), args.filepath + '_model_state_dict.pt')
        return auc
    elif args.dataset == 'mvtec':
        auc, feature_space = get_score_mvtec(model, device, train_loader, test_loader)
        print('Epoch: {}, AUROC is: {}'.format(0, auc))
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.00005)
        center = torch.FloatTensor(feature_space).mean(dim=0)
        if args.angular:
            center = F.normalize(center, dim=-1)
        center = center.to(device)
        for epoch in range(args.epochs):
            running_loss = run_epoch_mvtec(model, train_loader_1, optimizer, center, device, args.angular)
            print('Epoch: {}, Loss: {}'.format(epoch + 1, running_loss))
            auc, _ = get_score_mvtec(model, device, train_loader, test_loader)
            print('Epoch: {}, AUROC is: {}'.format(epoch + 1, auc))
        torch.save(model, args.filepath + '_model.pt')
        torch.save(model.state_dict(), args.filepath + '_model_state_dict.pt')
        return auc
    else:
        print("Unsupported dataset! \n")
        exit()

def test_model(model, train_loader, test_loader, device, args):
    model.eval()
    if args.dataset == 'mvtec':
        auc, _ = get_score_mvtec(model, device, train_loader, test_loader)
    elif args.dataset == 'cifar10':
        auc, _ = get_score(model, device, train_loader, test_loader)
    else: 
        print("Unsupported dataset! \n")
        exit()
    print('Epoch: {}, AUROC is: {}'.format(0, auc))
    return auc

def run_epoch(model, train_loader, optimizer, center, device, is_angular):
    total_loss, total_num = 0.0, 0
    for ((img1, img2), _) in tqdm(train_loader, desc='Train...'):
        img1, img2 = img1.to(device), img2.to(device)

        optimizer.zero_grad()

        out_1 = model(img1)
        out_2 = model(img2)
        out_1 = out_1 - center
        out_2 = out_2 - center

        loss = contrastive_loss(out_1, out_2)

        if is_angular:
            loss += ((out_1 ** 2).sum(dim=1).mean() + (out_2 ** 2).sum(dim=1).mean())

        loss.backward()

        optimizer.step()

        total_num += img1.size(0)
        total_loss += loss.item() * img1.size(0)

    return total_loss / (total_num)

def run_epoch_mvtec(model, train_loader, optimizer, center, device, is_angular):
    total_loss, total_num = 0.0, 0
    for ((img1, img2), _, _) in tqdm(train_loader, desc='Train...'):
        img1, img2 = img1.to(device), img2.to(device)

        optimizer.zero_grad()

        out_1 = model(img1)
        out_2 = model(img2)
        out_1 = out_1 - center
        out_2 = out_2 - center

        loss = contrastive_loss(out_1, out_2)

        if is_angular:
            loss += ((out_1 ** 2).sum(dim=1).mean() + (out_2 ** 2).sum(dim=1).mean())

        loss.backward()

        optimizer.step()

        total_num += img1.size(0)
        total_loss += loss.item() * img1.size(0)

    return total_loss / (total_num)

def get_score(model, device, train_loader, test_loader):
    train_feature_space = []
    with torch.no_grad():
        for (imgs, _) in tqdm(train_loader, desc='Train set feature extracting'):
            imgs = imgs.to(device)
            features = model(imgs)
            train_feature_space.append(features)
        train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()
    test_feature_space = []
    test_labels = []
    with torch.no_grad():
        for (imgs, labels) in tqdm(test_loader, desc='Test set feature extracting'):
            imgs = imgs.to(device)
            features = model(imgs)
            test_feature_space.append(features)
            test_labels.append(labels)
        test_feature_space = torch.cat(test_feature_space, dim=0).contiguous().cpu().numpy()
        test_labels = torch.cat(test_labels, dim=0).cpu().numpy()

    distances = utils.knn_score(train_feature_space, test_feature_space)

    auc = roc_auc_score(test_labels, distances)

    return auc, train_feature_space

def get_score_mvtec(model, device, train_loader, test_loader):
    train_feature_space = []
    with torch.no_grad():
        for (imgs, _, _) in tqdm(train_loader, desc='Train set feature extracting'):
            imgs = imgs.to(device) # (64, 3, 224, 224)
            features = model(imgs) # (64, 2048)
            train_feature_space.append(features)
        train_feature_space = torch.cat(train_feature_space, dim=0) # (192, 2048)

    test_feature_space = []
    test_labels = []
    gt_mask_list = []
    score_map_list = []
    with torch.no_grad():
        for (imgs, labels, mask) in tqdm(test_loader, desc='Test set feature extracting'):
            imgs = imgs.to(device)
            features = model(imgs)
            test_feature_space.append(features)
            test_labels.append(labels)
            # gt_mask_list.extend(mask.cpu().detach().numpy())
        test_feature_space = torch.cat(test_feature_space, dim=0) # (132, 2048)
        test_labels = torch.cat(test_labels, dim=0) # (132,)
        
        # for t_idx in tqdm(range(test_feature_space.shape[0]), desc='MVTec Localization'):
        #     feat_map = train_feature_space[t_idx].unsqueeze(-1)
        #     test_map = test_feature_space[t_idx]
        #     for d_idx in range(feat_map.shape[0]):
        #         dist_matrix = torch.pairwise_distance(feat_map[d_idx:], test_map)
        #         dist_matrix = F.interpolate(dist_matrix.unsqueeze(0).unsqueeze(0), size=224*224,
        #                                   mode='linear', align_corners=False) 
        #         score_map_list.append(dist_matrix)
        #     # dist_matrix = torch.cat(dist_matrix_list, 0)
            

    train_feature_space = train_feature_space.contiguous().cpu().numpy()
    test_feature_space = test_feature_space.contiguous().cpu().numpy() # (132, 2048)
    test_labels = test_labels.cpu().numpy()
    # flatten_gt_mask_list = np.concatenate(gt_mask_list).ravel()
    # flatten_score_map_list = np.concatenate(score_map_list).ravel()

    distances = utils.knn_score(train_feature_space, test_feature_space) # (132, )

    auc = roc_auc_score(test_labels, distances)
    # per_pixel_rocauc = roc_auc_score(flatten_gt_mask_list, flatten_score_map_list)

    return auc, train_feature_space

def distance_matrix(x, y=None, p=2):  # pairwise distance of vectors

    y = x if type(y) == type(None) else y

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    dist = torch.pow(x - y, p).sum(2)

    return dist


class NN():

    def __init__(self, X=None, Y=None, p=2):
        self.p = p
        self.train(X, Y)

    def train(self, X, Y):
        self.train_pts = X
        self.train_label = Y

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")

        dist = distance_matrix(x, self.train_pts, self.p) ** (1 / self.p)
        labels = torch.argmin(dist, dim=1)
        return self.train_label[labels]

class KNN(NN):

    def __init__(self, X=None, Y=None, k=3, p=2):
        self.k = k
        super().__init__(X, Y, p)

    def train(self, X, Y):
        super().train(X, Y)
        if type(Y) != type(None):
            self.unique_labels = self.train_label.unique()

    def predict(self, x):


        # dist = distance_matrix(x, self.train_pts, self.p) ** (1 / self.p)
        dist = torch.cdist(x, self.train_pts, self.p)

        knn = dist.topk(self.k, largest=False)


        return knn

def copy_files(src, dst, ignores=[]):
    src_files = os.listdir(src)
    for file_name in src_files:
        ignore_check = [True for i in ignores if i in file_name]
        if ignore_check:
            continue
        full_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, os.path.join(dst,file_name))
        if os.path.isdir(full_file_name):
            os.makedirs(os.path.join(dst, file_name), exist_ok=True)
            copy_files(full_file_name, os.path.join(dst, file_name), ignores)

def prep_dirs(root, dataset, perc):
    # make embeddings dir
    # embeddings_path = os.path.join(root, 'embeddings')
    if dataset == 'mvtec':
        embeddings_path = os.path.join('./', 'embeddings_'+str(dataset)+'_'+str(perc), args.class_name, str(args.coreset_sampling_ratio), str(args.batch_size), str(args.input_size), str(args.load_size))
    elif dataset == 'cifar10':
        embeddings_path = os.path.join('./', 'embeddings_cifar10', str(args.coreset_sampling_ratio), str(args.batch_size), str(args.input_size), str(args.load_size))
    os.makedirs(embeddings_path, exist_ok=True)
    # make sample dir
    sample_path = os.path.join(root, 'sample')
    os.makedirs(sample_path, exist_ok=True)
    # make source code record dir & copy
    source_code_save_path = os.path.join(root, 'src')
    os.makedirs(source_code_save_path, exist_ok=True)
    # copy_files('./', source_code_save_path, ['.git','.vscode','__pycache__','logs','README','samples','LICENSE']) # copy source code
    return embeddings_path, sample_path, source_code_save_path

def prep_dirs_cifar(root):
    # make embeddings dir
    embeddings_path = os.path.join('./', 'embeddings_cifar10', str(args.coreset_sampling_ratio), str(args.batch_size), str(args.input_size), str(args.load_size))
    os.makedirs(embeddings_path, exist_ok=True)
    # make sample dir
    sample_path = os.path.join(root, 'sample')
    os.makedirs(sample_path, exist_ok=True)
    # make source code record dir & copy
    source_code_save_path = os.path.join(root, 'src')
    os.makedirs(source_code_save_path, exist_ok=True)
    copy_files('./', source_code_save_path, ['.git','.vscode','__pycache__','logs','README','samples','LICENSE']) # copy source code
    return embeddings_path, sample_path, source_code_save_path

def embedding_concat(x, y):
    # from https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z

def reshape_embedding(embedding):
    embedding_list = []
    for k in range(embedding.shape[0]):
        for i in range(embedding.shape[2]):
            for j in range(embedding.shape[3]):
                embedding_list.append(embedding[k, :, i, j])
    return embedding_list

#imagenet
mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]

class MVTecDataset(Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase=='train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset() # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)
        
        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0]*len(img_paths))
                tot_labels.extend([0]*len(img_paths))
                tot_types.extend(['good']*len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1]*len(img_paths))
                tot_types.extend([defect_type]*len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"
        
        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)
        
        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, os.path.basename(img_path[:-4]), img_type

class CIFAR10_Dataset(Dataset):
    def __init__(self, root, transform, phase, normal):
        self.img_path = os.path.join(root, 'cifar-10-batches-py')
        if phase=='train':
            self.is_train = True
        else:
            self.is_train = False
            # self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        # self.gt_transform = gt_transform
        # load dataset

        # Define normal and abnormal classes
        self.normal_classes = tuple([normal])
        self.abnormal_classes = list(range(0,10))
        self.abnormal_classes.remove(normal)
        self.abnormal_classes = tuple(self.abnormal_classes)

        self.img_paths, self.labels = self.load_dataset_folder() # self.labels => good : 0, anomaly : 1


    def load_dataset_folder(self):
        train_data_dir = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        test_data_dir = ['test_batch']
        if self.is_train:
            data_dir = train_data_dir
        else:
            data_dir = test_data_dir

        x, y = [], []

        for dirname in data_dir:
            img_dir = os.path.join(self.img_path, dirname)
            with open(img_dir, 'rb') as f:
                batch_dict = pickle.load(f, encoding='latin1')
                # get labels (y) & get normal indexes (for train data)
                if self.is_train:
                    idx = np.argwhere(np.isin(batch_dict['labels'], self.normal_classes))
                    idx = idx.flatten().tolist()
                    y.extend([0] * len(idx))
                else:
                    cls_cnt = [0,0,0,0,0,0,0,0,0,0]
                    for class_num in batch_dict['labels']:
                        if class_num in self.normal_classes:
                            y.extend([0])
                        else:
                            if cls_cnt[class_num] >= 100 and class_num not in self.normal_classes:
                                continue
                            y.extend([1])
                            cls_cnt[class_num]  = cls_cnt[class_num] + 1
                                
                # get data (x)
                if self.is_train:
                    for i in idx:
                        x.append(np.reshape(batch_dict['data'][i], [3,32,32]))
                else:
                    cls_cnt = [0,0,0,0,0,0,0,0,0,0]
                    i = 0
                    for d in batch_dict['data']:
                        class_num = batch_dict['labels'][i]
                        i = i + 1
                        if cls_cnt[class_num] >= 100 and class_num not in self.normal_classes:
                            continue
                        x.append(np.reshape(d, [3,32,32]))
                        cls_cnt[class_num] = cls_cnt[class_num] + 1
        

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y)

    def __len__(self):
        return len(self.img_paths)


    def __getitem__(self, idx):
        img, label = self.img_paths[idx], self.labels[idx]
        img = np.transpose(img, (1,2,0))
        img = Image.fromarray(img, 'RGB')
        img = self.transform(img)

        return img, label

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def heatmap_on_image(heatmap, image):
    if heatmap.shape != image.shape:
        heatmap = cv2.resize(heatmap, (image.shape[0], image.shape[1]))
    out = np.float32(heatmap)/255 + np.float32(image)/255
    out = out / np.max(out)
    return np.uint8(255 * out)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)    


def cal_confusion_matrix(y_true, y_pred_no_thresh, thresh, img_path_list):
    pred_thresh = []
    false_n = []
    false_p = []
    for i in range(len(y_pred_no_thresh)):
        if y_pred_no_thresh[i] > thresh:
            pred_thresh.append(1)
            if y_true[i] == 0:
                false_p.append(img_path_list[i])
        else:
            pred_thresh.append(0)
            if y_true[i] == 1:
                false_n.append(img_path_list[i])

    cm = confusion_matrix(y_true, pred_thresh)
    print(cm)
    print('false positive')
    print(false_p)
    print('false negative')
    print(false_n)
    

class STPM_mvtec(pl.LightningModule):
    def __init__(self, hparams):
        super(STPM_mvtec, self).__init__()

        self.save_hyperparameters(hparams)
        self.dataset = hparams.dataset
        self.percent = hparams.percent
        self.init_features()
        def hook_t(module, input, output):
            self.features.append(output)

        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'wide_resnet50_2', pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.layer2[-1].register_forward_hook(hook_t)
        self.model.layer3[-1].register_forward_hook(hook_t)

        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.init_results_list()

        self.data_transforms = transforms.Compose([
                        transforms.Resize((args.load_size, args.load_size), Image.ANTIALIAS),
                        transforms.ToTensor(),
                        transforms.CenterCrop(args.input_size),
                        transforms.Normalize(mean=mean_train,
                                            std=std_train)])
        self.gt_transforms = transforms.Compose([
                        transforms.Resize((args.load_size, args.load_size)),
                        transforms.ToTensor(),
                        transforms.CenterCrop(args.input_size)])

        self.inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255], std=[1/0.229, 1/0.224, 1/0.255])

    def init_results_list(self):
        self.gt_list_px_lvl = []
        self.pred_list_px_lvl = []
        self.gt_list_img_lvl = []
        self.pred_list_img_lvl = []
        self.img_path_list = []        

    def init_features(self):
        self.features = []

    def forward(self, x_t):
        self.init_features()
        _ = self.model(x_t)
        return self.features

    def save_anomaly_map(self, anomaly_map, input_img, gt_img, file_name, x_type):
        if anomaly_map.shape != input_img.shape:
            anomaly_map = cv2.resize(anomaly_map, (input_img.shape[0], input_img.shape[1]))
        anomaly_map_norm = min_max_norm(anomaly_map)
        anomaly_map_norm_hm = cvt2heatmap(anomaly_map_norm*255)

        # anomaly map on image
        heatmap = cvt2heatmap(anomaly_map_norm*255)
        hm_on_img = heatmap_on_image(heatmap, input_img)

        # save images
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}.jpg'), input_img)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_amap.jpg'), anomaly_map_norm_hm)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_amap_on_img.jpg'), hm_on_img)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_gt.jpg'), gt_img)

    def train_dataloader(self):
        image_datasets = MVTecDataset(root=os.path.join(args.dataset_mvtec_path,args.class_name), transform=self.data_transforms, gt_transform=self.gt_transforms, phase='train')    
        train_loader = DataLoader(image_datasets, batch_size=args.batch_size, shuffle=True, num_workers=0) #, pin_memory=True)
        return train_loader

    def test_dataloader(self):
        test_datasets = MVTecDataset(root=os.path.join(args.dataset_mvtec_path,args.class_name), transform=self.data_transforms, gt_transform=self.gt_transforms, phase='test')
        test_loader = DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=0) #, pin_memory=True) # only work on batch_size=1, now.
        return test_loader

    def configure_optimizers(self):
        return None

    def on_train_start(self):
        self.model.eval() # to stop running_var move (maybe not critical)
        self.embedding_dir_path, self.sample_path, self.source_code_save_path = prep_dirs(self.logger.log_dir, self.dataset, self.percent)
        self.embedding_list = []
    
    def on_test_start(self):
        self.embedding_dir_path, self.sample_path, self.source_code_save_path = prep_dirs(self.logger.log_dir, self.dataset, self.percent)
        self.index = faiss.read_index(os.path.join(self.embedding_dir_path,'index.faiss'))
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0 ,self.index)
        self.init_results_list()
        
    def training_step(self, batch, batch_idx): # save locally aware patch features
        x, _, _, file_name, _ = batch
        features = self(x)
        embeddings = []
        for feature in features:
            m = torch.nn.AvgPool2d(3, 1, 1)
            embeddings.append(m(feature))
        embedding = embedding_concat(embeddings[0], embeddings[1])
        self.embedding_list.extend(reshape_embedding(np.array(embedding)))

    def training_epoch_end(self, outputs): 
        total_embeddings = np.array(self.embedding_list)
        # Random projection
        self.randomprojector = SparseRandomProjection(n_components='auto', eps=0.9) # 'auto' => Johnson-Lindenstrauss lemma
        self.randomprojector.fit(total_embeddings)
        # Coreset Subsampling
        selector = kCenterGreedy(total_embeddings,0,0)
        selected_idx = selector.select_batch(model=self.randomprojector, already_selected=[], N=int(total_embeddings.shape[0]*args.coreset_sampling_ratio))
        self.embedding_coreset = total_embeddings[selected_idx]
        
        print('initial embedding size : ', total_embeddings.shape)
        print('final embedding size : ', self.embedding_coreset.shape)
        #faiss
        self.index = faiss.IndexFlatL2(self.embedding_coreset.shape[1])
        self.index.add(self.embedding_coreset) 
        faiss.write_index(self.index,  os.path.join(self.embedding_dir_path,'index.faiss'))


    def test_step(self, batch, batch_idx): # Nearest Neighbour Search
        x, gt, label, file_name, x_type = batch
        # extract embedding
        features = self(x)
        embeddings = []
        for feature in features:
            m = torch.nn.AvgPool2d(3, 1, 1)
            embeddings.append(m(feature))
        embedding_ = embedding_concat(embeddings[0], embeddings[1])
        embedding_test = np.array(reshape_embedding(np.array(embedding_)))
        score_patches, _ = self.index.search(embedding_test , k=args.n_neighbors)
        anomaly_map = score_patches[:,0].reshape((28,28))
        N_b = score_patches[np.argmax(score_patches[:,0])]
        w = (1 - (np.max(np.exp(N_b))/np.sum(np.exp(N_b))))
        score = w*max(score_patches[:,0]) # Image-level score
        gt_np = gt.cpu().numpy()[0,0].astype(int)
        anomaly_map_resized = cv2.resize(anomaly_map, (args.input_size, args.input_size))
        anomaly_map_resized_blur = gaussian_filter(anomaly_map_resized, sigma=4)
        self.gt_list_px_lvl.extend(gt_np.ravel())
        self.pred_list_px_lvl.extend(anomaly_map_resized_blur.ravel())
        self.gt_list_img_lvl.append(label.cpu().numpy()[0])
        self.pred_list_img_lvl.append(score)
        self.img_path_list.extend(file_name)
        # save images
        x = self.inv_normalize(x)
        input_x = cv2.cvtColor(x.permute(0,2,3,1).cpu().numpy()[0]*255, cv2.COLOR_BGR2RGB)
        self.save_anomaly_map(anomaly_map_resized_blur, input_x, gt_np*255, file_name[0], x_type[0])

    def test_epoch_end(self, outputs):
        print("Total pixel-level auc-roc score :")
        pixel_auc = roc_auc_score(self.gt_list_px_lvl, self.pred_list_px_lvl)
        print(pixel_auc)
        print("Total image-level auc-roc score :")
        img_auc = roc_auc_score(self.gt_list_img_lvl, self.pred_list_img_lvl)
        print(img_auc)
        print('test_epoch_end')
        values = {'pixel_auc': pixel_auc, 'img_auc': img_auc}
        self.values = values
        self.log_dict(values)

class STPM_cifar10(pl.LightningModule):
    def __init__(self, hparams):
        super(STPM_cifar10, self).__init__()

        self.save_hyperparameters(hparams)

        self.init_features()
        def hook_t(module, input, output):
            self.features.append(output)

        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'wide_resnet50_2', pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.layer2[-1].register_forward_hook(hook_t)
        self.model.layer3[-1].register_forward_hook(hook_t)

        self.criterion = torch.nn.MSELoss(reduction='sum')

        self.init_results_list()

        self.normal = hparams.label
        self.data_transforms = transforms.Compose([
                        transforms.Resize((args.load_size, args.load_size), Image.ANTIALIAS),
                        transforms.ToTensor(),
                        transforms.CenterCrop(args.input_size),
                        transforms.Normalize(mean=mean_train,
                                            std=std_train)])
        self.gt_transforms = transforms.Compose([
                        transforms.Resize((args.load_size, args.load_size)),
                        transforms.ToTensor(),
                        transforms.CenterCrop(args.input_size)])

        self.inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255], std=[1/0.229, 1/0.224, 1/0.255])

    def init_results_list(self):
        # self.gt_list_px_lvl = []
        # self.pred_list_px_lvl = []
        self.gt_list_img_lvl = []
        self.pred_list_img_lvl = []
        self.img_path_list = []        

    def init_features(self):
        self.features = []

    def forward(self, x_t):
        self.init_features()
        _ = self.model(x_t)
        return self.features

    def save_anomaly_map(self, anomaly_map, input_img, gt_img, file_name, x_type):
        if anomaly_map.shape != input_img.shape:
            anomaly_map = cv2.resize(anomaly_map, (input_img.shape[0], input_img.shape[1]))
        anomaly_map_norm = min_max_norm(anomaly_map)
        anomaly_map_norm_hm = cvt2heatmap(anomaly_map_norm*255)

        # anomaly map on image
        heatmap = cvt2heatmap(anomaly_map_norm*255)
        hm_on_img = heatmap_on_image(heatmap, input_img)

        # save images
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}.jpg'), input_img)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_amap.jpg'), anomaly_map_norm_hm)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_amap_on_img.jpg'), hm_on_img)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_gt.jpg'), gt_img)

    def train_dataloader(self):
        image_datasets = CIFAR10_Dataset(root=os.path.join(args.dataset_cifar10_path), transform=self.data_transforms, phase='train', normal=self.normal)
        train_loader = DataLoader(image_datasets, batch_size=args.batch_size, shuffle=True, num_workers=0) #, pin_memory=True)
        print("len(train_dataset): ", len(image_datasets))
        print("len(train_loader): ", len(train_loader))
        return train_loader

    def test_dataloader(self):
        test_datasets = CIFAR10_Dataset(root=os.path.join(args.dataset_cifar10_path), transform=self.data_transforms, phase='test', normal=self.normal)
        test_loader = DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=0) #, pin_memory=True) # only work on batch_size=1, now.
        print("len(test_dataset): ", len(test_datasets))
        print("len(test_loader): ", len(test_loader))
        return test_loader

    def configure_optimizers(self):
        return None

    def on_train_start(self):
        self.model.eval() # to stop running_var move (maybe not critical)
        self.embedding_dir_path, self.sample_path, self.source_code_save_path = prep_dirs_cifar(self.logger.log_dir)
        self.embedding_list = []
    
    def on_test_start(self):
        self.embedding_dir_path, self.sample_path, self.source_code_save_path = prep_dirs_cifar(self.logger.log_dir)
        self.index = faiss.read_index(os.path.join(self.embedding_dir_path,'index.faiss'))
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0 ,self.index)
        self.init_results_list()

    def training_step(self, batch, batch_idx): # save locally aware patch features
        x, _ = batch
        features = self(x)
        embeddings = []
        for feature in features:
            m = torch.nn.AvgPool2d(3, 1, 1)
            embeddings.append(m(feature))
        embedding = embedding_concat(embeddings[0], embeddings[1])
        self.embedding_list.extend(reshape_embedding(np.array(embedding)))

    def training_epoch_end(self, outputs): 
        total_embeddings = np.array(self.embedding_list)
        # Random projection
        self.randomprojector = SparseRandomProjection(n_components='auto', eps=0.9) # 'auto' => Johnson-Lindenstrauss lemma
        self.randomprojector.fit(total_embeddings)
        # Coreset Subsampling
        selector = kCenterGreedy(total_embeddings,0,0)
        selected_idx = selector.select_batch(model=self.randomprojector, already_selected=[], N=int(total_embeddings.shape[0]*args.coreset_sampling_ratio))
        self.embedding_coreset = total_embeddings[selected_idx]
        
        print('initial embedding size : ', total_embeddings.shape)
        print('final embedding size : ', self.embedding_coreset.shape)
        #faiss
        self.index = faiss.IndexFlatL2(self.embedding_coreset.shape[1])
        self.index.add(self.embedding_coreset) 
        faiss.write_index(self.index,  os.path.join(self.embedding_dir_path,'index.faiss'))


    def test_step(self, batch, batch_idx): # Nearest Neighbour Search
        x, label = batch
        # extract embedding
        features = self(x)
        embeddings = []
        for feature in features:
            m = torch.nn.AvgPool2d(3, 1, 1)
            embeddings.append(m(feature))
        embedding_ = embedding_concat(embeddings[0], embeddings[1])
        embedding_test = np.array(reshape_embedding(np.array(embedding_)))
        score_patches, _ = self.index.search(embedding_test , k=args.n_neighbors) # [16, n_neighbors]
        anomaly_map = score_patches
        N_b = score_patches[np.argmax(score_patches[:,0])]
        w = (1 - (np.max(np.exp(N_b))/np.sum(np.exp(N_b))))
        score = w*max(score_patches[:,0]) # Image-level score
        self.gt_list_img_lvl.append(label.cpu().numpy()[0])
        self.pred_list_img_lvl.append(score)
        # self.save_anomaly_map(anomaly_map_resized_blur, input_x, gt_np*255, file_name[0], x_type[0])

    def test_epoch_end(self, outputs):
        print("Total image-level auc-roc score :")
        img_auc = roc_auc_score(self.gt_list_img_lvl, self.pred_list_img_lvl)
        print(img_auc)
        print('test_epoch_end')
        values = {'img_auc': img_auc}
        self.values = values
        self.log_dict(values)

def get_args():
    parser = argparse.ArgumentParser(description='ANOMALYDETECTION')
    parser.add_argument('--phase', choices=['train','test'], default='train')
    parser.add_argument('--dataset_mvtec_path', default='../data/mvtec') # 'D:\Dataset\mvtec_anomaly_detection')#
    parser.add_argument('--dataset_cifar10_path', default='../data') # 'D:\Dataset\mvtec_anomaly_detection')#
    parser.add_argument('--num_epochs', default=1)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--load_size', default=256, type=int)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--coreset_sampling_ratio', default=0.25, type=float)
    parser.add_argument('--project_root_path', default=r'./test') # 'D:\Project_Train_Results\mvtec_anomaly_detection\210624\test') #
    parser.add_argument('--save_src_code', default=True)
    parser.add_argument('--save_anomaly_map', default=True)
    parser.add_argument('--n_neighbors', type=int, default=9)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--dataset', default='cifar10', type=str, help='mvtec/cifar10 (Default=cifar10)')
    parser.add_argument('--class_name', '-cl', default='capsule', type=str, help='class name for mvtec')
    parser.add_argument('--data_path', default='/home/juyeon/data/mvtec', type=str)
    parser.add_argument('--epochs', default=20, type=int, metavar='epochs', help='number of epochs')
    parser.add_argument('--label', default=0, type=int, help='The normal class')
    parser.add_argument('--lr', type=float, default=1e-5, help='The initial learning rate.')
    parser.add_argument('--backbone', default=152, type=int, help='ResNet 18/152')
    parser.add_argument('--angular', action='store_true', help='Train with angular center loss')
    parser.add_argument('--save_path', default='./models/', type=str, help='where to save the weights')
    parser.add_argument('--load_path', default='./models/', type=str, help='where to get the weights')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args()
    
    if args.dataset == 'mvtec':
        filepath = args.load_path + str(args.backbone)+'_'+str(args.dataset)+'_'+str(args.class_name)

        train_loader, test_loader, train_loader_1 = utils.get_loaders(dataset=args.dataset, label_class=args.label, batch_size=args.batch_size, backbone=args.backbone, args=args)

        if args.phase == 'train':
            args.percent = 0.25
            args.coreset_sampling_ratio = 0.25
            trainer025 = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(args.project_root_path, args.class_name), max_epochs=args.num_epochs, gpus=args.gpu) #, check_val_every_n_epoch=args.val_freq,  num_sanity_val_steps=0) # ,fast_dev_run=True)    
            model025 = STPM_mvtec(hparams=args)
            trainer025.fit(model025)
            trainer025.test(model025)

            args.percent = 0.1
            args.coreset_sampling_ratio = 0.1
            trainer01 = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(args.project_root_path, args.class_name), max_epochs=args.num_epochs, gpus=args.gpu) #, check_val_every_n_epoch=args.val_freq,  num_sanity_val_steps=0) # ,fast_dev_run=True)
            model01 = STPM_mvtec(hparams=args)
            trainer01.fit(model01)
            trainer01.test(model01)

            args.percent = 0.01
            args.coreset_sampling_ratio = 0.01
            trainer001 = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(args.project_root_path, args.class_name), max_epochs=args.num_epochs, gpus=args.gpu) #, check_val_every_n_epoch=args.val_freq,  num_sanity_val_steps=0) # ,fast_dev_run=True)
            model001 = STPM_mvtec(hparams=args)
            trainer001.fit(model001)
            trainer001.test(model001)

            score025 = model025.values
            score01 = model01.values
            score001 = model001.values

            model_msc = utils.Model(args.backbone)
            model_msc = model_msc.to(device)    
            args.filepath = filepath
            os.makedirs(args.save_path, exist_ok=True)
            msc_score = train_model(model_msc, train_loader, test_loader, train_loader_1, device, args)

            pix_score = max(score025['pixel_auc'],score01['pixel_auc'],score001['pixel_auc'])
            img_score = max(score025['img_auc'],score01['img_auc'],score001['img_auc'],msc_score)
            print('MVTec Detection score {}'.format(img_score))
            print('MVTec Localization score {}'.format(pix_score))
        elif args.phase == 'test':
            args.percent = 0.25
            args.coreset_sampling_ratio = 0.25
            trainer025 = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(args.project_root_path, args.class_name), max_epochs=args.num_epochs, gpus=args.gpu) #, check_val_every_n_epoch=args.val_freq,  num_sanity_val_steps=0) # ,fast_dev_run=True)    
            model025 = STPM_mvtec(hparams=args)
            trainer025.test(model025)

            args.percent = 0.1
            args.coreset_sampling_ratio = 0.1
            trainer01 = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(args.project_root_path, args.class_name), max_epochs=args.num_epochs, gpus=args.gpu) #, check_val_every_n_epoch=args.val_freq,  num_sanity_val_steps=0) # ,fast_dev_run=True)
            model01 = STPM_mvtec(hparams=args)
            trainer01.test(model01)

            args.percent = 0.01
            args.coreset_sampling_ratio = 0.01
            trainer001 = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(args.project_root_path, args.class_name), max_epochs=args.num_epochs, gpus=args.gpu) #, check_val_every_n_epoch=args.val_freq,  num_sanity_val_steps=0) # ,fast_dev_run=True)
            model001 = STPM_mvtec(hparams=args)
            trainer001.test(model001)

            score025 = model025.values
            score01 = model01.values
            score001 = model001.values
            
            model_msc = load_weights(filepath)
            model_msc = model_msc.to(device)        
            msc_score = test_model(model_msc, train_loader, test_loader, device, args)
            pix_score = max(score025['pixel_auc'],score01['pixel_auc'],score001['pixel_auc'])
            img_score = max(score025['img_auc'],score01['img_auc'],score001['img_auc'],msc_score)
            print('MVTec Detection score {}'.format(img_score))
            print('MVTec Localization score {}'.format(pix_score))
    elif args.dataset == 'cifar10':
        model_pc = STPM_cifar10(hparams=args)
        filepath = args.load_path + str(args.backbone)+'_'+str(args.dataset)+'_'+str(args.label)
        train_loader, test_loader, train_loader_1 = utils.get_loaders(dataset=args.dataset, label_class=args.label, batch_size=args.batch_size, backbone=args.backbone, args=args)
        trainer = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(args.project_root_path, str(args.label)), max_epochs=args.num_epochs, gpus=args.gpu) #, check_val_every_n_epoch=args.val_freq,  num_sanity_val_steps=0) # ,fast_dev_run=True)
        if args.phase == 'train':
            trainer.fit(model_pc)
            trainer.test(model_pc)
            score_pc = model_pc.values
            model_msc = utils.Model(args.backbone)
            model_msc = model_msc.to(device)    
            args.filepath = filepath
            os.makedirs(args.save_path, exist_ok=True)
            msc_score = train_model(model_msc, train_loader, test_loader, train_loader_1, device, args)

            img_score = max(score_pc['img_auc'], msc_score)
            print('CIFAR10 Detection score {}'.format(img_score))
        elif args.phase == 'test':
            trainer.test(model_pc)
            score_pc = model_pc.values
            model_msc = load_weights(filepath)
            model_msc = model_msc.to(device)        
            msc_score = test_model(model_msc, train_loader, test_loader, device, args)

            img_score = max(score_pc['img_auc'], msc_score)
            print('CIFAR10 Detection score {}'.format(img_score))

    
