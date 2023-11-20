#!/usr/bin/env python
from __future__ import print_function

import os
import sys
import time
import glob
import pickle
import random
import traceback
import resource
import os.path as osp

from collections import OrderedDict

import torch
import torch.optim as optim
import numpy as np

from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from args import get_parser
from loss import LabelSmoothingCrossEntropy, get_mmd_loss
from utils import get_vector_property
from utils import BalancedSampler as BS

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))


class Processor():
    """
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        self.global_step = 0
        # pdb.set_trace()
        self.load_model()

        if self.arg.phase == 'model_size':
            pass
        else:
            self.load_optimizer()
            self.load_data()

        for k, v in self.data_loader.items():
            self.print_log(f"{k} loader has the size: {len(v)}")
            self.print_log(f"{k} dataset has the size: {len(v.dataset)}")

        self.train_acc_hist = []   
        self.train_loss_hist = []
        self.eval_acc_hist = []     
        self.eval_loss_hist = []
        self.best_acc = 0
        self.best_acc_epoch = 0

        self.model = self.model.cuda()

        if self.arg.half:
            self.scaler = torch.cuda.amp.GradScaler()

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.dataset is not None and self.arg.datacase is not None:
            data_path = f'data/{self.arg.dataset}/{self.arg.datacase}.npz'
        elif self.arg.datapath is not None:
            data_path = self.arg.datapath
        else:
            raise FileNotFoundError(f'No data is specified')

        if self.arg.phase == 'train':
            dt = Feeder(data_path=data_path,
                split='train',
                window_size=64,
                p_interval=[0.5, 1],
                vel=self.arg.use_vel,
                random_rot=self.arg.random_rot,
                sort=True if self.arg.balanced_sampling else False,
                fs_size=self.arg.fs_size,
                fold=self.arg.infact_fold,
                cross_fold = self.arg.cross_fold,
            )
            if self.arg.balanced_sampling:
                sampler = BS(data_source=dt, args=self.arg)
                shuffle = False
            else:
                sampler = None
                shuffle = True
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=dt,
                sampler=sampler,
                batch_size=self.arg.batch_size,
                shuffle=shuffle,
                num_workers=self.arg.num_worker,
                drop_last=False,
                pin_memory=True,
                worker_init_fn=init_seed)
            
        if self.arg.phase!='train_val':
            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=Feeder(
                    data_path=data_path,
                    split='test',
                    window_size=64,
                    p_interval=[0.95],
                    vel=self.arg.use_vel,
                    fs_size=self.arg.fs_size,
                    fold=self.arg.infact_fold,
                    cross_fold = self.arg.cross_fold,
                ),
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker,
                drop_last=False,
                pin_memory=True,
                worker_init_fn=init_seed)
            
        else:
            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=Feeder(
                    data_path=data_path,
                    split='train_val',
                    window_size=64,
                    p_interval=[0.95],
                    vel=self.arg.use_vel,
                    fs_size=self.arg.fs_size,
                    fold=self.arg.infact_fold
                ),
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker,
                drop_last=False,
                pin_memory=True,
                worker_init_fn=init_seed)

    def load_model(self):
        if self.arg.load_simple:
            from model.infogcn_simple import InfoGCN
            self.print_log('Using small model')
        else:
            from model.infogcn import InfoGCN
            self.print_log('load base model')
        self.model = InfoGCN(
            encoding_channels=self.arg.num_enc_channels,
            num_class=self.arg.num_class,
            num_point=self.arg.num_point,
            num_person=self.arg.num_person,
            graph=self.arg.graph,
            in_channels=3,
            drop_out=0,
            num_head=self.arg.n_heads,
            k=self.arg.k,
            noise_ratio=self.arg.noise_ratio,
            gain=self.arg.z_prior_gain
        )
        self.loss = LabelSmoothingCrossEntropy().cuda()

        if self.arg.weights:
            self.print_log('Initialize loading weights')
            self.global_step = int(self.arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict([[k.split('module.')[-1], v.cuda()] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff_keys = []
                for (state_k, state_w), (load_k, load_w) in zip(state.items(), weights.items()):
                    if state_k!=load_k or state_w.shape!=load_w.shape:
                        diff_keys.append(state_k)
                        
                self.print_log('These keys cannot be loaded:')
                for dk in diff_keys:
                    self.print_log('  ' + dk)

                filtered_dict = {k: v for k, v in state.items() if k in diff_keys}
                state.update(filtered_dict)
                self.model.load_state_dict(state)

        if self.arg.freeze_encoder:
            unwanted_keys = {'decoder.weight', 'decoder.bias'}
            for name, param in self.model.named_parameters():
                if name not in unwanted_keys:
                    param.requires_grad = False

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        os.makedirs(self.arg.work_dir, exist_ok=True)
        os.makedirs(os.path.join(self.arg.work_dir, 'runs'), exist_ok=True)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch and self.arg.weights is None:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        self.adjust_learning_rate(epoch)


        loss_value = []
        mmd_loss_value = []
        l2_z_mean_value = []
        acc_value = []
        cos_z_value = []
        dis_z_value = []
        cos_z_prior_value = []
        dis_z_prior_value = []

        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)

        for data, y, index in tqdm(self.data_loader['train'], dynamic_ncols=True):
            self.global_step += 1
            with torch.no_grad():
                data = data.float().cuda()
                y = y.long().cuda()
            timer['dataloader'] += self.split_time()

            with torch.cuda.amp.autocast(enabled=self.arg.half):
                # forward
                y_hat, z, z_mu, z_logvar = self.model(data)
                mmd_loss, l2_z_mean, z_mean = get_mmd_loss(z, self.model.z_prior, y, self.arg.num_class)
                cos_z, dis_z = get_vector_property(z_mean)
                cos_z_prior, dis_z_prior = get_vector_property(self.model.z_prior)
                cos_z_value.append(cos_z.data.item())
                dis_z_value.append(dis_z.data.item())
                cos_z_prior_value.append(cos_z_prior.data.item())
                dis_z_prior_value.append(dis_z_prior.data.item())

                cls_loss = self.loss(y_hat, y)
                loss = self.arg.lambda_2* mmd_loss + self.arg.lambda_1* l2_z_mean + cls_loss
                # backward
                self.optimizer.zero_grad()

            if self.arg.half:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            loss_value.append(cls_loss.data.item())
            mmd_loss_value.append(mmd_loss.data.item())
            l2_z_mean_value.append(l2_z_mean.data.item())
            timer['model'] += self.split_time()

            value, predict_label = torch.max(y_hat.data, 1)
            acc = torch.mean((predict_label == y.data).float())
            acc_value.append(acc.data.item())

            timer['statistics'] += self.split_time()

        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(f'\tTraining loss: {np.mean(loss_value):.4f}.  Training acc: {np.mean(acc_value)*100:.2f}%.')
        self.print_log(f'\tTime consumption: [Data]{proportion["dataloader"]}, [Network]{proportion["model"]}')

        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])

            torch.save(weights, f'{self.arg.work_dir}/runs/runs-{epoch+1}-{int(self.global_step)}.pt')

        train_epoch_acc = np.mean(acc_value)
        self.train_acc_hist.append(train_epoch_acc)
        self.train_loss_hist.append(loss_value)

    def eval(self, epoch, save_score=False, loader_name=['test'], save_z=False, save_best=True):
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            cls_loss_value = []
            mmd_loss_value = []
            l2_z_mean_value = []
            score_frag = []
            label_list = []
            pred_list = []
            cos_z_value = []
            dis_z_value = []
            cos_z_prior_value = []
            dis_z_prior_value = []
            step = 0
            z_list = []
            z_mu_list = []
            z_logvar_list = []
            for data, y, index in tqdm(self.data_loader[ln], dynamic_ncols=True):
                label_list.append(y)
                with torch.no_grad():
                    data = data.float().cuda()
                    y = y.long().cuda()
                    y_hat, z, z_mu, z_logvar = self.model(data)
                    if save_z:
                        z_mu_list.append(z_mu.data.cpu().numpy())
                        z_logvar_list.append(z_logvar.data.cpu().numpy())
                        z_list.append(z.data.cpu().numpy())

                    with torch.cuda.amp.autocast(enabled=self.arg.half):
                        mmd_loss, l2_z_mean, z_mean = get_mmd_loss(z, self.model.z_prior, y, self.arg.num_class)
                        cos_z, dis_z = get_vector_property(z_mean)
                        cos_z_prior, dis_z_prior = get_vector_property(self.model.z_prior)
                        cos_z_value.append(cos_z.data.item())
                        dis_z_value.append(dis_z.data.item())
                        cos_z_prior_value.append(cos_z_prior.data.item())
                        dis_z_prior_value.append(dis_z_prior.data.item())
                        cls_loss = self.loss(y_hat, y)
                        loss = self.arg.lambda_2*mmd_loss + self.arg.lambda_1*l2_z_mean + cls_loss
                        score_frag.append(y_hat.data.cpu().numpy())
                        loss_value.append(loss.data.item())
                        cls_loss_value.append(cls_loss.data.item())
                        mmd_loss_value.append(mmd_loss.data.item())
                        l2_z_mean_value.append(l2_z_mean.data.item())

                    _, predict_label = torch.max(y_hat.data, 1)
                    pred_list.append(predict_label.data.cpu().numpy())
                    step += 1

            
            score = np.concatenate(score_frag)
            loss = np.mean(loss_value)
            cls_loss = np.mean(cls_loss_value)
            mmd_loss = np.mean(mmd_loss_value)
            l2_z_mean_loss = np.mean(l2_z_mean_value)
            if any(substring in self.arg.feeder for substring in ['coco', 'h36m']):
                self.data_loader[ln].dataset.sample_name = np.arange(len(score))

            score_dict = dict(
                zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log('\tMean {} loss of {} batches: {:4f}.'.format(
                ln, self.arg.n_desired//self.arg.batch_size, np.mean(cls_loss_value)))
            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(
                    k, 100 * self.data_loader[ln].dataset.top_k(score, k)))

            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(
                        self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            if save_best:
                if accuracy > self.best_acc:
                    self.best_acc = accuracy
                    self.best_acc_epoch = epoch + 1
                    state_dict = self.model.state_dict()
                    weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
                    for pth in glob.glob(f'{self.arg.work_dir}/bestmodel-*.pt'):
                        os.remove(path=pth)
                    torch.save(weights, f'{self.arg.work_dir}/bestmodel-{epoch+1}-{int(self.global_step)}.pt')
                    with open(f'{self.arg.work_dir}/best_score.pkl', 'wb') as f:
                        pickle.dump(score_dict, f)

            self.print_log(f'Accuracy: {accuracy}, model: {self.arg.model_saved_name}')
            # acc for each class:
            label_list = np.concatenate(label_list)
            pred_list = np.concatenate(pred_list)

            if save_z:
                z_mu_list = np.concatenate(z_mu_list)
                z_logvar_list = np.concatenate(z_logvar_list)
                z_list = np.concatenate(z_list)
                np.savez(f'{self.arg.work_dir}/z_values_{self.arg.phase}.npz', 
                         z=z_list, z_prior=self.model.z_prior.cpu().numpy(), z_mu=z_mu_list, z_logvar=z_logvar_list,
                         labels=label_list, preds=pred_list)

            self.eval_acc_hist.append(accuracy)
            #self.eval_loss_hist.append()

    def start(self):
        if self.arg.phase == 'train':
            #self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = 0
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.print_log(f'# Parameters: {count_parameters(self.model)}')
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = (epoch + 1 == self.arg.num_epoch) or (epoch + 1 > self.arg.save_epoch)

                self.train(epoch, save_model=save_model)

                # if epoch > 80:
                self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'], save_z=False)

            acc_hist = {'train': self.train_acc_hist, 'test': self.eval_acc_hist, 'train_loss': self.train_loss_hist}
            with open(os.path.join(self.arg.work_dir, 'acc_hist.pkl'), 'wb') as f:
                pickle.dump(acc_hist, f)

            # test the best model
            weights_path = glob.glob(os.path.join(self.arg.work_dir, 'bestmodel'+'*'))[0]
            weights = torch.load(weights_path)
            self.model.load_state_dict(weights)

            self.arg.print_log = False
            self.eval(epoch=0, save_score=True, loader_name=['test'])
            self.arg.print_log = True


            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print_log(f'Best accuracy: {self.best_acc}')
            self.print_log(f'Epoch number: {self.best_acc_epoch}')
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
            self.print_log(f'seed: {self.arg.seed}')

        elif self.arg.phase == 'test' or self.arg.phase == 'train_val':
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            #self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], save_z=True, save_best=False)
            self.print_log('Done.\n')

def main():
    # parser arguments
    parser = get_parser()
    arg = parser.parse_args()
    path2name = lambda x: '_'.join([c for c in os.path.splitext(x)[0].split(os.sep) if '.' not in c])
    if arg.dataset is not None and arg.datacase is not None:
        arg.work_dir = f"{arg.output_dir}/infogcn_{arg.dataset}_{arg.datacase}"
    elif arg.datapath is not None:
        arg.work_dir = f"{arg.output_dir}/infogcn_{path2name(arg.datapath)}"
    if arg.weights!=None:
        arg.work_dir = arg.work_dir+f"_FT"
    print(arg.work_dir)
    init_seed(arg.seed)
    # execute process
    processor = Processor(arg)
    processor.start()

if __name__ == '__main__':
    main()
