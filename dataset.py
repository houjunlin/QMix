from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torchnet.meter import AUCMeter
import cv2
import pandas as pd

       
class cifar_dataset(Dataset): 
    def __init__(self, dataset, r, noise_mode, bad_r, transform, mode, noise_file, pred=[], probability=[],log=''): 
        
        self.dataset = dataset
        self.r = r 
        self.bad_r = bad_r
        self.transform = transform
        self.mode = mode 
        self.imgs = []
        self.labels = []
        self.quality_labels = []
        self.test_data=[]
        self.test_label=[]
        
        self.bad_img_train=[]
        self.bad_label_train=[]
        self.bad_quality_train=[]
       
        self.path = '/raid/hjl/DivideMix-DDR/DDR_preprocess1024/'
        self.train_path = 'preprocess1024_train/'
        self.val_path = 'preprocess1024_valid/'

        if self.mode=='test': 
            with open(self.path+'valid.txt',encoding='utf-8') as file:
                for line in file.readlines():
                    line = line.strip('\n')
                    img = line.split(' ')[0]
                    label = int(line.split(' ')[1])
                    if label != 5:
                        self.test_data.append(self.path+self.val_path+img)
                        self.test_label.append(label)

        else:
            with open(self.path+'train.txt',encoding='utf-8') as file:
                for line in file.readlines():
                    line = line.strip('\n')
                    img = line.split(' ')[0]
                    label = int(line.split(' ')[1])
                    quality = 1
                    if label == 5:
                        label = random.randint(0,4) # same
                        quality = 0
                        self.bad_img_train.append(self.path+self.train_path+img)
                        self.bad_label_train.append(label)
                        self.bad_quality_train.append(quality)
                    else: # no bad quality
                        self.imgs.append(self.path+self.train_path+img) 
                        self.labels.append(label)
                        self.quality_labels.append(quality)

            if os.path.exists(noise_file):
                noise_label = json.load(open(noise_file,"r"))
                print(len(noise_label))
            else:    #inject noise   
                noise_label = []
                num_images = len(self.imgs)
                idx = list(range(num_images))
                random.shuffle(idx)
                num_noise = int(self.r*num_images)            
                noise_idx = idx[:num_noise]
                for i in range(num_images):
                    if i in noise_idx:
                        if noise_mode=='sym':
                            noiselabel = random.randint(0,4)
                            noise_label.append(noiselabel)
                        elif noise_mode=='asym':   
                            p = np.array([0.5,0.05, 0.35, 0.02, 0.08])   
                            noiselabel = np.random.choice([0, 1, 2, 3, 4], p = p.ravel()).item()                  
                    else:    
                        noise_label.append(self.labels[i])   
                print("save noisy labels to %s ..."%noise_file)        
                json.dump(noise_label,open(noise_file,"w"))       

            print('good img:', len(self.imgs))
            if self.bad_r != 0:
                self.bad_img_add = []
                self.bad_label_add = []
                self.bad_quality_add = []
                for sets in ['valid', 'test']:
                    with open(self.path+sets+'.txt',encoding='utf-8') as file:
                        for line in file.readlines():
                            line = line.strip('\n')
                            img = line.split(' ')[0]
                            label = int(line.split(' ')[1])
                            if label == 5:
                                label = random.randint(0,4) # same
                                self.bad_img_add.append(self.path+'preprocess1024_'+sets+'/'+img)
                                self.bad_label_add.append(label)
                                self.bad_quality_add.append(0)
                self.imgs = self.imgs + self.bad_img_train + self.bad_img_add

                if noise_file.split('/')[1] == 'noise_file_uni':
                    self.bad_label_all = json.load(open('bad_file/0.15_uni.json',"r"))
                    print("load uni bad...................")
                elif noise_file.split('/')[1] == 'noise_file_pro':
                    self.bad_label_all = json.load(open('bad_file/0.15_pro.json',"r"))
                    print("load pro bad...................")
                

                self.labels = self.labels + self.bad_label_all #list(map(lambda x: x[0]-x[1], zip(bad_label_all, [10]*len(bad_label_all))))
                noise_label = noise_label + self.bad_label_all
                print(self.bad_label_all[:10])
                self.quality_labels = self.quality_labels + self.bad_quality_train +self.bad_quality_add                   

            if self.bad_r == 0.45:
                pseudo_bad_index_file = './noise_file_uni/pseudo_bad_index_0.3_noise0.2.json'
                pseudo_bad_index = json.load(open(pseudo_bad_index_file,"r"))
                for index in pseudo_bad_index:
                    self.imgs[index] = '/raid/hjl/DivideMix-DDR/degrade/preprocess1024_train_degrade/de_image/' + self.imgs[index].split('/')[-1]
                    self.quality_labels[index] = 0
            elif self.bad_r == 0.75:
                pseudo_bad_index_file = './noise_file_uni/pseudo_bad_index_0.6_noise0.2.json'
                pseudo_bad_index = json.load(open(pseudo_bad_index_file,"r"))
                for index in pseudo_bad_index:
                    self.imgs[index] = '/raid/hjl/DivideMix-DDR/degrade/preprocess1024_train_degrade/de_image/' + self.imgs[index].split('/')[-1]
                    self.quality_labels[index] = 0        
            elif self.bad_r == 0.35: #pro
                pseudo_bad_index_file = './noise_file_pro/pseudo_bad_index_0.2_noise0.2pro.json'
                pseudo_bad_index = json.load(open(pseudo_bad_index_file,"r"))
                for index in pseudo_bad_index:
                    self.imgs[index] = '/raid/hjl/DivideMix-DDR/degrade/preprocess1024_train_degrade/de_image/' + self.imgs[index].split('/')[-1]
                    self.quality_labels[index] = 0   
            elif self.bad_r == 0.55: #pro
                pseudo_bad_index_file = './noise_file_pro/pseudo_bad_index_0.4_noise0.2pro.json'
                pseudo_bad_index = json.load(open(pseudo_bad_index_file,"r"))
                for index in pseudo_bad_index:
                    self.imgs[index] = '/raid/hjl/DivideMix-DDR/degrade/preprocess1024_train_degrade/de_image/' + self.imgs[index].split('/')[-1]
                    self.quality_labels[index] = 0                                   

            print('final img: ', len(self.imgs))


            if self.mode == 'all':
                self.train_data = self.imgs #n,32,32,3
                self.noise_label = noise_label #n
                self.good = 1-(np.array(self.quality_labels) == 0)   
                self.clean = (np.array(noise_label)==np.array(self.labels))       #  这里把bad也算是clean了                                                                            
            else:                   
                if self.mode == "labeled":
                    pred_idx = []
                    self.ifgood = []
                    for i in range(pred.shape[0]):
                        if pred[i] == 0: # clean
                            pred_idx.append(i)
                            self.ifgood.append(1)
                        elif pred[i] == 1: # bad
                            pred_idx.append(i)
                            self.ifgood.append(0)
                    pred_idx = np.array(pred_idx)
                    self.ifgood = np.array(self.ifgood)
                    self.probability = []
                    self.all_prob = [] # for adaptive label smooth eps
                    for i in range(pred_idx.shape[0]):
                        if self.ifgood[i] == 1: # clean
                            self.probability.append(probability[pred_idx[i],0])
                        else:
                            self.probability.append(probability[pred_idx[i],1])
                        self.all_prob.append(probability[pred_idx[i]])
                    

                    self.clean = (np.array(noise_label)==np.array(self.labels))                                           
                    self.clean[self.quality_labels==0]=False 
                    auc_meter_clean = AUCMeter()
                    auc_meter_clean.reset()
                    auc_meter_clean.add(probability[:,0],self.clean)        
                    auc_clean,_,_ = auc_meter_clean.value()  

                    self.good = 1-(np.array(self.quality_labels) == 0)                                                
                    auc_meter_good = AUCMeter()
                    auc_meter_good.reset()
                    auc_meter_good.add(probability[:,0]+probability[:,2],self.good)        
                    auc_good,_,_ = auc_meter_good.value()

                    log.write('Numer of labeled samples:%d   AUC:%.3f   bad samples:%d   AUC:%.3f\n'%(np.sum(pred==0),auc_clean,np.sum(pred==1),auc_good))
                    log.flush()  

                elif self.mode == "unlabeled":
                    pred_idx = []
                    for i in range(pred.shape[0]):
                        if pred[i] == 2: # noisy
                            pred_idx.append(i)
                    pred_idx = np.array(pred_idx)
                    
                self.train_data = [self.imgs[i] for i in pred_idx]   
                self.noise_label = [noise_label[i] for i in pred_idx]                          
                print("%s data has a size of %d"%(self.mode,len(self.noise_label)))    

                if self.mode == 'labeled':
                    from collections import Counter
                    print(Counter(self.noise_label))
                
    def __getitem__(self, index):
        if self.mode=='labeled':
            img, target, prob, ifgood = self.train_data[index], self.noise_label[index], self.probability[index], self.ifgood[index]
            all_prob = self.all_prob[index]
            img = Image.open(img).convert('RGB')
            img1 = self.transform[0](img) 
            img2 = self.transform[1](img) #weak
            img3 = self.transform[2](img) 
            img4 = self.transform[3](img) #strong
            return img1, img2, img3, img4, target, prob, ifgood , all_prob        
        elif self.mode=='unlabeled':
            img = self.train_data[index]
            img = Image.open(img).convert('RGB')
            img1 = self.transform[0](img) 
            img2 = self.transform[1](img) #weak
            img3 = self.transform[2](img) 
            img4 = self.transform[3](img) #strong
            return img1, img2, img3, img4
        elif self.mode=='all': 
            img, target, clean, good = self.train_data[index], self.noise_label[index], self.clean[index], self.good[index]
            img = Image.open(img).convert('RGB')
            img = self.transform(img)            
            return img, target, index, clean, good        
        elif self.mode=='test':
            if self.dataset == 'kaggle':
                img1, img2 = self.test_data[index] 
                target1, target2 = self.test_label[index]
                img1 = Image.open(img1).convert('RGB')
                img1 = self.transform(img1)     
                img2 = Image.open(img2).convert('RGB')
                img2 = self.transform(img2)  
                return img1, img2, target1, target2

            elif self.dataset == 'ddr':
                img, target = self.test_data[index], self.test_label[index]
                img = Image.open(img).convert('RGB')
                img = self.transform(img)
                return img, target


           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_data)
        else:
            return len(self.test_data)         
        
        
class cifar_dataloader():  
    def __init__(self, dataset, r, noise_mode, bad_r, batch_size, num_workers, log, noise_file=''):
        self.dataset = dataset
        self.r = r
        self.bad_r = bad_r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.log = log
        self.noise_file = noise_file

        data_aug = {
            'brightness': 0.4,  # how much to jitter brightness
            'contrast': 0.4,  # How much to jitter contrast
            'saturation': 0.4,
            'hue': 0.1,
            'scale': (0.8, 1.2),  # range of size of the origin size cropped
            'ratio': (0.8, 1.2),  # range of aspect ratio of the origin aspect ratio cropped
            'degrees': (-180, 180),  # range of degrees to select from
            'translate': (0.2, 0.2),  # tuple of maximum absolute fraction for horizontal and vertical translations
            'size': 384
        }

        self.transform_train_s = transforms.Compose([
            transforms.Resize((data_aug['size'],data_aug['size'])),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(
                brightness=data_aug['brightness'],
                contrast=data_aug['contrast'],
                saturation=data_aug['saturation'],
                # hue=data_aug['hue']
            ),
            transforms.RandomResizedCrop(
                size=(data_aug['size'], data_aug['size']),
                scale=data_aug['scale'],
                ratio=data_aug['ratio']
            ),
            transforms.RandomAffine(
                degrees=data_aug['degrees'],
                # translate=data_aug['translate']
            ),
            transforms.RandomGrayscale(0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.426, .298, .213],std=[.277, .203, .169])
        ])


        self.transform_train_w = transforms.Compose([
            transforms.Resize((data_aug['size'],data_aug['size'])),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.ColorJitter(
            #     brightness=data_aug['brightness'],
            #     contrast=data_aug['contrast'],
            #     saturation=data_aug['saturation'],
            #     # hue=data_aug['hue']
            # ),
            # transforms.RandomResizedCrop(
            #     size=(data_aug['size'], data_aug['size']),
            #     scale=data_aug['scale'],
            #     ratio=data_aug['ratio']
            # ),
            transforms.RandomAffine(
                degrees=data_aug['degrees'],
                # translate=data_aug['translate']
            ),
            # transforms.RandomGrayscale(0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.426, .298, .213],std=[.277, .203, .169])
        ])

        self.transform_train = {
                "warmup": self.transform_train_s,
                "unlabeled": [
                            self.transform_train_w,
                            self.transform_train_w,
                            self.transform_train_s,
                            self.transform_train_s
                        ],
                "labeled": [
                            self.transform_train_w,
                            self.transform_train_w,
                            self.transform_train_s,
                            self.transform_train_s
                        ],
            }

        self.transform_test = transforms.Compose([
            transforms.Resize((data_aug['size'],data_aug['size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.426, .298, .213],std=[.277, .203, .169])
        ])


    def run(self,mode,pred=[],prob=[]):
        if mode=='warmup':
            all_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, bad_r=self.bad_r,transform=self.transform_train['warmup'], mode="all", noise_file=self.noise_file)                  
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)             
            return trainloader
                                     
        elif mode=='train':
            labeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,bad_r=self.bad_r,transform=self.transform_train['labeled'], mode="labeled", pred=pred, probability=prob, log=self.log, noise_file=self.noise_file)                 
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)   
            
            unlabeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,bad_r=self.bad_r,transform=self.transform_train['unlabeled'], mode="unlabeled", pred=pred, noise_file=self.noise_file)                      
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)     
            return labeled_trainloader, unlabeled_trainloader
        
        elif mode=='test':
            test_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,bad_r=self.bad_r,transform=self.transform_test, mode='test', noise_file=self.noise_file)           
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size*4,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)          
            return test_loader
        
        elif mode=='eval_train':
            eval_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,bad_r=self.bad_r,transform=self.transform_test, mode='all', noise_file=self.noise_file)          
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size*4,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)          
            return eval_loader        
