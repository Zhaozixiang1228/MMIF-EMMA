# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.getcwd())
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  
import torch
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.CRITICAL)
import numpy as np

from utils import image_read_cv2,img_save
from nets.Ufuser import Ufuser


path_ir=r"test_img\ir"
path_vi=r"test_img\vi"
path_save=r"test_result"
path_model=r"model\EMMA.pth"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model=Ufuser().to(device)
model.load_state_dict(torch.load(path_model))
model.eval()

with torch.no_grad():
    for imgname in tqdm(os.listdir(path_ir)):
    
        IR = image_read_cv2(os.path.join(path_ir, imgname), 'GRAY')[np.newaxis,np.newaxis,...]/255
        VI = image_read_cv2(os.path.join(path_vi, imgname), 'GRAY')[np.newaxis,np.newaxis,...]/255

        h, w = IR.shape[2:]
        h1 = h - h % 32
        w1 = w - w % 32
        h2 = h % 32
        w2 = w % 32

        if h1==h and w1==w: # Image size can be divided by 32
            ir = ((torch.FloatTensor(IR))).to(device)
            vi = ((torch.FloatTensor(VI))).to(device)
            data_Fuse=model(ir,vi)
            data_Fuse=(data_Fuse-torch.min(data_Fuse))/(torch.max(data_Fuse)-torch.min(data_Fuse))
            fused_image = np.squeeze((data_Fuse * 255).cpu().numpy())
            img_save(fused_image, imgname.split(sep='.')[0], path_save)
        else:
            # Upper left 
            fused_temp=np.zeros((h,w),dtype=np.float32)
            ir_temp = ((torch.FloatTensor(IR))[:,:,:h1,:w1]).to(device)
            vi_temp = ((torch.FloatTensor(VI))[:,:,:h1,:w1]).to(device)
            data_Fuse=model(ir_temp,vi_temp)
            fused_image = np.squeeze((data_Fuse * 255).cpu().numpy())
            fused_temp[:h1,:w1]=fused_image

            # upper right
            if w1!=w:
                ir_temp = ((torch.FloatTensor(IR))[:,:,:h1,-w1:]).to(device)
                vi_temp = ((torch.FloatTensor(VI))[:,:,:h1,-w1:]).to(device)
                data_Fuse=model(ir_temp,vi_temp)
                fused_image = np.squeeze((data_Fuse * 255).cpu().numpy())
                fused_temp[:h1,-w2:]=fused_image[:,-w2:]

            # lower left
            if h1!=h:    
                ir_temp = ((torch.FloatTensor(IR))[:,:,-h1:,:w1]).to(device)
                vi_temp = ((torch.FloatTensor(VI))[:,:,-h1:,:w1]).to(device)
                data_Fuse=model(ir_temp,vi_temp)
                fused_image = np.squeeze((data_Fuse * 255).cpu().numpy())
                fused_temp[-h2:,:w1]=fused_image[-h2:,:]

            
            # lower right
            if h1!=h and w1!=w:
                ir_temp = ((torch.FloatTensor(IR))[:,:,-h1:,-w1:]).to(device)
                vi_temp = ((torch.FloatTensor(VI))[:,:,-h1:,-w1:]).to(device)
                data_Fuse=model(ir_temp,vi_temp)
                fused_image = np.squeeze((data_Fuse * 255).cpu().numpy())
                fused_temp[-h2:,-w2:]=fused_image[-h2:,-w2:]

            fused_temp=(fused_temp-np.min(fused_temp))/(np.max(fused_temp)-np.min(fused_temp))
            fused_temp=((fused_temp)*255)
            img_save(fused_temp, imgname.split(sep='.')[0], path_save) 
