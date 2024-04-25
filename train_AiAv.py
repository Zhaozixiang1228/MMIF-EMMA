# -*- coding: utf-8 -*-
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  
import time
import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')  
import logging
logging.basicConfig(level=logging.CRITICAL)

from utils import H5Dataset_AiAv
from nets.Unet5 import UNet5

num_epochs = 40
lr = 1e-4
step_size=10
gamma=0.5
weight_decay = 0
batch_size =64


stat="Ai"  # A_i or A_v
if stat=="Ai":
    datapath=r"Data/MSRS_A_i_128_200.h5"    
if stat=="Av":
    datapath=r"Data//MSRS_A_v_128_200.h5"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = UNet5().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
trainloader = DataLoader(H5Dataset_AiAv(datapath),batch_size=batch_size,shuffle=True,num_workers=0)
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
torch.backends.cudnn.benchmark = True
prev_time = time.time()

for epoch in range(num_epochs):
    model.train()

    for i, (input, target, index)in enumerate(trainloader):  
        data_input, data_target = input.cuda(), target.cuda()
        optimizer.zero_grad()
        data_out=model(data_input)
        loss=F.mse_loss(data_out,data_target)
        loss.backward()
        optimizer.step()  

        batches_done = epoch * len(trainloader) + i
        batches_left = num_epochs * len(trainloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        print(
            "[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %.10s"
            % (
                epoch+1,
                num_epochs,
                i,
                len(trainloader),
                loss.item(),
                time_left,
            ))

    scheduler.step()  
torch.save(model.state_dict(), os.path.join("model",stat+"_"+timestamp+'.pth'))


