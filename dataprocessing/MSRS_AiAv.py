import os
import sys
sys.path.append(os.getcwd())
import h5py
import numpy as np
from tqdm import tqdm
from utils import image_read_cv2


def is_low_contrast(image, fraction_threshold=0.1, lower_percentile=10,
                    upper_percentile=90):
    """Determine if an image is low contrast."""
    limits = np.percentile(image, [lower_percentile, upper_percentile])
    ratio = (limits[1] - limits[0]) / limits[1]
    return ratio < fraction_threshold


path1_list=[ 
r"G:\reproduce_IVIF_result\MSRS\DIDFuse\Gray",
r"G:\reproduce_IVIF_result\MSRS\AUIF\Gray",
] # The fusion result path of sota

for data_name in ["MSRS_A_i","MSRS_A_v"]:
    if data_name=="MSRS_A_i":
        path2=r"dataprocessing\MSRS_train\ir"
    elif data_name=="MSRS_A_v":
        path2=r"dataprocessing\MSRS_train\vi"


    file_name_list=os.listdir(path2) 

    patchsize=128   
    stride=200     
    h5_path=os.path.join('.\\Data',data_name+'_'+str(patchsize)+'_'+str(stride)+'.h5')
    h5f = h5py.File(h5_path,'w')
    h5_ir = h5f.create_group('input_patchs')
    h5_vis = h5f.create_group('target_patchs')

    patch_num=0
    p_H_num=(480-patchsize)//stride + 1
    p_W_num=(640-patchsize)//stride + 1

    for path1 in path1_list:
        print(path1)
        for k in tqdm(range(len(file_name_list))):
            IR=image_read_cv2(os.path.join(path1,file_name_list[k]),mode='GRAY')[None,...]/255.0
            VI=image_read_cv2(os.path.join(path2,file_name_list[k]),mode='GRAY')[None,...]/255.0

            for kk in range(p_H_num*p_W_num):
                    
                    a0=kk//p_W_num 
                    a1=kk-a0*p_W_num 

                    IR_patch=IR[:,a0*stride:a0*stride+patchsize,a1*stride:a1*stride+patchsize].astype(np.float32)
                    VI_patch=VI[:,a0*stride:a0*stride+patchsize,a1*stride:a1*stride+patchsize].astype(np.float32)
                    
                    if not (is_low_contrast(IR_patch) or is_low_contrast(VI_patch)):
                        h5_ir.create_dataset(str(patch_num),     data=IR_patch, 
                                    dtype=IR_patch.dtype,   shape=IR_patch.shape)
                        h5_vis.create_dataset(str(patch_num),    data=VI_patch, 
                                    dtype=VI_patch.dtype,  shape=VI_patch.shape)
                        patch_num+=1
    h5f.close()     
    print('training set, # samples %d\n' % (patch_num))       
    with h5py.File(h5_path,"r") as f:
        for key in f.keys():
            print(f[key], key, f[key].name) 