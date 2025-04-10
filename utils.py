import matplotlib.pyplot as plt
import numpy as np
import hydra
import os, logging
import torch
import sys
from einops import rearrange

def imshow(img,save_dir=None):
    img = img     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if save_dir:
      plt.savefig(save_dir)
    #plt.show()
    plt.clf()

     
def list_round(data,th = 4):
    # type(preds) = type(labels) = torch.tensor
    rounded_data = []
    for i in range(len(data)):
        rounded_data.append(round(data[i],th))

    return rounded_data
    
    
def get_std(classwise_data):
    mean = 0
    for i in range(len(classwise_data)):
        mean +=classwise_data[i]/len(classwise_data)
        
    var = 0
    for i in range(len(classwise_data)):
        var +=(classwise_data[i]-mean)**2/len(classwise_data)
     
    std = np.sqrt(var)
    return std    
    
    

    
# B X C X H X W -> B X d^2 X C X H/d X W/d (patchwise division), B should be larger than 1.
def patch_division(image_tensor,d):
    d = int(d)
    batch_patch_image_tensor = rearrange(image_tensor,'b c (h dh) (w dw) -> b (h w) c dh dw', h=d,w=d)    
    return batch_patch_image_tensor


# B X d^2 X C X H/d X W/d -> B X C X H X W (reverse patch division), B should be larger than 1.      
def reverse_patch_division(batch_patch_image_tensor):
    B, P, C, h, w = batch_patch_image_tensor.size()
    d = int(np.sqrt(P))    
    image_tensor = rearrange(image_tensor,'b (h w) c dh dw -> b c (h dh) (w dw)', h=d,w=d)     
    return image_tensor   
 
        
def patch_wise_calculation(batch_patch_image_tensor_hat,batch_patch_image_tensor,image_wise_criterion): 
    #batch_patch_image_tensor: B X d^2 X C X H/d X W/d
    #image_wise_criterion: return criterion results for image_wise
    input_dim = batch_patch_image_tensor.size()
    patch_image_tensors_hat = batch_patch_image_tensor_hat.reshape(-1,input_dim[2],input_dim[3],input_dim[4])
    patch_image_tensors = batch_patch_image_tensor.reshape(-1,input_dim[2],input_dim[3],input_dim[4])

    patch_wise_calculation_result = image_wise_criterion(patch_image_tensors_hat,patch_image_tensors)
    patch_wise_calculation_result = patch_wise_calculation_result.reshape(input_dim[0],input_dim[1])
    #print("patch_wise_calculation_result:",patch_wise_calculation_result)
    return patch_wise_calculation_result # It is criterion result for each patch. Dimension is B X d^2.
    
    
    
     
#BLC -> BCHW -> BPCHW -> BPLC   
    
def BLC_to_BCHW(z_BLC,H,W):
    z_BLC_dim = z_BLC.size()
    B, L, C = z_BLC_dim
    assert L == H * W, "input feature has wrong size"   
    z_BCHW = z_BLC.reshape(B, H, W, C).permute(0, 3, 1, 2)
    return z_BCHW
    
    
def BCHW_to_BLC(z_BCHW):
    z_BCHW_dim = z_BCHW.size()
    B, C,H,W = z_BCHW_dim
    z_BLC = z_BCHW.flatten(2).permute(0, 2, 1)
    return z_BLC
    

def BPLC_to_BPCHW(z_BPLC,H,W):    
    z_BPLC_dim = z_BPLC.size()
    B, P, L, C = z_BPLC_dim
    assert L == H * W, "input feature has wrong size"   
    z_BPCHW = z_BPLC.reshape(B, P, H, W, C).permute(0, 1, 4, 2, 3)
    return z_BPCHW
    
    
def BPCHW_to_BPLC(z_BPCHW):    
    z_BPCHW_dim = z_BPCHW.size()
    B, P, C, H,W = z_BPCHW_dim  
    z_BPLC = z_BPCHW.flatten(3).permute(0,1, 3, 2)
    return z_BPLC    

    
    
    
    
    
    
