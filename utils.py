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

    
    
def make_init_classwise_metric_info(num_classes):
    init_classwise_metric_info = []
    for i in range(num_classes):
        init_classwise_metric_info.append([0,0])
    #init_class_acc_info[i][0] means the cumulative number of i'th class-tested which is total.
    #init_class_acc_info[i][1] means the cumulative number of i'th class-tested which is correct classified or mse or other metrics.
    return init_classwise_metric_info

def refine_classwise_metric_info(classwise_metric_info,metric_type = "mse"):
    classwise_metric = []
    if metric_type == "accuracy":
        constant = 100
    else:
        constant = 1
    
    for i in range(len(classwise_metric_info)):
        if classwise_metric_info[i][0] == 0:
            classwise_metric.append(None)
        else:
            classwise_metric.append(classwise_metric_info[i][1]/classwise_metric_info[i][0]*constant)
    # classwise_metric[i] = metric for i'th class
    return classwise_metric
    
def refine_classwise_metric_info_list(classwise_metric_info_list):
    classwise_metric_list = []
    for i in range(len(class_acc_info_list)):
        acc_per_class = refine_classwise_metric_info(classwise_metric_info_list[i],metric_type = "mse")
        classwise_metric_list.append(acc_per_class)
    # classwise_metric_list[i] = classwise_metric for i'th epoch
    return classwise_metric_list

def update_class_acc_info(class_acc_info,preds,labels):
    # type(preds) = type(labels) = torch.tensor
    for i in range(len(preds)):
        class_acc_info[labels.data[i]][0] +=1
        if labels.data[i] == preds[i]:
            class_acc_info[labels.data[i]][1] +=1
    return class_acc_info

def update_class_mse_info(class_mse_info,image_wise_mse,labels):
    # type(preds) = type(labels) = torch.tensor
    for i in range(image_wise_mse.size()[0]):
        class_mse_info[labels.data[i]][0] +=1
        class_mse_info[labels.data[i]][1] +=image_wise_mse[i].item()
    return class_mse_info

    
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

    
    
    
    
    
    
