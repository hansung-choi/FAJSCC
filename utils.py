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
    
    
    
def get_channelwise_valid_z_mask(z,rate_info): #z: latent_feature_of_channel, rate_info:[start_feature_channel, end_feature_channel]  of valid feature   
    z_dim = z.size()
    mask = torch.zeros_like(z)  #.bool()
    if len(z_dim) ==3: #when z is B X L X C tensor
        mask[:,:,rate_info[0]:rate_info[1]] = 1.
        mask = mask.bool()
        valid_z = torch.masked_select(z, mask).reshape(z_dim[0],z_dim[1],-1)
    elif len(z_dim) ==4: #when z is B X C X H X W tensor
        mask[:,rate_info[0]:rate_info[1],:,:] = 1.
        mask = mask.bool()
        valid_z = torch.masked_select(z, mask).reshape(z_dim[0],-1,z_dim[2],z_dim[3])
    else:
        raise ValueError(f'shape for{z_dim} (len(z_dim)) is not implemented yet')
    
    return valid_z, mask
    
def get_SR_steps_channelwise_valid_zs_masks(z,step_num,channel_size):
    rate_info_list = [[step*channel_size,(step+1)*channel_size] for step in range(step_num)]
    valid_z_list = []
    mask_list = []
    for rate_info in rate_info_list:
        valid_z, mask = get_channelwise_valid_z_mask(z,rate_info)
        valid_z_list.append(valid_z)
        mask_list.append(mask)
    return valid_z_list, mask_list

def padding_received_z_hat_list(z,z_hat_list,mask_list):
    received_z_hat_list = []
    for i in range(len(mask_list)):
        received_z_hat = torch.zeros_like(z)
        for j in range(i+1):
            received_z_hat[mask_list[j]] = z_hat_list[j].reshape(-1)
        received_z_hat_list.append(received_z_hat)
    return received_z_hat_list

def padding_received_z_hat(z,z_hat,mask):
    received_z_hat = torch.zeros_like(z)
    received_z_hat[mask] = z_hat.reshape(-1)
    return received_z_hat

    
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
    
    
    
    
def oracle_patch_rate_allocation(patch_wise_psnr, channel_rate_list):
    #patch_wise_psnr: B X d^2, channel_rate_list: list of possible patchwise channel rate
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    patch_wise_psnr = torch.tensor(patch_wise_psnr).clone().detach().to(device)
    num_patch = patch_wise_psnr.size()[1]
    sorted_patch_wise_psnr, sort_index = patch_wise_psnr.sort(dim=-1,descending=True)
    
    # the number of patch for each rate allocation
    rate_wise_patch_num_list = [num_patch//len(channel_rate_list) for i in range(len(channel_rate_list))]
    rate_wise_patch_num_list[len(channel_rate_list)//2] = int(num_patch - num_patch//len(channel_rate_list)*(len(channel_rate_list)-1))
    
    # the cumulative number of patches for rate allocation
    rate_allocation_standard_list = [sum(rate_wise_patch_num_list[:i+1]) for i in range(len(rate_wise_patch_num_list))]
    
    patch_wise_rate = torch.zeros_like(patch_wise_psnr).long()

    
    for rate_num in range(len(channel_rate_list)):
        if rate_num ==0:
            interest_sort_index = sort_index[:,:rate_allocation_standard_list[rate_num]]
        else:
            interest_sort_index = sort_index[:,rate_allocation_standard_list[rate_num-1]:rate_allocation_standard_list[rate_num]]
        for i in range(len(interest_sort_index)):
            patch_wise_rate[i,interest_sort_index[i]] = channel_rate_list[rate_num]
    
    # patch_wise_rate: B X d^2, each element means appropriate rate allocation for each patch.
    #print("patch_wise_rate.float().mean(dim=-1):",patch_wise_rate.float().mean(dim=-1))
    
    return patch_wise_rate
    
def oracle_patch_power_allocation(patch_wise_psnr, channel_rate_list):
    #patch_wise_psnr: B X d^2, channel_rate_list: list of possible patchwise channel rate
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    patch_wise_psnr = torch.tensor(patch_wise_psnr).clone().detach().to(device)
    num_patch = patch_wise_psnr.size()[1]
    sorted_patch_wise_psnr, sort_index = patch_wise_psnr.sort(dim=-1,descending=True)
    
    # the number of patch for each rate allocation
    rate_wise_patch_num_list = [num_patch//len(channel_rate_list) for i in range(len(channel_rate_list))]
    rate_wise_patch_num_list[len(channel_rate_list)//2] = int(num_patch - num_patch//len(channel_rate_list)*(len(channel_rate_list)-1))
    
    # the cumulative number of patches for rate allocation
    rate_allocation_standard_list = [sum(rate_wise_patch_num_list[:i+1]) for i in range(len(rate_wise_patch_num_list))]
    
    patch_wise_rate = torch.zeros_like(patch_wise_psnr).float()

    
    for rate_num in range(len(channel_rate_list)):
        if rate_num ==0:
            interest_sort_index = sort_index[:,:rate_allocation_standard_list[rate_num]]
        else:
            interest_sort_index = sort_index[:,rate_allocation_standard_list[rate_num-1]:rate_allocation_standard_list[rate_num]]
        for i in range(len(interest_sort_index)):
            patch_wise_rate[i,interest_sort_index[i]] = channel_rate_list[rate_num]
    
    # patch_wise_rate: B X d^2, each element means appropriate rate allocation for each patch.
    #print("patch_wise_rate.float().mean(dim=-1):",patch_wise_rate.float().mean(dim=-1))
    
    return patch_wise_rate


def random_patch_rate_allocation(B, P, channel_rate_list): 
    # B: Batch size, P: Patch number , channel_rate_list: list of possible patchwise channel rate
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    patch_wise_psnr = np.random.uniform(low=0.5, high=50.0, size=(B,P))
    patch_wise_psnr = torch.tensor(patch_wise_psnr).reshape(B, P).clone().detach().to(device)
    num_patch = patch_wise_psnr.size()[1]
    sorted_patch_wise_psnr, sort_index = patch_wise_psnr.sort(dim=-1,descending=True)
    
    # the number of patch for each rate allocation
    rate_wise_patch_num_list = [num_patch//len(channel_rate_list) for i in range(len(channel_rate_list))]
    rate_wise_patch_num_list[len(channel_rate_list)//2] = int(num_patch - num_patch//len(channel_rate_list)*(len(channel_rate_list)-1))
    
    # the cumulative number of patches for rate allocation
    rate_allocation_standard_list = [sum(rate_wise_patch_num_list[:i+1]) for i in range(len(rate_wise_patch_num_list))]
    
    patch_wise_rate = torch.zeros_like(patch_wise_psnr).long()

    
    for rate_num in range(len(channel_rate_list)):
        if rate_num ==0:
            interest_sort_index = sort_index[:,:rate_allocation_standard_list[rate_num]]
        else:
            interest_sort_index = sort_index[:,rate_allocation_standard_list[rate_num-1]:rate_allocation_standard_list[rate_num]]
        for i in range(len(interest_sort_index)):
            patch_wise_rate[i,interest_sort_index[i]] = channel_rate_list[rate_num]
    
    # patch_wise_rate: B X d^2, each element means appropriate rate allocation for each patch.
    #print("patch_wise_rate.float().mean(dim=-1):",patch_wise_rate.float().mean(dim=-1))
    
    return patch_wise_rate
    
def random_patch_power_allocation(B, P, channel_rate_list): 
    # B: Batch size, P: Patch number , channel_rate_list: list of possible patchwise channel rate
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    patch_wise_psnr = np.random.uniform(low=0.5, high=50.0, size=(B,P))
    patch_wise_psnr = torch.tensor(patch_wise_psnr).reshape(B, P).clone().detach().to(device)
    num_patch = patch_wise_psnr.size()[1]
    sorted_patch_wise_psnr, sort_index = patch_wise_psnr.sort(dim=-1,descending=True)
    
    # the number of patch for each rate allocation
    rate_wise_patch_num_list = [num_patch//len(channel_rate_list) for i in range(len(channel_rate_list))]
    rate_wise_patch_num_list[len(channel_rate_list)//2] = int(num_patch - num_patch//len(channel_rate_list)*(len(channel_rate_list)-1))
    
    # the cumulative number of patches for rate allocation
    rate_allocation_standard_list = [sum(rate_wise_patch_num_list[:i+1]) for i in range(len(rate_wise_patch_num_list))]
    
    patch_wise_rate = torch.zeros_like(patch_wise_psnr).float()

    
    for rate_num in range(len(channel_rate_list)):
        if rate_num ==0:
            interest_sort_index = sort_index[:,:rate_allocation_standard_list[rate_num]]
        else:
            interest_sort_index = sort_index[:,rate_allocation_standard_list[rate_num-1]:rate_allocation_standard_list[rate_num]]
        for i in range(len(interest_sort_index)):
            patch_wise_rate[i,interest_sort_index[i]] = channel_rate_list[rate_num]
    
    # patch_wise_rate: B X d^2, each element means appropriate rate allocation for each patch.
    #print("patch_wise_rate.float().mean(dim=-1):",patch_wise_rate.float().mean(dim=-1))
    
    return patch_wise_rate

    
def get_patchwise_valid_z_mask(batch_patch_image_tensor,patch_wise_rate):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # batch_patch_image_tensor: B X d^2 X C X H/d X W/d
    # patch_wise_rate: B X d^2, each element means appropriate rate allocation for each patch.
    z = batch_patch_image_tensor   
    B, P, C, h, w = z.size()
    #.bool()

    z_BPLC = z.flatten(3).permute(0,1,3,2) # B X P X L X C where P=d^2, L=H/d X W/d
    valid_mask = torch.zeros_like(z_BPLC)
    max_rate = C
    mask = torch.arange(0, max_rate).repeat(B, P, h*w, 1).to(device)
    extended_patch_wise_rate = patch_wise_rate.reshape(B, P, 1, 1).repeat(1,1,h*w, max_rate)

    
    
    valid_mask[mask<extended_patch_wise_rate] = 1.
    valid_mask[mask>=extended_patch_wise_rate] = 0.
    valid_mask = valid_mask.bool()
    valid_z_BPLC = torch.masked_select(z_BPLC, valid_mask).reshape(B,-1)

    return z_BPLC, valid_z_BPLC, valid_mask
    
def get_valid_z(z,mask): #z: latent_feature_of_channel, mask: info of valid feature
    B = z.size()[0]
    valid_z = torch.masked_select(z, mask).reshape(B,-1) # shape should be B X -1 due to power allocation of channel.
    
    return valid_z
    
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

    
    
    
    
    
    