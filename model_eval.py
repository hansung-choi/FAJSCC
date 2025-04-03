from data_maker import *
from model.model_maker import *
import time
import math
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
from utils import *
from loss_maker import *
import csv
from fvcore.nn import FlopCountAnalysis, flop_count_table    
import gc

def cal_flops(cfg, logger, model): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    H, W = cfg.input_resolution
    H, W = 512,768  # resolution of DIV2K0801 image data
    input_image = torch.rand(1,3,H,W).float()
    input_image = input_image.to(device) 
    model.to(device)    
    model.eval()
    with torch.no_grad():
        flops = FlopCountAnalysis(model, input_image)

    GFlops = flops.total()/10**9
    logger.info(f'GFlops: {GFlops}')

    return GFlops


def to_MB(a):
    return a/1024.0/1024.0

def cal_MB(cfg, logger, model): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to('cpu')
    gc.collect()
    torch.cuda.empty_cache()
    previous_memory = to_MB(torch.cuda.memory_allocated())
    model.to(device)    
    model.eval()
    Mmemory = to_MB(torch.cuda.memory_allocated())
    Model_Memory = Mmemory-previous_memory
    logger.info(f'Allocated gpu memory for model usage: {Model_Memory} Mb')

    return Model_Memory
    
def get_n_model_params(cfg, logger, model): 
    model.to('cpu')
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    Mparams = params/10**6
    logger.info(f'The number of parameters of model: {Mparams} M')
    
    return Mparams   



def eval_model(cfg: DictConfig, logger, model, trainloader,testloader, criterion):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    since = time.time()
    evaluater = ModelEvaluater(cfg)

    evaluation_dictionary = evaluater.one_epoch_eval(cfg, logger, model, trainloader, testloader, criterion)
    GFlops = cal_flops(cfg, logger, model)
    evaluation_dictionary['GFlops'] = GFlops

    save_model_evaluation_result_plot(cfg,evaluation_dictionary)
    
    logger.info(f'---------------------------------------------------------------')
    time_elapsed = time.time() - since
    logger.info(f'model evaluation complete in {time_elapsed // 60:.0f}m { time_elapsed % 60:.0f}s')

    return evaluation_dictionary


class ModelEvaluater():

    def __init__(self,cfg: DictConfig):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.task = cfg.task_name

    def one_epoch_eval(self,cfg, logger, model, trainloader, testloader, criterion):
        evaluation_dictionary = {}
        if self.task == "ImageTransmission" or self.task == "FeatureAwareIT" or self.task == "FAwoSIIT":
            evaluation_dictionary = self.eval_task(cfg, logger, model, trainloader, testloader, criterion)
        else:
            raise ValueError(f'{self.task} task train is not implemented yet')
        return evaluation_dictionary
    


    def eval_task(self,cfg: DictConfig, logger, model, trainloader, testloader, criterion):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        evaluation_dictionary = {}
        evaluation_dictionary['task'] = cfg.task_name
        
        test_epoch_total_loss = 0
        test_epoch_performance = 0
        performance_metric = cfg.performance_metric

        count = 0           
        
        for i in range(1):
            for images, labels in testloader:

                count += images.shape[0]
                images = images.to(device)
                with torch.no_grad():
                    images_hat = model(images, SNR_info=cfg.SNR_info)
                total_loss = 0.

                total_loss, performance = criterion(images_hat, images)



                test_epoch_total_loss += total_loss.item() * images.size(0)
                test_epoch_performance += performance.item() * images.size(0)
                
                
        test_epoch_total_loss = test_epoch_total_loss / count
        test_epoch_performance = test_epoch_performance / count

        logger.info(f'Test count per epoch: {count}')
        logger.info(f'Test loss: {test_epoch_total_loss}')
        logger.info(f'{performance_metric}: {test_epoch_performance}')   

        evaluation_dictionary[performance_metric] = test_epoch_performance

        return evaluation_dictionary


def save_model_evaluation_result_plot(cfg,evaluation_dictionary):
    save_folder = "../../test_results/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    result_type_folder = f'{cfg.task_name}/'
    save_path = save_folder + result_type_folder 
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    chan_type = cfg.chan_type
    save_name = save_path + f'{chan_type}_{cfg.data_info.data_name}_{cfg.model_name}_SNR{str(cfg.SNR_info).zfill(3)}' \
                            f'_rcpp{str(cfg.rcpp).zfill(3)}_random_seed{str(cfg.random_seed).zfill(3)}' 
    
    if self.task == "ImageTransmission" or self.task == "FeatureAwareIT" or self.task == "FAwoSIIT":
        save_IT_model_evaluation_result_plot(cfg,evaluation_dictionary,save_name)
    else:
        raise ValueError(f'model evaluation for {cfg.task_name} is not implemented yet')

    """
    elif cfg.task_name == "PwRe03":
        save_PwRe03_model_evaluation_result_plot(cfg,evaluation_dictionary,save_name)
    """

def save_IT_model_evaluation_result_plot(cfg,evaluation_dictionary,save_name):
    
    return None


def visualize_model(cfg: DictConfig, logger, model, trainloader,testloader,visualloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    since = time.time()
    visualizer = Visualizer(cfg)
    
    test_SNR_info = cfg.SNR_info
    
    count = 0
    max_count = 10 #3,10
    index = 1
    max_index = max_count    
    
    for images, labels in visualloader:
        print('visualization start')
        visualizer.visualize_model_inference(cfg,logger, model, images, test_SNR_info,count)
        print('visualization end')
        count +=1
        if count>=max_count:
            break
    


    
    logger.info(f'---------------------------------------------------------------')
    time_elapsed = time.time() - since
    logger.info(f'model visualization complete in {time_elapsed // 60:.0f}m { time_elapsed % 60:.0f}s')

    return None


class Visualizer():


    def __init__(self,cfg: DictConfig):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.task = cfg.task_name

    def visualize_model_inference(self, cfg: DictConfig, logger, model, images, test_SNR_info,index):
        save_folder = "../../test_results/"
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        result_type_folder = f'{cfg.task_name}_visualize/'
        save_path = save_folder + result_type_folder 
        chan_type = cfg.chan_type
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        save_name = save_path + f'{chan_type}_{cfg.data_info.data_name}_Index{str(index).zfill(3)}_{cfg.model_name}_SNR{str(cfg.SNR_info).zfill(3)}' \
                            f'_rcpp{str(cfg.rcpp).zfill(3)}_{cfg.performance_metric}' 

        save_name = save_path + f'{chan_type}_Kodak_Index{str(index).zfill(3)}_{cfg.model_name}_SNR{str(cfg.SNR_info).zfill(3)}' \
                            f'_rcpp{str(cfg.rcpp).zfill(3)}_{cfg.performance_metric}' 

    
        since = time.time()
        if self.task == "ImageTransmission" or self.task == "FeatureAwareIT" or self.task == "FAwoSIIT":
            test_result = self.visualize_IT_test_result(cfg, model, images, test_SNR_info,save_name)
            test_result = self.visualize_IT_patchwise_test_result(cfg, model, images, test_SNR_info,save_name)
        else:
            raise ValueError(f'visualization of {self.task} task is not implemented yet')
        time_elapsed = time.time() - since
        logger.info(f'Visualization complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        return test_result

    def visualize_IT_test_result(self, cfg, model, images, test_SNR_info,save_name):
        #visualize_img_num = cfg.visualize_img_num
        #H,W = cfg.input_resolution
        B,C,H,W = images.shape
        h = H//64 #//2
        w = W//64

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        gpu_images = images.to(device)
        model.to(device)
        model.eval()

        with torch.no_grad():
            images_hat = model(gpu_images, SNR_info=cfg.SNR_info)
        if cfg.performance_metric == "PSNR":
            criterion = imagewisePSNR()
        elif cfg.performance_metric == "SSIM":
            criterion = imagewiseSSIM()
        elif cfg.performance_metric == "MS-SSIM":
            criterion = imagewiseMS_SSIM()
        else:
            raise ValueError(f'Imagewise criterion for {cfg.performance_metric} of {cfg.task_name} task is not implemented yet')   

        image_wise_metric = criterion(images_hat,gpu_images)
        
        images = images.clone().detach().cpu()
        cpu_images_hat = images_hat.clone().detach().cpu()
        images_dim = cpu_images_hat.size() # B X C X H X W
        
        #print(images.size())
        images = images.flatten(2).transpose(1, 2).reshape(images_dim[0],images_dim[2],images_dim[3],images_dim[1])
        cpu_images_hat = cpu_images_hat.flatten(2).transpose(1, 2).reshape(images_dim[0],images_dim[2],images_dim[3],images_dim[1])
        
        
        npimg = (images.numpy()+1)/2 # denormalize    
        npimg = np.clip(npimg, 0, 1) #clip
        
        npimg_hat = (cpu_images_hat.numpy()+1)/2 # denormalize
        npimg_hat = np.clip(npimg_hat, 0, 1) # clip


        
        fig = plt.figure(figsize=(w, h))
        ax = fig.add_subplot(1, 1, 1) 
        ax.imshow(npimg_hat[0])
        ax.set_title('Reconstructed',fontsize=5/2*w) #fontsize=5/2*w
        ax.set_xlabel(f'{cfg.performance_metric}: {round(image_wise_metric[0].item(),2)}',fontsize=5/2*w) #fontsize=5/2*w
        ax.set_xticks([]), ax.set_yticks([])

        save_name = save_name + ".png"
        if save_name:
            plt.savefig(save_name)
        # plt.show()
        plt.clf()
        plt.close()

        return cpu_images_hat

    def visualize_IT_patchwise_test_result(self, cfg, model, images, test_SNR_info,save_name):
        d = 4
        patch_division_info = [[i for i in range(d)] for j in range(d)]
        B,C,H,W = images.shape
        h = H//64 #//2
        w = W//64
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        images = images.to(device)
        model.to(device)
        model.eval()

        with torch.no_grad():
            images_hat = model(images, SNR_info=cfg.SNR_info)
        # reconstruction with second step feature, this should be generalized 
        if cfg.performance_metric == "PSNR":
            criterion = imagewisePSNR()
        elif cfg.performance_metric == "SSIM":
            criterion = imagewiseSSIM()
        elif cfg.performance_metric == "MS-SSIM":
            criterion = imagewiseMS_SSIM()
        else:
            raise ValueError(f'Imagewise criterion for {cfg.performance_metric} of {cfg.task_name} task is not implemented yet')   

        images_dim = images.size() # B X C X H X W
        patchwise_images = patch_division(images,d) # B X d^2 X C X H/d X W/d (patchwise division), B should be larger than 1.
        patchwise_images_dim = patchwise_images.size()
        patchwise_images_hat = patch_division(images_hat,d)
        
        patchwise_image_wise_psnr = patch_wise_calculation(patchwise_images_hat,patchwise_images,criterion)
        
        # B X d^2 X C X H/d X W/d -> B X d^2 X H/d X W/d X C
        patchwise_images = patchwise_images.clone().detach().cpu().flatten(3).transpose(2, 3).reshape(patchwise_images_dim[0],patchwise_images_dim[1],patchwise_images_dim[3],patchwise_images_dim[4],patchwise_images_dim[2])
        patchwise_cpu_images_hat = patchwise_images_hat.clone().detach().cpu().flatten(3).transpose(2, 3).reshape(patchwise_images_dim[0],patchwise_images_dim[1],patchwise_images_dim[3],patchwise_images_dim[4],patchwise_images_dim[2])
        
        
        
        patchwise_npimg = (patchwise_images.numpy()+1)/2 # denormalize      
        patchwise_npimg = np.clip(patchwise_npimg, 0, 1) #clip
        
        patchwise_npimg_hat = np.clip((patchwise_cpu_images_hat.numpy()+1)/2, 0, 1) # denormalize


        #figure.suptitle(f'{cfg.data_info.data_name}, SNR: {cfg.SNR_info},rcpp: {cfg.rcpp}')


        figure = plt.figure(figsize=(w,h))    
        subfig = figure.subfigures(1,1) 
        subfig.subplots_adjust(wspace=0.1,hspace=0.1, left=0, right=0.95, bottom=.04)
        subfig.suptitle(f'Reconstructed', fontsize=20) # 5/2*w              
        axs = subfig.subplots(d,d)                
        for h in range(len(patch_division_info)):
            for w in range(len(patch_division_info[h])):
                ax = axs[h,w]
                ax.imshow(patchwise_npimg_hat[0][h*d+w])
                ax.set_xlabel(f'{cfg.performance_metric}: {round(patchwise_image_wise_psnr[0][h*d+w].item(),2)}',labelpad=2, fontsize=7)                        
                ax.set_xticks([]), ax.set_yticks([])
        save_name = save_name + "_patchwise.png"
        if save_name:
            plt.savefig(save_name)
        # plt.show()
        plt.clf()
        plt.close()

        return None


