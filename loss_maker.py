from model.model_maker import *
from data_maker import *
from torch.nn import functional as F
from utils import *
from einops import rearrange
from torch_msssim import ssim_, ms_ssim_, SSIM_, MS_SSIM_


def get_task_info(cfg):

    if cfg.model_name == "smallConvJSCC" or cfg.model_name == "baseConvJSCC":
        cfg.task_name = "ImageTransmission"
    elif cfg.model_name == "smallResJSCC" or cfg.model_name == "baseResJSCC":
        cfg.task_name = "ImageTransmission"
    elif cfg.model_name == "smallSwinJSCC" or cfg.model_name == "baseSwinJSCC":
        cfg.task_name = "ImageTransmission"
        
    elif cfg.model_name == "smallFAJSCCwSA" or cfg.model_name == "baseFAJSCCwSA":
        cfg.task_name = "FAITwSA"
    elif cfg.model_name == "smallFAJSCCwoSA" or cfg.model_name == "baseFAJSCCwoSA":
        cfg.task_name = "FAITwoSA"

    elif cfg.model_name == "baseFAJSCCwSAr12_00" or cfg.model_name == "baseFAJSCCwSAr12_02" or cfg.model_name == "baseFAJSCCwSAr12_04" or cfg.model_name == "baseFAJSCCwSAr12_05" or cfg.model_name == "baseFAJSCCwSAr12_06" or cfg.model_name == "baseFAJSCCwSAr12_08" or cfg.model_name == "baseFAJSCCwSAr12_10":
        cfg.task_name = "FAITwSA"
    elif cfg.model_name == "baseFAJSCCwSAr1_00" or cfg.model_name == "baseFAJSCCwSAr1_02" or cfg.model_name == "baseFAJSCCwSAr1_04" or cfg.model_name == "baseFAJSCCwSAr1_05" or cfg.model_name == "baseFAJSCCwSAr1_06" or cfg.model_name == "baseFAJSCCwSAr1_08" or cfg.model_name == "baseFAJSCCwSAr1_10":
        cfg.task_name = "FAITwSA"
    elif cfg.model_name == "baseFAJSCCwSAr2_00" or cfg.model_name == "baseFAJSCCwSAr2_02" or cfg.model_name == "baseFAJSCCwSAr2_04" or cfg.model_name == "baseFAJSCCwSAr2_05" or cfg.model_name == "baseFAJSCCwSAr2_06" or cfg.model_name == "baseFAJSCCwSAr2_08" or cfg.model_name == "baseFAJSCCwSAr2_10":
        cfg.task_name = "FAITwSA"
        
        
    elif cfg.model_name == "baseFAJSCCwoSAr12_00" or cfg.model_name == "baseFAJSCCwoSAr12_02" or cfg.model_name == "baseFAJSCCwoSAr12_04" or cfg.model_name == "baseFAJSCCwoSAr12_05" or cfg.model_name == "baseFAJSCCwoSAr12_06" or cfg.model_name == "baseFAJSCCwoSAr12_08" or cfg.model_name == "baseFAJSCCwoSAr12_10":
        cfg.task_name = "FAITwoSA"
    elif cfg.model_name == "baseFAJSCCwoSAr1_00" or cfg.model_name == "baseFAJSCCwoSAr1_02" or cfg.model_name == "baseFAJSCCwoSAr1_04" or cfg.model_name == "baseFAJSCCwoSAr1_05" or cfg.model_name == "baseFAJSCCwoSAr1_06" or cfg.model_name == "baseFAJSCCwoSAr1_08" or cfg.model_name == "baseFAJSCCwoSAr1_10":
        cfg.task_name = "FAITwoSA"
    elif cfg.model_name == "baseFAJSCCwoSAr2_00" or cfg.model_name == "baseFAJSCCwoSAr2_02" or cfg.model_name == "baseFAJSCCwoSAr2_04" or cfg.model_name == "baseFAJSCCwoSAr2_05" or cfg.model_name == "baseFAJSCCwoSAr2_06" or cfg.model_name == "baseFAJSCCwoSAr2_08" or cfg.model_name == "baseFAJSCCwoSAr2_10":
        cfg.task_name = "FAITwoSA"        
        
        
    else:
        raise ValueError(f'task for {cfg.model_name} model is not implemented yet')

def get_loss_info(cfg):
    cfg.loss_name = None
    get_task_info(cfg)
    
    if cfg.task_name == "ImageTransmission":
        if cfg.performance_metric == "PSNR":
            cfg.loss_name = "IT_MSE"
        elif cfg.performance_metric == "SSIM":
            cfg.loss_name = "IT_SSIM"
        else:
            raise ValueError(f'loss function for {cfg.performance_metric} of {cfg.task_name} task is not implemented yet')   
    elif cfg.task_name == "FAITwSA" or cfg.task_name == "FAITwoSA":
        if cfg.performance_metric == "PSNR":
            cfg.loss_name = "FAIT_MSE"
        elif cfg.performance_metric == "SSIM":
            cfg.loss_name = "FAIT_SSIM"
        else:
            raise ValueError(f'loss function for {cfg.performance_metric} of {cfg.task_name} task is not implemented yet')   
    else:
        raise ValueError(f'loss function for {cfg.task_name} task is not implemented yet')       

def LossMaker(cfg,d=4): #cfg: DictConfig
    get_loss_info(cfg)
    if cfg.loss_name == "IT_MSE":
        loss = IT_MSE()
    elif cfg.loss_name == "IT_SSIM":
        loss = IT_SSIM() 
    elif cfg.loss_name == "FAIT_MSE":
        loss = FAIT_MSE()
    elif cfg.loss_name == "FAIT_SSIM":
        loss = FAIT_SSIM() 
    else:
        raise ValueError(f'{cfg.loss_name} is not implemented yet')
    return loss


class IT_MSE(torch.nn.Module):
    def __init__(self):
        super(IT_MSE, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.mse = nn.MSELoss()

    def forward(self, image_hat, image):
    	# inputs => N x C x H x W
        image_hat = image_hat.to(self.device)
        image = image.to(self.device)
        #image_dim = image.size()
        
      # [-1 1] to [0 1]
        image_hat = (image_hat+1)/2
        image = (image+1)/2

        mse = self.mse(image_hat, image)
        total_loss = mse
        psnr = 10 * (np.log(1. / mse.clone().detach().cpu()) / np.log(10))

        return total_loss, psnr
        
    def get_performance_metric(self):
        return "PSNR"

        
class IT_SSIM(torch.nn.Module):
    def __init__(self):
        super(IT_SSIM, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

    def forward(self, image_hat, image):
    	# inputs => N x C x H x W
        image_hat = image_hat.to(self.device)
        image = image.to(self.device)
        #image_dim = image.size()
        
      # [-1 1] to [0 1]
        image_hat = (image_hat+1)/2
        image = (image+1)/2

        ssim = ssim_(image_hat, image, data_range=1, size_average=True)
        total_loss = 1-ssim

        return total_loss, ssim.clone().detach().cpu()
        
    def get_performance_metric(self):
        return "SSIM"



class FAIT_MSE(torch.nn.Module):
    def __init__(self, CA_ratio=0.5,gamma = 0.5):
        super(FAIT_MSE, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.mse = nn.MSELoss()
        self.CA_ratio = CA_ratio
        self.gamma = gamma

    def forward(self, image_hat, image,decision=None):
    	# inputs => N x C x H x W
        image_hat = image_hat.to(self.device)
        image = image.to(self.device)
        #image_dim = image.size()
        
      # [-1 1] to [0 1]
        image_hat = (image_hat+1)/2
        image = (image+1)/2

        mse = self.mse(image_hat, image)
        total_loss = mse
        if decision:
            mask_loss = 2*self.gamma*(torch.mean(torch.cat(decision,dim=1),dim=(0,1))-self.CA_ratio)**2
            mask_loss = mask_loss.to(self.device)
            total_loss += mask_loss
        psnr = 10 * (np.log(1. / mse.clone().detach().cpu()) / np.log(10))
        
        return mse, psnr
        
    def get_performance_metric(self):
        return "PSNR"

        
class FAIT_SSIM(torch.nn.Module):
    def __init__(self, CA_ratio=0.5,gamma = 0.5):
        super(FAIT_SSIM, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.CA_ratio = CA_ratio
        self.gamma = gamma

    def forward(self, image_hat, image,decision=None):
    	# inputs => N x C x H x W
        image_hat = image_hat.to(self.device)
        image = image.to(self.device)
        #image_dim = image.size()
        
      # [-1 1] to [0 1]
        image_hat = (image_hat+1)/2
        image = (image+1)/2

        ssim = ssim_(image_hat, image, data_range=1, size_average=True)
        total_loss = 1-ssim
        if decision:
            mask_loss = 2*self.gamma*(torch.mean(torch.cat(decision,dim=1),dim=(0,1))-self.CA_ratio)**2
            mask_loss = mask_loss.to(self.device)
            total_loss += mask_loss

        return total_loss, ssim.clone().detach().cpu()
        
    def get_performance_metric(self):
        return "SSIM"




class imagewisePSNR(torch.nn.Module):
    def __init__(self):
        super(imagewisePSNR, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, image_hat, image):
    	# inputs => N x C x H x W
        image_hat = image_hat.to(self.device)
        image = image.to(self.device)
        #image_dim = image.size()
        
      # [-1 1] to [0 1]
        image_hat = (image_hat+1)/2
        image = (image+1)/2

        unreduced_mse = self.mse(image_hat, image)
        image_wise_mse = unreduced_mse.mean(dim=[i for i in range(1,len(image.size()))]).reshape(-1).clone().detach().cpu()

        image_wise_psnr = 10 * (np.log(1. / image_wise_mse) / np.log(10))
        return image_wise_psnr



class imagewiseSSIM(torch.nn.Module):
    def __init__(self):
        super(imagewiseSSIM, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

    def forward(self, image_hat, image):
    	# inputs => N x C x H x W
        image_hat = image_hat.to(self.device)
        image = image.to(self.device)
        #image_dim = image.size()
        
      # [-1 1] to [0 1]
        image_hat = (image_hat+1)/2
        image = (image+1)/2

        unreduced_SSIM = ssim_(image_hat, image, data_range=1, size_average=False)
        image_wise_SSIM = unreduced_SSIM.reshape(-1).clone().detach().cpu()

        return image_wise_SSIM







        