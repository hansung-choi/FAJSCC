from data_maker import *
from loss_maker import *
from optimizer_maker import *
from train import *
from model.model_maker import *
from total_eval import *
import random
import os



@hydra.main(version_base = '1.1',config_path="configs",config_name='model_eval')
def main(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'---------------------------------------------------------------')
    logger.info(f'device: {device}')
    
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()

    # set random seed number
    random_seed_num = cfg.random_seed
    torch.manual_seed(random_seed_num)
    np.random.seed(random_seed_num)
    random.seed(random_seed_num)
    
    # make data_info
    data_info = DataMaker(cfg)
    
    DeepJSCC_list = ["smallDeepJSCC","baseDeepJSCC","largeDeepJSCC"]
    DeepJSCCV_list = ["smallDeepJSCC-V","baseDeepJSCC-V","largeDeepJSCC-V"]

    ConvJSCC_list = ["smallConvJSCC","baseConvJSCC","largeConvJSCC"]
    ResJSCC_list = ["smallResJSCC","baseResJSCC","largeResJSCC"]
    SWJSCC_list = ["smallSWJSCC","baseSWJSCC","largeSWJSCC"]
    CAJSCC_list = ["smallCAJSCC","baseCAJSCC","largeCAJSCC"]
    CASAJSCC_list = ["smallCASAJSCC","baseCASAJSCC","largeCASAJSCC"]
    
    
    
    model_name_full_list = DeepJSCC_list + ConvJSCC_list + ResJSCC_list + SWJSCC_list + CAJSCC_list + CASAJSCC_list
    model_full_type_list = ["DeepJSCC","ConvJSCC","ResJSCC","SWJSCC","CAJSCC","CASAJSCC"]

 
    #full_rcpp_list=[12,16,24,32]
    #full_SNR_list=[1,4,7,10]

             

    
    model_name_list = ["smallConvJSCC","baseConvJSCC","smallResJSCC","baseResJSCC","smallSwinJSCC","baseSwinJSCC","smallFAwoSIJSCC","baseFAwoSIJSCC","smallFAJSCC","baseFAJSCC"]
    model_type_list = ["ConvJSCC","ResJSCC","SwinJSCC","FAwoSIJSCC","FAJSCC"]


    model_name_list0 = ["smallConvJSCC","baseConvJSCC","smallResJSCC","baseResJSCC","smallSwinJSCC","baseSwinJSCC","smallFAwoSIJSCC","baseFAwoSIJSCC"]
    model_type_list0 = ["ConvJSCC","ResJSCC","SwinJSCC","FAwoSIJSCC"]
    model_name_list1 = ["smallConvJSCC","smallResJSCC","smallSwinJSCC","smallFAwoSIJSCC","smallFAJSCC"]
    model_name_list2 = ["baseConvJSCC","baseResJSCC","baseSwinJSCC","baseFAwoSIJSCC","baseFAJSCC"]
    
    model_name_list.reverse()
    model_type_list.reverse()

    model_name_list0.reverse()
    model_type_list0.reverse()
        
    model_name_list1.reverse()
    model_name_list2.reverse()
    rcpp_list=[12]
    SNR_list=[7]
    

    total_eval_dict1 = get_total_eval_dict(cfg,logger,model_name_list,rcpp_list,SNR_list)


    save_performance_GFlops_Mmemory_plot(cfg,logger,total_eval_dict1,model_name_list,model_type_list,12,7)
    
    #save_SNR_performance_table(cfg,logger,total_eval_dict1,model_name_list,12,SNR_list)
    
    
    

    
if __name__ == '__main__':
    main()
    














    