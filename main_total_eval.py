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
    

    ConvJSCC_list = ["smallConvJSCC","baseConvJSCC"]
    ResJSCC_list = ["smallResJSCC","baseResJSCC"]
    SwinJSCC_list = ["smallSwinJSCC","baseSwinJSCC"]
    FAwoSIJSCC_list = ["smallFAwoSIJSCC","baseFAwoSIJSCC"]
    FAJSCC_list = ["smallFAJSCC","baseFAJSCC"]
    
    
    
    model_name_full_list = ConvJSCC_list + ResJSCC_list + SwinJSCC_list + FAwoSIJSCC_list + FAJSCC_list
    model_full_type_list = ["ConvJSCC","ResJSCC","SwinJSCC","FAwoSIJSCC","FAJSCC"]

 
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
    SNR_list=[1,4,7,10]
    

    total_eval_dict1 = get_total_eval_dict(cfg,logger,model_name_list,rcpp_list,SNR_list)


    save_GFlops_performance_plot_type2(cfg,logger,total_eval_dict1,model_name_list0,model_type_list0,12,1)
    
    save_Mmemory_performance_plot_type2(cfg,logger,total_eval_dict1,model_name_list0,model_type_list0,12,1)

    save_GFlops_performance_plot_type2(cfg,logger,total_eval_dict1,model_name_list,model_type_list,12,1)
    
    save_Mmemory_performance_plot_type2(cfg,logger,total_eval_dict1,model_name_list,model_type_list,12,1)
    
    save_GFlops_performance_plot_type2(cfg,logger,total_eval_dict1,model_name_list0,model_type_list0,12,4)
    
    save_Mmemory_performance_plot_type2(cfg,logger,total_eval_dict1,model_name_list0,model_type_list0,12,4)

    save_GFlops_performance_plot_type2(cfg,logger,total_eval_dict1,model_name_list,model_type_list,12,4)
    
    save_Mmemory_performance_plot_type2(cfg,logger,total_eval_dict1,model_name_list,model_type_list,12,4)

    save_GFlops_performance_plot_type2(cfg,logger,total_eval_dict1,model_name_list0,model_type_list0,12,7)
    
    save_Mmemory_performance_plot_type2(cfg,logger,total_eval_dict1,model_name_list0,model_type_list0,12,7)

    save_GFlops_performance_plot_type2(cfg,logger,total_eval_dict1,model_name_list,model_type_list,12,7)
    
    save_Mmemory_performance_plot_type2(cfg,logger,total_eval_dict1,model_name_list,model_type_list,12,7)

    save_GFlops_performance_plot_type2(cfg,logger,total_eval_dict1,model_name_list0,model_type_list0,12,10)
    
    save_Mmemory_performance_plot_type2(cfg,logger,total_eval_dict1,model_name_list0,model_type_list0,12,10)

    save_GFlops_performance_plot_type2(cfg,logger,total_eval_dict1,model_name_list,model_type_list,12,10)
    
    save_Mmemory_performance_plot_type2(cfg,logger,total_eval_dict1,model_name_list,model_type_list,12,10)


    
    save_SNR_performance_plot(cfg,logger,total_eval_dict1,model_name_list1,12,SNR_list)
    
    save_SNR_performance_plot(cfg,logger,total_eval_dict1,model_name_list2,12,SNR_list)
    
    save_SNR_performance_table(cfg,logger,total_eval_dict1,model_name_list0,12,SNR_list)
    
    save_SNR_performance_table(cfg,logger,total_eval_dict1,model_name_list1,12,SNR_list)
    
    save_SNR_performance_table(cfg,logger,total_eval_dict1,model_name_list2,12,SNR_list)

    model_name_list12 = ["smallConvJSCC","smallResJSCC","smallSwinJSCC","smallFAwoSIJSCC","smallFAJSCC","baseConvJSCC","baseResJSCC","baseSwinJSCC","baseFAwoSIJSCC","baseFAJSCC"]
    model_name_list12.reverse()
    
    save_SNR_performance_plot(cfg,logger,total_eval_dict1,model_name_list12,12,SNR_list)
        
    
if __name__ == '__main__':
    main()
    














    