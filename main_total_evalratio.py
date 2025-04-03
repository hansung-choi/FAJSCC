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
    
    model_name_list_r12 = ["baseFAwoSIJSCCr12_00","baseFAwoSIJSCCr12_02","baseFAwoSIJSCCr12_04","baseFAwoSIJSCCr12_05","baseFAwoSIJSCCr12_06","baseFAwoSIJSCCr12_08","baseFAwoSIJSCCr12_10","baseFAJSCCr12_00","baseFAJSCCr12_02","baseFAJSCCr12_04","baseFAJSCCr12_05","baseFAJSCCr12_06","baseFAJSCCr12_08","baseFAJSCCr12_10"]

    model_name_list_r1 = ["baseFAwoSIJSCCr1_00","baseFAwoSIJSCCr1_02","baseFAwoSIJSCCr1_04","baseFAwoSIJSCCr1_05","baseFAwoSIJSCCr1_06","baseFAwoSIJSCCr1_08","baseFAwoSIJSCCr1_10","baseFAJSCCr1_00","baseFAJSCCr1_02","baseFAJSCCr1_04","baseFAJSCCr1_05","baseFAJSCCr1_06","baseFAJSCCr1_08","baseFAJSCCr1_10"]

    model_name_list_r2 = ["baseFAwoSIJSCCr2_00","baseFAwoSIJSCCr2_02","baseFAwoSIJSCCr2_04","baseFAwoSIJSCCr2_05","baseFAwoSIJSCCr2_06","baseFAwoSIJSCCr2_08","baseFAwoSIJSCCr2_10","baseFAJSCCr2_00","baseFAJSCCr2_02","baseFAJSCCr2_04","baseFAJSCCr2_05","baseFAJSCCr2_06","baseFAJSCCr2_08","baseFAJSCCr2_10"]
    
    model_name_list = ["baseSwinJSCC"]


    model_type_list_r1 = ["FAwoSIJSCCr1","FAJSCCr1"]
    model_type_list_r2 = ["FAwoSIJSCCr2","FAJSCCr2"]
    model_type_list_r12 = ["FAwoSIJSCCr12","FAJSCCr12"]

    
    model_name_list_r12.reverse()
    model_name_list_r1.reverse()
    model_name_list_r2.reverse()
    
    model_name_list.reverse()
    
    model_type_list_r12.reverse()
    model_type_list_r1.reverse()
    model_type_list_r2.reverse()
     
    
    rcpp_list=[12]
    SNR_list=[1,10]
    
    total_model_list = model_name_list_r12 + model_name_list_r1 + model_name_list_r2 + model_name_list
    
    
    total_eval_dict = get_total_eval_dict(cfg,logger,total_model_list,rcpp_list,SNR_list)
    
    save_GFlops_performance_ratio_plot(cfg,logger,total_eval_dict,model_name_list_r1,model_name_list_r12,model_name_list_r2,model_name_list,model_type_list_r1,model_type_list_r12,model_type_list_r2,12,1)
    
    save_GFlops_performance_ratio_plot(cfg,logger,total_eval_dict,model_name_list_r1,model_name_list_r12,model_name_list_r2,model_name_list,model_type_list_r1,model_type_list_r12,model_type_list_r2,12,10)
    
    
    
    

    
if __name__ == '__main__':
    main()
    














    