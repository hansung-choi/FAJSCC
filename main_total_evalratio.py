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
    
    model_name_list_r12 = ["baseFAJSCCwSAr12_00","baseFAJSCCwSAr12_02","baseFAJSCCwSAr12_04","baseFAJSCCwSAr12_05","baseFAJSCCwSAr12_06","baseFAJSCCwSAr12_08","baseFAJSCCwSAr12_10","baseFAJSCCwoSAr12_00","baseFAJSCCwoSAr12_02","baseFAJSCCwoSAr12_04","baseFAJSCCwoSAr12_05","baseFAJSCCwoSAr12_06","baseFAJSCCwoSAr12_08","baseFAJSCCwoSAr12_10"]

    model_name_list_r1 = ["baseFAJSCCwSAr1_00","baseFAJSCCwSAr1_02","baseFAJSCCwSAr1_04","baseFAJSCCwSAr1_05","baseFAJSCCwSAr1_06","baseFAJSCCwSAr1_08","baseFAJSCCwSAr1_10","baseFAJSCCwoSAr1_00","baseFAJSCCwoSAr1_02","baseFAJSCCwoSAr1_04","baseFAJSCCwoSAr1_05","baseFAJSCCwoSAr1_06","baseFAJSCCwoSAr1_08","baseFAJSCCwoSAr1_10"]

    model_name_list_r2 = ["baseFAJSCCwSAr2_00","baseFAJSCCwSAr2_02","baseFAJSCCwSAr2_04","baseFAJSCCwSAr2_05","baseFAJSCCwSAr2_06","baseFAJSCCwSAr2_08","baseFAJSCCwSAr2_10","baseFAJSCCwoSAr2_00","baseFAJSCCwoSAr2_02","baseFAJSCCwoSAr2_04","baseFAJSCCwoSAr2_05","baseFAJSCCwoSAr2_06","baseFAJSCCwoSAr2_08","baseFAJSCCwoSAr2_10"]
    
    model_name_list = ["baseSwinJSCC"]


    model_type_list_r1 = ["FAJSCCwSAr1","FAJSCCwoSAr1"]
    model_type_list_r2 = ["FAJSCCwSAr2","FAJSCCwoSAr2"]
    model_type_list_r12 = ["FAJSCCwSAr12","FAJSCCwoSAr12"]

    
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
    














    