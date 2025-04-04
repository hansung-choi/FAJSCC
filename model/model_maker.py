from .common_component import *
from .JSCC import *

def get_model_info(cfg):
    model_info = dict()
    model_info['chan_type'] = cfg.chan_type
    model_info['color_channel'] = cfg.data_info.color_channel
    model_info['rcpp'] = cfg.rcpp #reverse of channel per pixel
    model_info['window_size'] = 8
    model_info['ratio'] = 0.5
    cfg.ratio = model_info['ratio'] #importance ratio
    model_info['gamma'] = 0.5
    cfg.gamma = model_info['gamma']
    model_info['ratio1'] = 0.5
    cfg.ratio1 = model_info['ratio1'] # encoder's importance ratio
    model_info['ratio2'] = 0.5
    cfg.ratio2 = model_info['ratio2'] # decoder's importance ratio

    model_info['window_size_list'] = [8,8,8,8]
    model_info['num_heads_list'] = [4, 6, 8, 10] ##careful! n_feats_list[i]/num_heads_list[i] should be integer
    model_info['input_resolution'] = cfg.input_resolution
    model_info['n_block_list'] = [2,2,2,2]
    
    if cfg.model_name == "smallConvJSCC":
        model_info['n_feats_list'] = [32,32,32,32]  
    elif cfg.model_name == "baseConvJSCC":
        model_info['n_feats_list'] = [64,64,64,64]
    elif cfg.model_name == "smallResJSCC":
        model_info['n_feats_list'] = [32,32,32,32]
    elif cfg.model_name == "baseResJSCC":
        model_info['n_feats_list'] = [64,64,64,64]
    elif cfg.model_name == "smallSwinJSCC":
        model_info['n_feats_list'] = [40,60,80,160] 
    elif cfg.model_name == "baseSwinJSCC":
        model_info['n_feats_list'] = [60,90,120,200] 
    elif cfg.model_name == "smallFAJSCCwSA" or cfg.model_name == "smallFAJSCCwoSA":
        model_info['n_feats_list'] = [40,60,80,260] 
    elif cfg.model_name == "baseFAJSCCwSA" or cfg.model_name == "baseFAJSCCwoSA":
        model_info['n_feats_list'] = [60,90,120,360] 

    elif cfg.model_name == "baseFAJSCCwSAr12_00" or cfg.model_name == "baseFAJSCCwoSAr12_00":
        model_info['n_feats_list'] = [60,90,120,360]
        model_info['ratio1'] = 0.0
        model_info['ratio2'] = 0.0
        cfg.ratio1 = model_info['ratio1']
        cfg.ratio2 = model_info['ratio2']         
    elif cfg.model_name == "baseFAJSCCwSAr12_02" or cfg.model_name == "baseFAJSCCwoSAr12_02":
        model_info['n_feats_list'] = [60,90,120,360]
        model_info['ratio1'] = 0.2
        model_info['ratio2'] = 0.2
        cfg.ratio1 = model_info['ratio1']
        cfg.ratio2 = model_info['ratio2'] 
    elif cfg.model_name == "baseFAJSCCwSAr12_04" or cfg.model_name == "baseFAJSCCwoSAr12_04":
        model_info['n_feats_list'] = [60,90,120,360]
        model_info['ratio1'] = 0.4
        model_info['ratio2'] = 0.4
        cfg.ratio1 = model_info['ratio1']
        cfg.ratio2 = model_info['ratio2'] 
    elif cfg.model_name == "baseFAJSCCwSAr12_05" or cfg.model_name == "baseFAJSCCwoSAr12_05":
        model_info['n_feats_list'] = [60,90,120,360]
        model_info['ratio1'] = 0.5
        model_info['ratio2'] = 0.5
        cfg.ratio1 = model_info['ratio1']
        cfg.ratio2 = model_info['ratio2'] 
    elif cfg.model_name == "baseFAJSCCwSAr12_06" or cfg.model_name == "baseFAJSCCwoSAr12_06":
        model_info['n_feats_list'] = [60,90,120,360]
        model_info['ratio1'] = 0.6
        model_info['ratio2'] = 0.6
        cfg.ratio1 = model_info['ratio1']
        cfg.ratio2 = model_info['ratio2'] 
    elif cfg.model_name == "baseFAJSCCwSAr12_08" or cfg.model_name == "baseFAJSCCwoSAr12_08":
        model_info['n_feats_list'] = [60,90,120,360]
        model_info['ratio1'] = 0.8
        model_info['ratio2'] = 0.8
        cfg.ratio1 = model_info['ratio1']
        cfg.ratio2 = model_info['ratio2'] 
    elif cfg.model_name == "baseFAJSCCwSAr12_10" or cfg.model_name == "baseFAJSCCwoSAr12_10":
        model_info['n_feats_list'] = [60,90,120,360]
        model_info['ratio1'] = 1.0
        model_info['ratio2'] = 1.0
        cfg.ratio1 = model_info['ratio1']
        cfg.ratio2 = model_info['ratio2'] 
        
        
        
    elif cfg.model_name == "baseFAJSCCwSAr1_00" or cfg.model_name == "baseFAJSCCwoSAr1_00":
        model_info['n_feats_list'] = [60,90,120,360]
        model_info['ratio1'] = 0.0
        cfg.ratio1 = model_info['ratio1']       
    elif cfg.model_name == "baseFAJSCCwSAr1_02" or cfg.model_name == "baseFAJSCCwoSAr1_02":
        model_info['n_feats_list'] = [60,90,120,360]
        model_info['ratio1'] = 0.2
        cfg.ratio1 = model_info['ratio1']
    elif cfg.model_name == "baseFAJSCCwSAr1_04" or cfg.model_name == "baseFAJSCCwoSAr1_04":
        model_info['n_feats_list'] = [60,90,120,360]
        model_info['ratio1'] = 0.4
        cfg.ratio1 = model_info['ratio1']
    elif cfg.model_name == "baseFAJSCCwSAr1_05" or cfg.model_name == "baseFAJSCCwoSAr1_05":
        model_info['n_feats_list'] = [60,90,120,360]
        model_info['ratio1'] = 0.5
        cfg.ratio1 = model_info['ratio1']
    elif cfg.model_name == "baseFAJSCCwSAr1_06" or cfg.model_name == "baseFAJSCCwoSAr1_06":
        model_info['n_feats_list'] = [60,90,120,360]
        model_info['ratio1'] = 0.6
        cfg.ratio1 = model_info['ratio1']
    elif cfg.model_name == "baseFAJSCCwSAr1_08" or cfg.model_name == "baseFAJSCCwoSAr1_08":
        model_info['n_feats_list'] = [60,90,120,360]
        model_info['ratio1'] = 0.8
        cfg.ratio1 = model_info['ratio1']
    elif cfg.model_name == "baseFAJSCCwSAr1_10" or cfg.model_name == "baseFAJSCCwoSAr1_10":
        model_info['n_feats_list'] = [60,90,120,360]
        model_info['ratio1'] = 1.0
        cfg.ratio1 = model_info['ratio1']
        
        
        
    elif cfg.model_name == "baseFAJSCCwSAr2_00" or cfg.model_name == "baseFAJSCCwoSAr2_00":
        model_info['n_feats_list'] = [60,90,120,360]
        model_info['ratio2'] = 0.0
        cfg.ratio2 = model_info['ratio2']         
    elif cfg.model_name == "baseFAJSCCwSAr2_02" or cfg.model_name == "baseFAJSCCwoSAr2_02":
        model_info['n_feats_list'] = [60,90,120,360]
        model_info['ratio2'] = 0.2
        cfg.ratio2 = model_info['ratio2'] 
    elif cfg.model_name == "baseFAJSCCwSAr2_04" or cfg.model_name == "baseFAJSCCwoSAr2_04":
        model_info['n_feats_list'] = [60,90,120,360]
        model_info['ratio2'] = 0.4
        cfg.ratio2 = model_info['ratio2'] 
    elif cfg.model_name == "baseFAJSCCwSAr2_05" or cfg.model_name == "baseFAJSCCwoSAr2_05":
        model_info['n_feats_list'] = [60,90,120,360]
        model_info['ratio2'] = 0.5
        cfg.ratio2 = model_info['ratio2'] 
    elif cfg.model_name == "baseFAJSCCwSAr2_06" or cfg.model_name == "baseFAJSCCwoSAr2_06":
        model_info['n_feats_list'] = [60,90,120,360]
        model_info['ratio2'] = 0.6
        cfg.ratio2 = model_info['ratio2'] 
    elif cfg.model_name == "baseFAJSCCwSAr2_08" or cfg.model_name == "baseFAJSCCwoSAr2_08":
        model_info['n_feats_list'] = [60,90,120,360]
        model_info['ratio2'] = 0.8
        cfg.ratio2 = model_info['ratio2'] 
    elif cfg.model_name == "baseFAJSCCwSAr2_10" or cfg.model_name == "baseFAJSCCwoSAr2_10":
        model_info['n_feats_list'] = [60,90,120,360]
        model_info['ratio2'] = 1.0
        cfg.ratio2 = model_info['ratio2'] 

    else:
        raise ValueError(f'n_feats_list for {cfg.model_name} model is not implemented yet')

    return model_info


def ModelMaker(cfg):
    model = None
    model_info = dict()
    
    model_info = get_model_info(cfg)

    if cfg.model_name == "smallConvJSCC" or cfg.model_name == "baseConvJSCC":
        model = ConvJSCC(model_info)
    elif cfg.model_name == "smallResJSCC" or cfg.model_name == "baseResJSCC":
        model = ResJSCC(model_info)
    elif cfg.model_name == "smallSwinJSCC" or cfg.model_name == "baseSwinJSCC":
        model = SwinJSCC(model_info)
    elif cfg.model_name == "smallFAJSCCwSA" or cfg.model_name == "baseFAJSCCwSA":
        model = FAJSCCwSA(model_info)
    elif cfg.model_name == "smallFAJSCCwoSA" or cfg.model_name == "baseFAJSCCwoSA":
        model = FAJSCCwoSA(model_info)

    elif cfg.model_name == "baseFAJSCCwSAr12_00" or cfg.model_name == "baseFAJSCCwSAr12_02" or cfg.model_name == "baseFAJSCCwSAr12_04" or cfg.model_name == "baseFAJSCCwSAr12_05" or cfg.model_name == "baseFAJSCCwSAr12_06" or cfg.model_name == "baseFAJSCCwSAr12_08" or cfg.model_name == "baseFAJSCCwSAr12_10":
        model = FAJSCCwSA(model_info)
    elif cfg.model_name == "baseFAJSCCwSAr1_00" or cfg.model_name == "baseFAJSCCwSAr1_02" or cfg.model_name == "baseFAJSCCwSAr1_04" or cfg.model_name == "baseFAJSCCwSAr1_05" or cfg.model_name == "baseFAJSCCwSAr1_06" or cfg.model_name == "baseFAJSCCwSAr1_08" or cfg.model_name == "baseFAJSCCwSAr1_10":
        model = FAJSCCwSA(model_info)
    elif cfg.model_name == "baseFAJSCCwSAr2_00" or cfg.model_name == "baseFAJSCCwSAr2_02" or cfg.model_name == "baseFAJSCCwSAr2_04" or cfg.model_name == "baseFAJSCCwSAr2_05" or cfg.model_name == "baseFAJSCCwSAr2_06" or cfg.model_name == "baseFAJSCCwSAr2_08" or cfg.model_name == "baseFAJSCCwSAr2_10":
        model = FAJSCCwSA(model_info)
        
        
    elif cfg.model_name == "baseFAJSCCwoSAr12_00" or cfg.model_name == "baseFAJSCCwoSAr12_02" or cfg.model_name == "baseFAJSCCwoSAr12_04" or cfg.model_name == "baseFAJSCCwoSAr12_05" or cfg.model_name == "baseFAJSCCwoSAr12_06" or cfg.model_name == "baseFAJSCCwoSAr12_08" or cfg.model_name == "baseFAJSCCwoSAr12_10":
        model = FAJSCCwoSA(model_info)
    elif cfg.model_name == "baseFAJSCCwoSAr1_00" or cfg.model_name == "baseFAJSCCwoSAr1_02" or cfg.model_name == "baseFAJSCCwoSAr1_04" or cfg.model_name == "baseFAJSCCwoSAr1_05" or cfg.model_name == "baseFAJSCCwoSAr1_06" or cfg.model_name == "baseFAJSCCwoSAr1_08" or cfg.model_name == "baseFAJSCCwoSAr1_10":
        model = FAJSCCwoSA(model_info)
    elif cfg.model_name == "baseFAJSCCwoSAr2_00" or cfg.model_name == "baseFAJSCCwoSAr2_02" or cfg.model_name == "baseFAJSCCwoSAr2_04" or cfg.model_name == "baseFAJSCCwoSAr2_05" or cfg.model_name == "baseFAJSCCwoSAr2_06" or cfg.model_name == "baseFAJSCCwoSAr2_08" or cfg.model_name == "baseFAJSCCwoSAr2_10":
        model = FAJSCCwoSA(model_info)

    else:
        raise ValueError(f'{cfg.model_name} model is not implemented yet')
    return model
    
    
    
    
    
    
    
    




















































