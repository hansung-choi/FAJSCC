from data_maker import *
from loss_maker import *
from optimizer_maker import *
from train import *
from model.model_maker import *
from model_eval import *
from matplotlib.pyplot import cm
import csv
import random
import os
import gc

def get_model_save_name(cfg,model_name,rcpp,SNR):
    data = cfg.data_info.data_name
    cfg.model_name = model_name
    get_loss_info(cfg)
    get_task_info(cfg)
    task = cfg.task_name
    cfg.SNR_info = SNR
    chan_type = cfg.chan_type
    SNR = str(cfg.SNR_info).zfill(3)
    cfg.rcpp = rcpp
    rcpp = str(cfg.rcpp).zfill(3)
    metric = cfg.performance_metric
    random_seed_num = cfg.random_seed
    random_num = str(random_seed_num).zfill(3)
      
    save_name = f"{task}_{data}_{chan_type}_SNR{SNR}_rcpp{rcpp}_{metric}_{model_name}.pt"

    return save_name    


def save_Mmemory_performance_plot(cfg,logger,total_eval_dict,model_name_list,rcpp,SNR):
    save_folder = "../../test_results/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    chan_type = cfg.chan_type
    metric = cfg.performance_metric
    plot_save_name = f"{chan_type}_Mmemory_{metric}_at_SNR{str(SNR).zfill(3)}_rcpp{str(rcpp).zfill(3)}"
    if len(model_name_list)==0:
        return None
    
    for model_name in model_name_list:
        plot_save_name += "_" + model_name

 
    color_list = ['r-','b-','g-','c-','m-','y-','k-','w-'] #https://stackoverflow.com/questions/4971269/how-to-pick-a-new-color-for-each-plotted-line-within-a-figure
    #marker_list =  #https://stackoverflow.com/questions/59647765/how-to-obtain-a-list-of-all-markers-in-matplotlib
    #color_list = ['r-','b-','g-','c-','m-','y-','k-','w-']
    color_list = ['r-','b-','g-','c-','m-','r--','b--','g--','c--','m--']    
    color_list = ['b-','r-','g-','c-','m-','b--','r--','g--','c--','m--']    
    
    n = len(model_name_list)
    color = cm.rainbow(np.linspace(0, 1, n))

    

    plt.rcParams["figure.figsize"] = (14,8)
    
    fig, ax1 = plt.subplots()
    line_list = []
    
    for i, model_name in enumerate(model_name_list):
        valid_Mmemory_list = []
        valid_performance_list = []
        model_save_name = get_model_save_name(cfg,model_name,rcpp,SNR)
        eval_dict = total_eval_dict[model_save_name]            
            
        if eval_dict:
            Mmemory = eval_dict['Mmemory']
            valid_Mmemory_list.append(Mmemory)
            performance = eval_dict[metric]
            valid_performance_list.append(performance)
        line = ax1.plot(valid_Mmemory_list, valid_performance_list, color_list[i],label=f'{cfg.model_name}',marker='o',linewidth=3,markersize=6)
        line_list.append(line)   
        
    lines = []
    for line in line_list:
        lines += line


    ax1.set_xlabel('Memory (MB)', fontsize=20)
    ax1.set_ylabel(metric, fontsize=20)
    plt.xticks( fontsize=14)
    plt.yticks(fontsize=14)

    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=10)
    plt.tight_layout(rect=[0,0,0.6,0.8])
    plt.title(f'{cfg.chan_type}, CPP=1/{rcpp}, SNR={SNR}dB', fontdict = {'fontsize' : 20})

    save_name = save_folder + plot_save_name + ".pdf"
    if save_name:
        plt.savefig(save_name)
    plt.clf()
    logger.info(f'{plot_save_name} is saved')


def save_Mparams_performance_plot(cfg,logger,total_eval_dict,model_name_list,rcpp,SNR):
    save_folder = "../../test_results/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    chan_type = cfg.chan_type
    metric = cfg.performance_metric
    plot_save_name = f"{chan_type}_Mparams_{metric}_at_SNR{str(SNR).zfill(3)}_rcpp{str(rcpp).zfill(3)}"
    if len(model_name_list)==0:
        return None
    
    for model_name in model_name_list:
        plot_save_name += "_" + model_name


    #color_list = ['r-','b-','g-','c-','m-','y-','k-','w-']
    color_list = ['b-','r-','g-','c-','m-','b--','r--','g--','c--','m--']   
    n = len(model_name_list)
    color = cm.rainbow(np.linspace(0, 1, n))


    plt.rcParams["figure.figsize"] = (14,8)
    
    fig, ax1 = plt.subplots()
    line_list = []
    
    for i, model_name in enumerate(model_name_list):
        valid_Mparams_list = []
        valid_performance_list = []
        model_save_name = get_model_save_name(cfg,model_name,rcpp,SNR)
        eval_dict = total_eval_dict[model_save_name]            
            
        if eval_dict:
            Mparams = eval_dict['Mparams']
            valid_Mparams_list.append(Mparams)
            performance = eval_dict[metric]
            valid_performance_list.append(performance)
        line = ax1.plot(valid_Mparams_list, valid_performance_list, color_list[i],label=f'{cfg.model_name}',marker='o',linewidth=3,markersize=6)
        line_list.append(line)   
        
    lines = []
    for line in line_list:
        lines += line


    ax1.set_xlabel('Params (M)', fontsize=20)
    ax1.set_ylabel(metric, fontsize=20)
    plt.xticks( fontsize=14)
    plt.yticks(fontsize=14)

    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=10)
    plt.tight_layout(rect=[0,0,0.6,0.8])
    plt.title(f'{cfg.chan_type}, CPP=1/{rcpp}, SNR={SNR}dB', fontdict = {'fontsize' : 20})

    save_name = save_folder + plot_save_name + ".pdf"
    if save_name:
        plt.savefig(save_name)
    plt.clf()
    logger.info(f'{plot_save_name} is saved')



def save_GFlops_performance_plot(cfg,logger,total_eval_dict,model_name_list,rcpp,SNR):
    save_folder = "../../test_results/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    chan_type = cfg.chan_type
    metric = cfg.performance_metric
    plot_save_name = f"{chan_type}_GFlops_{metric}_at_SNR{str(SNR).zfill(3)}_rcpp{str(rcpp).zfill(3)}"
    if len(model_name_list)==0:
        return None
    
    for model_name in model_name_list:
        plot_save_name += "_" + model_name


    #color_list = ['r-','b-','g-','c-','m-','y-','k-','w-']
    color_list = ['b-','r-','g-','c-','m-','b--','r--','g--','c--','m--']   
    n = len(model_name_list)
    color = cm.rainbow(np.linspace(0, 1, n))


    plt.rcParams["figure.figsize"] = (14,8)
    
    fig, ax1 = plt.subplots()
    line_list = []
    
    for i, model_name in enumerate(model_name_list):
        valid_GFlops_list = []
        valid_performance_list = []
        model_save_name = get_model_save_name(cfg,model_name,rcpp,SNR)
        eval_dict = total_eval_dict[model_save_name]            
            
        if eval_dict:
            GFlops = eval_dict['GFlops']
            valid_GFlops_list.append(GFlops)
            performance = eval_dict[metric]
            valid_performance_list.append(performance)
        line = ax1.plot(valid_GFlops_list, valid_performance_list, color_list[i],label=f'{cfg.model_name}',marker='o',linewidth=3,markersize=6)
        line_list.append(line)   
        
    lines = []
    for line in line_list:
        lines += line


    ax1.set_xlabel('GFlops', fontsize=20)
    ax1.set_ylabel(metric, fontsize=20)
    plt.xticks( fontsize=14)
    plt.yticks(fontsize=14)

    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=10)
    plt.tight_layout(rect=[0,0,0.6,0.8])
    plt.title(f'{cfg.chan_type}, CPP=1/{rcpp}, SNR={SNR}dB', fontdict = {'fontsize' : 20})

    save_name = save_folder + plot_save_name + ".pdf"
    if save_name:
        plt.savefig(save_name)
    plt.clf()
    logger.info(f'{plot_save_name} is saved')

def save_Mmemory_performance_plot_type2(cfg,logger,total_eval_dict,model_name_list,model_type_list,rcpp,SNR):
    save_folder = "../../test_results/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    chan_type = cfg.chan_type
    metric = cfg.performance_metric
    plot_save_name = f"{chan_type}_Mmemory_{metric}_at_SNR{str(SNR).zfill(3)}_rcpp{str(rcpp).zfill(3)}_type2"
    if len(model_name_list)==0:
        return None
    
    for model_type in model_type_list:
        plot_save_name += "_" + model_type

    num_size = len(model_name_list)//len(model_type_list)
    th = 0
    m_type_index = 0

    #color_list = ['r-','b-','g-','c-','m-','y-','k-','w-']
    color_list = ['b-','r-','g-','c-','m-','b--','r--','g--','c--','m--']  
    
    plt.rcParams["figure.figsize"] = (14,8)
    
    fig, ax1 = plt.subplots()
    line_list = []
    
    valid_Mmemory_list = []
    valid_performance_list = []    
    for i, model_name in enumerate(model_name_list):
        model_save_name = get_model_save_name(cfg,model_name,rcpp,SNR)
        eval_dict = total_eval_dict[model_save_name]            
            
        if eval_dict:
            Mmemory = eval_dict['Mmemory']
            valid_Mmemory_list.append(Mmemory)
            performance = eval_dict[metric]
            valid_performance_list.append(performance)
        
        th +=1            
        if th >= num_size:
            line = ax1.plot(valid_Mmemory_list, valid_performance_list,color_list[m_type_index],label=f'{model_type_list[m_type_index]}',marker='o',linewidth=3,markersize=6)
            line_list.append(line)
            m_type_index +=1
            th = 0
            valid_Mmemory_list = []
            valid_performance_list = []    
   
        
    lines = []
    for line in line_list:
        lines += line


    ax1.set_xlabel('Memory (MB)', fontsize=20)
    ax1.set_ylabel(metric, fontsize=20)
    plt.xticks( fontsize=14)
    plt.yticks(fontsize=14)

    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=10)
    plt.tight_layout(rect=[0,0,0.6,0.8])
    plt.title(f'{cfg.chan_type}, CPP=1/{rcpp}, SNR={SNR}dB', fontdict = {'fontsize' : 20})

    save_name = save_folder + plot_save_name + ".pdf"
    if save_name:
        plt.savefig(save_name)
    plt.clf()
    logger.info(f'{plot_save_name} is saved')


def save_Mparams_performance_plot_type2(cfg,logger,total_eval_dict,model_name_list,model_type_list,rcpp,SNR):
    save_folder = "../../test_results/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    chan_type = cfg.chan_type
    metric = cfg.performance_metric
    plot_save_name = f"{chan_type}_Mparams_{metric}_at_SNR{str(SNR).zfill(3)}_rcpp{str(rcpp).zfill(3)}_type2"
    if len(model_name_list)==0:
        return None
    
    for model_type in model_type_list:
        plot_save_name += "_" + model_type

    num_size = len(model_name_list)//len(model_type_list)
    th = 0
    m_type_index = 0

    #color_list = ['r-','b-','g-','c-','m-','y-','k-','w-']
    color_list = ['b-','r-','g-','c-','m-','b--','r--','g--','c--','m--']  
    
    plt.rcParams["figure.figsize"] = (14,8)
    
    fig, ax1 = plt.subplots()
    line_list = []
    
    valid_Mparams_list = []
    valid_performance_list = []    
    for i, model_name in enumerate(model_name_list):
        model_save_name = get_model_save_name(cfg,model_name,rcpp,SNR)
        eval_dict = total_eval_dict[model_save_name]            
            
        if eval_dict:
            Mparams = eval_dict['Mparams']
            valid_Mparams_list.append(Mparams)
            performance = eval_dict[metric]
            valid_performance_list.append(performance)
        
        th +=1            
        if th >= num_size:
            line = ax1.plot(valid_Mparams_list, valid_performance_list,color_list[m_type_index],label=f'{model_type_list[m_type_index]}',marker='o',linewidth=3,markersize=6)
            line_list.append(line)
            m_type_index +=1
            th = 0
            valid_Mparams_list = []
            valid_performance_list = []    


    lines = []
    for line in line_list:
        lines += line


    ax1.set_xlabel('Params (M)', fontsize=20)
    ax1.set_ylabel(metric, fontsize=20)
    plt.xticks( fontsize=14)
    plt.yticks(fontsize=14)

    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=10)
    plt.tight_layout(rect=[0,0,0.6,0.8])
    plt.title(f'{cfg.chan_type}, CPP=1/{rcpp}, SNR={SNR}dB', fontdict = {'fontsize' : 20})

    save_name = save_folder + plot_save_name + ".pdf"
    if save_name:
        plt.savefig(save_name)
    plt.clf()
    logger.info(f'{plot_save_name} is saved')


def save_GFlops_performance_plot_type2(cfg,logger,total_eval_dict,model_name_list,model_type_list,rcpp,SNR):
    save_folder = "../../test_results/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    chan_type = cfg.chan_type
    metric = cfg.performance_metric
    plot_save_name = f"{chan_type}_GFlops_{metric}_at_SNR{str(SNR).zfill(3)}_rcpp{str(rcpp).zfill(3)}_type2"
    if len(model_name_list)==0:
        return None
    
    for model_type in model_type_list:
        plot_save_name += "_" + model_type

    num_size = len(model_name_list)//len(model_type_list)
    th = 0
    m_type_index = 0

    #color_list = ['r-','b-','g-','c-','m-','y-','k-','w-']
    color_list = ['b-','r-','g-','c-','m-','b--','r--','g--','c--','m--']  
    
    plt.rcParams["figure.figsize"] = (14,8)
    
    fig, ax1 = plt.subplots()
    line_list = []
    
    valid_GFlops_list = []
    valid_performance_list = []    
    for i, model_name in enumerate(model_name_list):
        model_save_name = get_model_save_name(cfg,model_name,rcpp,SNR)
        eval_dict = total_eval_dict[model_save_name]            
            
        if eval_dict:
            GFlops = eval_dict['GFlops']
            valid_GFlops_list.append(GFlops)
            performance = eval_dict[metric]
            valid_performance_list.append(performance)
        
        th +=1            
        if th >= num_size:
            line = ax1.plot(valid_GFlops_list, valid_performance_list,color_list[m_type_index],label=f'{model_type_list[m_type_index]}',marker='o',linewidth=3,markersize=6)
            line_list.append(line)
            m_type_index +=1
            th = 0
            valid_GFlops_list = []
            valid_performance_list = []    
        
    lines = []
    for line in line_list:
        lines += line


    ax1.set_xlabel('GFlops', fontsize=20)
    ax1.set_ylabel(metric, fontsize=20)
    plt.xticks( fontsize=14)
    plt.yticks(fontsize=14)

    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=10)
    plt.tight_layout(rect=[0,0,0.6,0.8])
    plt.title(f'{cfg.chan_type}, CPP=1/{rcpp}, SNR={SNR}dB', fontdict = {'fontsize' : 20})

    save_name = save_folder + plot_save_name + ".pdf"
    if save_name:
        plt.savefig(save_name)
    plt.clf()
    logger.info(f'{plot_save_name} is saved')

def save_GFlops_performance_plot_type3(cfg,logger,total_eval_dict,model_name_list,model_type_list,rcpp,SNR):
    save_folder = "../../test_results/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    chan_type = cfg.chan_type
    metric = cfg.performance_metric
    plot_save_name = f"{chan_type}_GFlops_{metric}_at_SNR{str(SNR).zfill(3)}_rcpp{str(rcpp).zfill(3)}_type3"
    if len(model_name_list)==0:
        return None
    
    for model_type in model_type_list:
        plot_save_name += "_" + model_type

    num_size = len(model_name_list)//len(model_type_list)
    th = 0
    m_type_index = 0

    #color_list = ['r-','b-','g-','c-','m-','y-','k-','w-']
    color_list = ['b-','r-','g-','c-','m-','b--','r--','g--','c--','m--']  
    
    plt.rcParams["figure.figsize"] = (10,8)
    
    fig, ax1 = plt.subplots()
    line_list = []
    
    valid_GFlops_list = []
    valid_performance_list = []    
    for i, model_name in enumerate(model_name_list):
        model_save_name = get_model_save_name(cfg,model_name,rcpp,SNR)
        eval_dict = total_eval_dict[model_save_name]            
            
        if eval_dict:
            GFlops = eval_dict['GFlops']
            valid_GFlops_list.append(GFlops)
            performance = eval_dict[metric]
            valid_performance_list.append(performance)
        
        th +=1            
        if th >= num_size:
            line = ax1.plot(valid_GFlops_list, valid_performance_list,color_list[m_type_index],label=f'{model_type_list[m_type_index]}',marker='o',linewidth=3,markersize=6)
            line_list.append(line)
            m_type_index +=1
            th = 0
            valid_GFlops_list = []
            valid_performance_list = []    
        
    lines = []
    for line in line_list:
        lines += line


    ax1.set_xlabel('GFlops', fontsize=20)
    ax1.set_ylabel(metric, fontsize=20)
    plt.xticks( fontsize=14)
    plt.yticks(fontsize=14)

    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=10)
    plt.tight_layout(rect=[0,0,0.6,0.8])
    plt.title(f'{cfg.chan_type}, CPP=1/{rcpp}, SNR={SNR}dB', fontdict = {'fontsize' : 20})

    save_name = save_folder + plot_save_name + ".pdf"
    if save_name:
        plt.savefig(save_name)
    plt.clf()
    logger.info(f'{plot_save_name} is saved')
    
    
    
def save_GFlops_performance_ratio_plot(cfg,logger,total_eval_dict,encoder_side_list,both_side_list,decoder_side_list,fixed_model_list,encoder_type_list,both_type_list,decoder_type_list,rcpp,SNR):
    save_folder = "../../test_results/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    chan_type = cfg.chan_type
    metric = cfg.performance_metric
    plot_save_name = f"{chan_type}_GFlops_{metric}_ratio_at_SNR{str(SNR).zfill(3)}_rcpp{str(rcpp).zfill(3)}"
    if len(encoder_side_list)==0:
        return None
    
    for model_type in encoder_type_list:
        plot_save_name += "_" + model_type

    #color_list = ['r-','b-','g-','c-','m-','y-','k-','w-']
    color_list = ['b-','r-','g-','c-','m-','b--','r--','g--','c--','m--']  
    
    encoder_color_list = ['b^--','r^--']
    both_color_list = ['bd-.','rd-.']
    decoder_color_list = ['bx:','rx:']
    fixed_color_list = ['go-','co-','mo-']
    
    
    plt.rcParams["figure.figsize"] = (20,8)
    
    fig, ax1 = plt.subplots()
    line_list = []

    num_size = len(encoder_side_list)//len(encoder_type_list)
    th = 0
    m_type_index = 0    
    valid_GFlops_list = []
    valid_performance_list = []    
    for i, model_name in enumerate(encoder_side_list):
        model_save_name = get_model_save_name(cfg,model_name,rcpp,SNR)
        eval_dict = total_eval_dict[model_save_name]            
            
        if eval_dict:
            GFlops = eval_dict['GFlops']
            valid_GFlops_list.append(GFlops)
            performance = eval_dict[metric]
            valid_performance_list.append(performance)
        
        th +=1            
        if th >= num_size:
            line = ax1.plot(valid_GFlops_list, valid_performance_list,encoder_color_list[m_type_index],label=f'{encoder_type_list[m_type_index]}',linewidth=3,markersize=6)
            line_list.append(line)
            m_type_index +=1
            th = 0
            valid_GFlops_list = []
            valid_performance_list = []    


    num_size = len(both_side_list)//len(both_type_list)
    th = 0
    m_type_index = 0    
    valid_GFlops_list = []
    valid_performance_list = []    
    for i, model_name in enumerate(both_side_list):
        model_save_name = get_model_save_name(cfg,model_name,rcpp,SNR)
        eval_dict = total_eval_dict[model_save_name]            
            
        if eval_dict:
            GFlops = eval_dict['GFlops']
            valid_GFlops_list.append(GFlops)
            performance = eval_dict[metric]
            valid_performance_list.append(performance)
        
        th +=1            
        if th >= num_size:
            line = ax1.plot(valid_GFlops_list, valid_performance_list,both_color_list[m_type_index],label=f'{both_type_list[m_type_index]}',linewidth=3,markersize=6)
            line_list.append(line)
            m_type_index +=1
            th = 0
            valid_GFlops_list = []
            valid_performance_list = []    


    num_size = len(decoder_side_list)//len(decoder_type_list)
    th = 0
    m_type_index = 0    
    valid_GFlops_list = []
    valid_performance_list = []    
    for i, model_name in enumerate(decoder_side_list):
        model_save_name = get_model_save_name(cfg,model_name,rcpp,SNR)
        eval_dict = total_eval_dict[model_save_name]            
            
        if eval_dict:
            GFlops = eval_dict['GFlops']
            valid_GFlops_list.append(GFlops)
            performance = eval_dict[metric]
            valid_performance_list.append(performance)
        
        th +=1            
        if th >= num_size:
            line = ax1.plot(valid_GFlops_list, valid_performance_list,decoder_color_list[m_type_index],label=f'{decoder_type_list[m_type_index]}',linewidth=3,markersize=6)
            line_list.append(line)
            m_type_index +=1
            th = 0
            valid_GFlops_list = []
            valid_performance_list = []    


    m_type_index = 0    
    valid_GFlops_list = []
    valid_performance_list = []    
    for i, model_name in enumerate(fixed_model_list):
        model_save_name = get_model_save_name(cfg,model_name,rcpp,SNR)
        eval_dict = total_eval_dict[model_save_name]            
            
        if eval_dict:
            GFlops = eval_dict['GFlops']
            valid_GFlops_list.append(GFlops)
            performance = eval_dict[metric]
            valid_performance_list.append(performance)
                  
        line = ax1.plot(valid_GFlops_list, valid_performance_list,fixed_color_list[m_type_index],label=f'{fixed_model_list[m_type_index]}',linewidth=3,markersize=6)
        line_list.append(line)
        m_type_index +=1
        valid_GFlops_list = []
        valid_performance_list = []    

    lines = []
    for line in line_list:
        lines += line


    ax1.set_xlabel('GFlops', fontsize=20)
    ax1.set_ylabel(metric, fontsize=20)
    plt.xticks( fontsize=14)
    plt.yticks(fontsize=14)

    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.2, 1.0), fontsize=10)
    plt.tight_layout(rect=[0,0,0.6,0.8])
    plt.title(f'{cfg.chan_type}, CPP=1/{rcpp}, SNR={SNR}dB', fontdict = {'fontsize' : 20})
    ax1.grid(True)
    save_name = save_folder + plot_save_name + ".pdf"
    if save_name:
        plt.savefig(save_name)
    plt.clf()
    logger.info(f'{plot_save_name} is saved')    
    
    
    
    
    
    
def save_performance_GFlops_Mmemory_plot(cfg,logger,total_eval_dict,model_name_list,model_type_list,rcpp,SNR):
    save_folder = "../../test_results/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    chan_type = cfg.chan_type
    metric = cfg.performance_metric
    plot_save_name = f"{chan_type}_GFLOPs_Mmemory_{metric}_at_SNR{str(SNR).zfill(3)}_rcpp{str(rcpp).zfill(3)}"
    if len(model_name_list)==0:
        return None
    
    for model_type in model_type_list:
        plot_save_name += "_" + model_type

    num_size = len(model_name_list)//len(model_type_list)
    th = 0
    m_type_index = 0

    #color_list = ['r-','b-','g-','c-','m-','y-','k-','w-']
    color_list = ['bo-','ro-','go-','co-','mo-','bo--','ro--','go--','co--','mo--']
    
    plt.rcParams["figure.figsize"] = (20,8) #(14,8)
    
    fig, ax1 = plt.subplots()
    line_list = []
    
    valid_GFlops_list = []
    valid_performance_list = []    
    for i, model_name in enumerate(model_name_list):
        model_save_name = get_model_save_name(cfg,model_name,rcpp,SNR)
        eval_dict = total_eval_dict[model_save_name]            
            
        if eval_dict:
            GFlops = eval_dict['GFlops']
            valid_GFlops_list.append(GFlops)
            performance = eval_dict[metric]
            valid_performance_list.append(performance)
        
        th +=1            
        if th >= num_size:
            line = ax1.plot(valid_GFlops_list, valid_performance_list,color_list[m_type_index],label=f'{model_type_list[m_type_index]} Gflops',linewidth=3,markersize=6,zorder=i)
            line_list.append(line)
            m_type_index +=1
            th = 0
            valid_GFlops_list = []
            valid_performance_list = []    
        

    ax1.grid(True)
    ax1.set_xlabel('GFLOPs', fontsize=20)
    ax1.set_ylabel(metric, fontsize=20)
    plt.xticks( fontsize=14)
    plt.yticks(fontsize=14)
    
    
    
    num_size = len(model_name_list)//len(model_type_list)
    th = 0
    m_type_index = 0

    #color_list = ['r-','b-','g-','c-','m-','y-','k-','w-']
    color_list = ['bd--','rd--','gd--','cd--','md--']
    
    
    ax2 = ax1.twiny() # ax1.twiny():use same y axis, ax1.twinx():use same x axis

    
    valid_Mmemory_list = []
    valid_performance_list = []    
    for i, model_name in enumerate(model_name_list):
        model_save_name = get_model_save_name(cfg,model_name,rcpp,SNR)
        eval_dict = total_eval_dict[model_save_name]            
            
        if eval_dict:
            Mmemory = eval_dict['Mmemory']
            valid_Mmemory_list.append(Mmemory)
            performance = eval_dict[metric]
            valid_performance_list.append(performance)
        
        th +=1            
        if th >= num_size:
            line = ax2.plot(valid_Mmemory_list, valid_performance_list,color_list[m_type_index],label=f'{model_type_list[m_type_index]} Memory',linewidth=4,markersize=6,zorder=i+num_size)
            line_list.append(line)
            m_type_index +=1
            th = 0
            valid_Mmemory_list = []
            valid_performance_list = []    

    lines = []
    for line in line_list:
        lines += line


    ax2.set_xlabel('Memory (MB)', fontsize=20)
    plt.xticks( fontsize=14)
    plt.yticks(fontsize=14)
    #ax2.set_xticks(np.linspace(ax2.get_xticks()[1],ax2.get_xticks()[-2],len(ax1.get_xticks())-2))

    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.2, 1.0), fontsize=10)
    plt.tight_layout(rect=[0,0,0.6,0.8])
    plt.title(f'{cfg.chan_type}, CPP=1/{rcpp}, SNR={SNR}dB\n', fontdict = {'fontsize' : 20})

    
    save_name = save_folder + plot_save_name + ".pdf"
    if save_name:
        plt.savefig(save_name)
    plt.clf()
    logger.info(f'{plot_save_name} is saved')

    
    
    

def save_SNR_performance_plot(cfg,logger,total_eval_dict,model_name_list,rcpp,SNR_list):
    save_folder = "../../test_results/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    chan_type = cfg.chan_type
    metric = cfg.performance_metric
    plot_save_name = f"{chan_type}_SNR_{metric}_at_rcpp{str(rcpp).zfill(3)}"
    if len(model_name_list)==0:
        return None
    
    for model_name in model_name_list:
        plot_save_name += "_" + model_name


    #color_list = ['r-','b-','g-','c-','m-','y-','k-','w-']
    color_list = ['b-','r-','g-','c-','m-','b--','r--','g--','c--','m--']
    
    plt.rcParams["figure.figsize"] = (14,5) #(14,8)
    
    fig, ax1 = plt.subplots()
    line_list = []
    
    
    for i, model_name in enumerate(model_name_list):
        valid_SNR_list = []
        valid_performance_list = []
        for SNR in SNR_list:
            model_save_name = get_model_save_name(cfg,model_name,rcpp,SNR)
            eval_dict = total_eval_dict[model_save_name]            
            
            if eval_dict:
                valid_SNR_list.append(SNR)
                performance = eval_dict[metric]
                valid_performance_list.append(performance)
        line = ax1.plot(valid_SNR_list, valid_performance_list,color_list[i],label=f'{cfg.model_name}',marker='o',linewidth=3,markersize=6)
        line_list.append(line)   
        
    lines = []
    for line in line_list:
        lines += line


    ax1.set_xlabel('SNR (dB)', fontsize=20)
    ax1.set_ylabel(metric, fontsize=20)
    plt.xticks( fontsize=14)
    plt.yticks(fontsize=14)

    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.2, 1.0), fontsize=10)
    plt.tight_layout(rect=[0,0,0.6,0.8])
    plt.title(f'{cfg.chan_type}, CPP=1/{rcpp}', fontdict = {'fontsize' : 20})

    save_name = save_folder + plot_save_name + ".pdf"
    if save_name:
        plt.savefig(save_name)
    plt.clf()
    logger.info(f'{plot_save_name} is saved')

def save_rcpp_performance_plot(cfg,logger,total_eval_dict,model_name_list,rcpp_list,SNR):
    save_folder = "../../test_results/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    chan_type = cfg.chan_type
    metric = cfg.performance_metric
    plot_save_name = f"{chan_type}_rcpp_{metric}_at_SNR{str(SNR).zfill(3)}"
    if len(model_name_list)==0:
        return None
    
    for model_name in model_name_list:
        plot_save_name += "_" + model_name


    #color_list = ['r-','b-','g-','c-','m-','y-','k-','w-']
    color_list = ['b-','r-','g-','c-','m-','b--','r--','g--','c--','m--']
    
    plt.rcParams["figure.figsize"] = (10,8) #(14,8)
    
    fig, ax1 = plt.subplots()
    line_list = []
    
    for i, model_name in enumerate(model_name_list):
        valid_rcpp_list = []
        valid_performance_list = []
        for rcpp in rcpp_list:
            model_save_name = get_model_save_name(cfg,model_name,rcpp,SNR)
            eval_dict = total_eval_dict[model_save_name]
            if eval_dict:
                valid_rcpp_list.append(rcpp)
                performance = eval_dict[metric]
                valid_performance_list.append(performance)
        valid_cpp_list = 1/np.array(valid_rcpp_list)        
        line = ax1.plot(valid_cpp_list, valid_performance_list,color_list[i],label=f'{cfg.model_name}',marker='o',linewidth=3,markersize=6)
        line_list.append(line)   
        
    lines = []
    for line in line_list:
        lines += line


    ax1.set_xlabel('cpp', fontsize=20)
    ax1.set_ylabel(metric, fontsize=20)
    plt.xticks( fontsize=14)
    plt.yticks(fontsize=14)

    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=10)
    plt.tight_layout(rect=[0,0,0.6,0.8])
    #plt.xlabel(r'$test \frac{{1}}{{{}}}$'.format(g))
    #r'$\mathregular{T_{s1}}$'.replace('s1', 'toto')
    #g = 3
    #plt.title(f'{cfg.chan_type}, SNR: {SNR}', r'$\frac{{5}}{{{}}}$'.format(g))
    #plt.title(f'{cfg.chan_type}, SNR: {SNR}', r'$\frac{5}{a}$'.replace('a', 'g'))
    plt.title(f'{cfg.chan_type}, SNR={SNR}dB', fontdict = {'fontsize' : 20})

    save_name = save_folder + plot_save_name + ".pdf"
    if save_name:
        plt.savefig(save_name)
    plt.clf()
    logger.info(f'{plot_save_name} is saved')




def save_SNR_performance_table(cfg,logger,total_eval_dict,model_name_list,rcpp,SNR_list):
    save_folder = "../../test_results/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    chan_type = cfg.chan_type
    metric = cfg.performance_metric
    table_save_name = f"{chan_type}_SNR_{metric}_at_rcpp{str(rcpp).zfill(3)}"
    if len(model_name_list)==0:
        return None
    
    for model_name in model_name_list:
        table_save_name += "_" + model_name

    save_name = save_folder + table_save_name + ".csv"
    
    
    first_line = ["rcpp",rcpp,"metric",metric]
    second_line = ["SNR"]
    second_line.extend(SNR_list)
    second_line.append('GFlops')
    second_line.append('Mmemory')
    
    
    with open(save_name,'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(first_line)
        writer.writerow(second_line)
    
        for i, model_name in enumerate(model_name_list):
            valid_performance_list = [model_name]
            for SNR in SNR_list:
                model_save_name = get_model_save_name(cfg,model_name,rcpp,SNR)
                eval_dict = total_eval_dict[model_save_name]            
            
                if eval_dict:
                    performance = eval_dict[metric]
                    GFlops = eval_dict['GFlops']
                    Mmemory = eval_dict['Mmemory']
                    valid_performance_list.append(performance)
                else:
                    valid_performance_list.append("None")
            valid_performance_list.append(GFlops)
            valid_performance_list.append(Mmemory)
            writer.writerow(valid_performance_list)
                    
    f.close()  
        
    logger.info(f'{table_save_name} is saved')

def save_rcpp_performance_table(cfg,logger,total_eval_dict,model_name_list,rcpp_list,SNR):
    save_folder = "../../test_results/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    chan_type = cfg.chan_type
    metric = cfg.performance_metric
    table_save_name = f"{chan_type}_rcpp_{metric}_at_SNR{str(SNR).zfill(3)}"
    if len(model_name_list)==0:
        return None
    
    for model_name in model_name_list:
        table_save_name += "_" + model_name

    save_name = save_folder + table_save_name + ".csv"
    
    
    first_line = ["SNR",SNR,"metric",metric]
    second_line = ["rcpp"]
    second_line.extend(rcpp_list)
    second_line.append('GFlops')
    second_line.append('Mmemory')
    
    
    with open(save_name,'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(first_line)
        writer.writerow(second_line)
    
        for i, model_name in enumerate(model_name_list):
            valid_performance_list = [model_name]
            for rcpp in rcpp_list:
                model_save_name = get_model_save_name(cfg,model_name,rcpp,SNR)
                eval_dict = total_eval_dict[model_save_name]            
            
                if eval_dict:
                    performance = eval_dict[metric]
                    GFlops = eval_dict['GFlops']
                    Mmemory = eval_dict['Mmemory']
                    valid_performance_list.append(performance)
                else:
                    valid_performance_list.append("None")
            valid_performance_list.append(GFlops)
            valid_performance_list.append(Mmemory)
            writer.writerow(valid_performance_list)
                    
    f.close()  
        
    logger.info(f'{table_save_name} is saved')

