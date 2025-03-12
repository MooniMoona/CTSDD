import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import torch
import pickle
import os

def loss_train_valid_plot_save(loss_train, loss_valid, foldername):
    e = range(len(loss_train))
    # 绘制折线图
    plt.figure(figsize=(10, 6))  # 设置图像大小
    plt.plot(e, loss_train, label='Train Loss', color='blue', marker='o')
    plt.plot(e, loss_valid, label='Valid Loss', color='red', marker='x')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Valid Loss Over Epochs')
    plt.legend()

    # 保存图像
    plt.savefig(foldername+'/loss_train_valid.png', dpi=300)  # 保存图像，dpi设置图像分辨率
    plt.show()
    
def get_quantile(samples,q,dim=1):  # torch.quantile() 函数用于计算输入张量在指定分位数（quantile）处的值 如：50%分位数（中位数）
    # 每个例子，多个扩散步的预测值，不同分数位的值
    # return torch.quantile(samples,q,dim=dim).cpu().numpy()
    return torch.quantile(samples,q,dim=dim).numpy()

def get_results_quantile(nsample, foldername, mean_std_path):
    results_path = foldername + 'generated_outputs_nsample'+ str(nsample) +'.pk'
    with open(results_path, 'rb') as f:
        samples,all_target,all_evalpoint,all_observed,all_observed_time,scaler,mean_scaler = pickle.load(f)
    all_target_np = all_target.cpu().numpy() #原始带缺失值的真实数据 
    all_evalpoint_np = all_evalpoint.cpu().numpy()  #自己建立的遮挡位置mask
    all_observed_np = all_observed.cpu().numpy()  # 本身nan的位置
    all_given_np = all_observed_np - all_evalpoint_np # 所有遮挡位置observed_mask-target_mask（本身未知位置+自己遮挡位置）1-有值；0-遮挡
    # samples [774,50,300,27] ; all_target,all_evalpoint,all_observed[774,300,27]
    K = samples.shape[-1] #feature
    L = samples.shape[-2] #time length
    print("K,L: ", K, L)

    # 加载mean与std，反标准化
    with open(mean_std_path, 'rb') as f:
        train_mean,train_std = pickle.load(f)
    # train_std_cuda = torch.from_numpy(train_std).cuda()
    # train_mean_cuda = torch.from_numpy(train_mean).cuda()
    # 修改
    samples = samples.cpu()
    train_std_cuda = torch.from_numpy(train_std)
    train_mean_cuda = torch.from_numpy(train_mean)
    all_target_np=(all_target_np*train_std+train_mean)
    samples=(samples*train_std_cuda+train_mean_cuda)
    
    qlist =[0.05,0.25,0.5,0.75,0.95]
    quantiles_imp= []
    # 计算所有遮挡之外能看到值不同分数位的真实值。dim=1：nsample次恢复实验，在多次试验中求不同分数位的值
    for q in qlist: 
        quantiles_imp.append(get_quantile(samples, q, dim=1)*(1-all_given_np) + all_target_np * all_given_np)
        
    quantiles_pred= []
    # 计算所有预测值不同分数位的真实值 dim=1：nsample次恢复实验的
    for q in qlist: 
        quantiles_pred.append(get_quantile(samples, q, dim=1))  #5*[774,300,27]
        
    print(all_target_np.shape)
    print(all_evalpoint_np.shape)

    return all_target_np, all_evalpoint_np, all_given_np, quantiles_pred
  
    
def wp_quantiles_plot(wp_target_np, wp_evalpoint_np, wp_given_np, wp_quantiles_np, \
                         dataind,select_case_name,save_foldername,save=1):
    # choose data
    wp_target, wp_evalpoint, wp_given = \
        wp_target_np[dataind], wp_evalpoint_np[dataind], wp_given_np[dataind]
    wp_quantiles = [wp[dataind] for wp in wp_quantiles_np]
   
    L = wp_target.shape[0]
    
    plt.rcParams["font.size"] = 14
    plt.figure(dpi=600,figsize=[10,3.0])
    
    # val: actual data；y: actual data with mask；
    df = pd.DataFrame({"x":np.arange(0,L), "val":wp_target, "y":wp_evalpoint})
    df = df[df.y != 0] # mask
    df2 = pd.DataFrame({"x":np.arange(0,L), "val":wp_target, "y":wp_given})
    df2 = df2[df2.y != 0] # all miss       

    color_name = '#df3881'

    plt.plot(range(0,L), wp_quantiles[2] ,label="CTSDD",alpha=1.0, 
                color=color_name, linewidth=1.1) 
    plt.fill_between(range(0,L), wp_quantiles[0],wp_quantiles[4],
                    alpha=0.3, color=color_name) # restoration 5-95% 
        
    plt.plot(np.arange(0,L),wp_target, '--', color = 'g', label='actual data', linewidth=1.3) # 
    plt.plot(df.x,df.val, color = 'black',marker = 'o', markersize = 3, linestyle='None', 
             label='masked data') 
    
    plt.xlim(0, L-1)  # 设置横坐标范围为 [0, 12]
    
    # center WP data
    # if select_mask_name == 'mid':
    start = (L-30)//2  
    end = start + 30    
    plt.axvspan(start, end, color='green', alpha=0.1)
        

    plt.tick_params(labelsize=11)
    plt.legend(ncol=1, loc="lower right", fontsize=12)

    plt.grid(True)
    plt.title(select_case_name+'_results', y=-0.3, fontsize=15)
    plt.xlabel('Time (1min)', fontsize=12)
    plt.ylabel('Wind Power', fontsize=12)
    if save==1:  
        save_path = save_foldername+'pictures/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_plot_path = save_path+select_case_name+\
            '_wp_quantiles_results'+str(dataind) +'.png'
        plt.savefig(save_plot_path, dpi=600, bbox_inches='tight')
    plt.show()  
    
   
def plot_case_scores(model_name, select_case_name, scores_name, save_foldername,
                        threshold_path, fault_time, threshold_name, detect_interval):
    
    score_path = save_foldername + 'scores.csv'
    df = pd.read_csv(score_path, index_col=0)
    scores = df[scores_name]  # combined_mae_dtw_scores
    scores = scores.iloc[::detect_interval]
    
    # threshold_path = './dataset/normal_Robust_threshold.csv'
    threshold_df = pd.read_csv(threshold_path, index_col=0)
    threshold =threshold_df[threshold_name].iloc[0]
    
    plt.figure(dpi=600,figsize=(7, 3))
    
    plt.plot(scores, label=model_name, linewidth=1.5, color='blue')
    # fault time score point
    if fault_time != None:
        fault_score = scores[fault_time]
        plt.plot(fault_time, fault_score, 'o', color='black')
    # threshold
    plt.axhline(y=threshold, linestyle='--', linewidth=1.5, color='red')
    
    if (scores[scores >= threshold]).any():
        warn_time = scores[scores >= threshold].index[0]
        plt.scatter(warn_time-1, threshold, 
                            color='blue',marker = '^', s=150)  # 标记点
        warn_time = scores[scores >= threshold].index.tolist()
        threshold_all = [threshold]*len(warn_time)
        plt.scatter(warn_time, threshold_all,
                        color='blue',marker = '^', s=15)  # 预警区间
    min = scores.min().min()
    if fault_time != None:
        plt.axvline(x=fault_time, color='black', linestyle='--', linewidth=2)
        # plt.text(fault_time-90, min-0.3, f'Fault time', fontsize=12)
    plt.tick_params(labelsize=11)
    plt.legend(loc="upper left", fontsize=12)

    plt.xlabel('Time (1min)', fontsize=11)
    plt.ylabel('Composite Score', fontsize=12)
    plt.xticks(range(0, len(scores)*detect_interval, 100))
    plt.xlim([0, len(scores)*detect_interval])
    plt.title(select_case_name+'_Scores', fontsize=13)
    
    plt.grid(True)

    output_image_path = save_foldername + 'scores_interval'+ str(detect_interval)+'.png'
    plt.savefig(output_image_path)
    plt.show()
    plt.close() 
    
    
    
    