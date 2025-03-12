#%%
import argparse
import torch
import json
import yaml
import os
import torch.nn as nn
import pandas as pd
import numpy as np 
from dataset_wind_ice import get_dataloader, get_dataloader_case
from main_diff_model import DIFF_wind
# from utils import train
from utils_prediction import evaluate


parser = argparse.ArgumentParser(description="CTSDD")
# Configuration file hyperparameters
parser.add_argument("--config", type=str, default="WT_ice")
# normalization parameter
parser.add_argument("--mean_std_path", type=str, default="./dataset/data_mean_std.pk")
parser.add_argument("--threshold_path", type=str, default="./dataset/normal_Robust_threshold.csv")

# only test
parser.add_argument('--test_only', type=int, default=1)
parser.add_argument("--test_sample_interval", type=int, default=1, help="sample interval for test")

# model & pretrained weigths
parser.add_argument("--model", type=str, default="CTSDD")
parser.add_argument("--pretrained_model", type=str, default="./models/CTSDD_pretrained_model.pth")

parser.add_argument('--device', default='cuda', help='Device for Attack')
parser.add_argument('--num_works', type=int, default=3,help='num_works: 0, 1, 2, 3')
parser.add_argument('--batch_size', type=int, default=300, help='batch_size')
parser.add_argument('--nsample', type=int, default=5, help='Number of runs per sample.')

# select case
parser.add_argument('--case_name', default='normal', help='Select case: normal, case1, case2')

args =parser.parse_known_args()[0]
print(args)

#%%
# Configuration file hyperparameters
path = "./config/" + args.config + ".yaml"
with open(path, "r") as f:
    config = yaml.safe_load(f)
    
# Change config file based on customized values
config["train"]["batch_size"] = args.batch_size
config["diffusion"]["nsample"] = args.nsample
#%%
# Loading Models
model = DIFF_wind(config, args.device, args.model, target_dim=config["model"]["input_size"]).to(args.device)

if args.device == 'cuda' and args.num_works != 0:
    works_list = list(range(args.num_works+1))
    print(works_list)
    model = torch.nn.DataParallel(model,device_ids=works_list)

model.load_state_dict(torch.load(args.pretrained_model))

model = model.module 
model = model.to(args.device)
#%%
# Composite score-related parameters and warning thresholds
if args.threshold_path is None:
    threshold_sign = 0
else:    
    threshold_sign = 1
    threshold_data = pd.read_csv(args.threshold_path)
    mae_median = threshold_data['mae_median'].iloc[0]
    mae_iqr = threshold_data['mae_iqr'].iloc[0]
    dtw_median = threshold_data['dtw_median'].iloc[0]
    dtw_iqr = threshold_data['dtw_iqr'].iloc[0]
    mae_dtw_threshold = threshold_data['mae_dtw_threshold'].iloc[0]

# load data
random_mask_percent = config["model"]["random_mask_percent"]
config["files"]["gt_test_folder"] = './dataset/'+args.case_name+'/'+args.case_name+'.csv'
config["files"]["self_mask_test_folder"] = './dataset/'+args.case_name+'/'+args.case_name+\
                                            '_mask'+str(random_mask_percent)+'.csv'
test_loader, scaler, mean_scaler = get_dataloader_case(args.mean_std_path, config["model"]["eval_length"],
    config["train"]["batch_size"], args.test_sample_interval, device=args.device, 
    gt_test_folder=config["files"]["gt_test_folder"], self_mask_test_folder=config["files"]["self_mask_test_folder"])    

# save path
foldername = ("./save/test_results/"+args.case_name+"_test/")
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

# detect
evaluate(
    model,
    args.device,
    test_loader,
    mae_median, mae_iqr, dtw_median, dtw_iqr, mae_dtw_threshold,
    args.test_sample_interval,
    nsample=config["diffusion"]["nsample"],
    scaler=scaler,
    mean_scaler=mean_scaler,
    foldername=foldername,
    threshold_sign = threshold_sign
)  


