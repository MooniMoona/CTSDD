import pickle
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch


class Wind_Dataset(Dataset):
    def __init__(self, mean_std_path, eval_length=36, sample_interval=300, mode="train", validindex=0, gt_folder="", self_mask_folder=""):
        self.eval_length = eval_length
        self.sample_interval=sample_interval

        with open(mean_std_path, "rb") as f:
            self.train_mean, self.train_std = pickle.load(f)  
        
        df = pd.read_csv(
            gt_folder,
            index_col="Time",
            parse_dates=True, 
        )   
        df_gt = pd.read_csv(
            self_mask_folder,
            index_col="Time",
            parse_dates=True, 
        ) 
    
        month_list = df.index.month.unique() 
        self.month_list = month_list

        # create data for batch
        self.observed_data = []  # values (separated into each month)
        self.observed_mask = []  # masks (separated into each month)
        self.gt_mask = []  # ground-truth masks (separated into each month)
        self.index_month = []  # indicate month
        self.position_in_month = []  # indicate the start position in month (length is the same as index_month)
        self.use_index = []  # to separate train/valid/test
        self.cut_length = []  # excluded from evaluation targets

        for i in range(len(month_list)):
            current_df = df[df.index.month == month_list[i]] 
            current_df_gt = df_gt[df_gt.index.month == month_list[i]] 
            current_length = len(current_df) - eval_length + 1  

            last_index = len(self.index_month) # 0, 637, 

            position_current_month = []
            if mode == "train":
                for idx in range(current_length):
                    if not current_df.iloc[idx:idx+eval_length].isnull().any().all():
                        position_current_month.append(idx)
            else:
                position_current_month+= np.arange(current_length).tolist()
                
            self.position_in_month += position_current_month  
            self.index_month += np.array([i] * len(position_current_month)).tolist() 
    
            # mask values for observed indices are 1
            c_mask = 1 - current_df.isnull().values  
            c_gt_mask = 1 - current_df_gt.isnull().values   
            c_data = (
                (current_df.fillna(0).values - self.train_mean) / self.train_std
            ) * c_mask  

            self.observed_mask.append(c_mask) # len = 7 
            self.gt_mask.append(c_gt_mask)
            self.observed_data.append(c_data)
     
        if mode != "train":
            all_sample_number = np.arange(len(self.index_month))
            n_sample = len(all_sample_number[::self.sample_interval])
            
            print("n_sample: ", n_sample)
            c_index = []
            for idx in range(n_sample):
                c_index.append(idx*self.sample_interval)  
                                    
            self.use_index += c_index
            self.cut_length += [0] * len(c_index) 

            if (len(self.index_month)-1) % self.sample_interval != 0 :
                print("Add a sample, n_sample: ", n_sample+1)
                self.use_index += [len(self.index_month) - 1] 
                self.cut_length += [self.sample_interval - len(self.index_month) % self.sample_interval] 

        else:
            self.use_index = np.arange(len(self.index_month))
            self.cut_length = [0] * len(self.use_index) 

        self.index_month_histmask = self.index_month
        self.position_in_month_histmask = self.position_in_month


    def __getitem__(self, org_index):
        index = self.use_index[org_index] 
        c_month = self.index_month[index] 
        c_index = self.position_in_month[index]
        hist_month = self.index_month_histmask[index]
        hist_index = self.position_in_month_histmask[index]
        s = {
            "observed_data": self.observed_data[c_month][
                c_index : c_index + self.eval_length
            ],
            "observed_mask": self.observed_mask[c_month][
                c_index : c_index + self.eval_length
            ],
            "gt_mask": self.gt_mask[c_month][
                c_index : c_index + self.eval_length
            ],  # 7*
            "hist_mask": self.observed_mask[hist_month][
                hist_index : hist_index + self.eval_length
            ], 
            "timepoints": np.arange(self.eval_length),
            "cut_length": self.cut_length[org_index],
        } 

        return s

    def __len__(self):
        return len(self.use_index)


def get_dataloader(mean_std_path, eval_length, batch_size, sample_interval, device, gt_folder, self_mask_folder, 
                   gt_test_folder, self_mask_test_folder, validindex=0):
    dataset = Wind_Dataset(mean_std_path=mean_std_path, eval_length=eval_length, sample_interval=sample_interval, mode="train", 
                           validindex=validindex, gt_folder=gt_folder, self_mask_folder=self_mask_folder) # 字典种的6种数据
    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=1, shuffle=True)
    
    dataset_valid = Wind_Dataset(mean_std_path=mean_std_path, eval_length=eval_length, sample_interval=sample_interval, mode="valid", 
                                 validindex=validindex, gt_folder=gt_test_folder, self_mask_folder=self_mask_test_folder)
    valid_loader = DataLoader(dataset_valid, batch_size=batch_size, num_workers=1, shuffle=False)
    
    dataset_test = Wind_Dataset(mean_std_path=mean_std_path, eval_length=eval_length, sample_interval=sample_interval, mode="test", 
                                validindex=validindex, gt_folder=gt_test_folder, self_mask_folder=self_mask_test_folder)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, num_workers=1, shuffle=False)

    scaler = torch.from_numpy(dataset.train_std).to(device).float()
    mean_scaler = torch.from_numpy(dataset.train_mean).to(device).float()

    return train_loader, valid_loader, test_loader, scaler, mean_scaler


def get_dataloader_case(mean_std_path, eval_length, batch_size, sample_interval, device,gt_test_folder, self_mask_test_folder, validindex=0):
    
    dataset_test = Wind_Dataset(mean_std_path=mean_std_path, eval_length=eval_length, sample_interval=sample_interval, mode="test", 
                                validindex=validindex, gt_folder=gt_test_folder, self_mask_folder=self_mask_test_folder)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, num_workers=1, shuffle=False)

    scaler = torch.from_numpy(dataset_test.train_std).to(device).float()
    mean_scaler = torch.from_numpy(dataset_test.train_mean).to(device).float()
    return test_loader, scaler, mean_scaler