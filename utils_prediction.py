import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import pandas as pd
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from scipy.interpolate import interp1d
import time


def evaluate(model, device, test_loader, mae_median, mae_iqr, dtw_median, dtw_iqr,mae_dtw_threshold,
             sample_interval=300, nsample=5, scaler=1, mean_scaler=0, foldername="", threshold_sign = 0):
             
    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []
        all_samples_median = []
        
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample)

                samples, c_target, eval_points, observed_points, observed_time = output

                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                samples_median = samples.median(dim=1) #2*[B, K, L]
                all_samples_median.append(samples_median.values) # [B, K, L]
                
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)
                mse_current = (
                    ((samples_median.values - c_target) * observed_points) ** 2
                ) * (scaler ** 2)
                mae_current = (
                    torch.abs((samples_median.values - c_target) * observed_points) 
                ) * scaler

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                evalpoints_total += observed_points.sum().item()
 
                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )
                
            all_target = torch.cat(all_target, dim=0)
            all_evalpoint = torch.cat(all_evalpoint, dim=0)
            all_observed_point = torch.cat(all_observed_point, dim=0)
            all_observed_time = torch.cat(all_observed_time, dim=0)
            all_generated_samples = torch.cat(all_generated_samples, dim=0)  
            all_samples_median = torch.cat(all_samples_median, dim=0)  

            with open(
                foldername + "/generated_outputs_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                pickle.dump(
                    [
                        all_generated_samples,
                        all_target,
                        all_evalpoint,
                        all_observed_point,
                        all_observed_time,
                        scaler,
                        mean_scaler,
                    ],
                    f,
                )
                
            if threshold_sign == 0:
                print('Unable to calculate composite scores with determination of warnings!')
                print('Threshold-related parameter file path required! (args.threshold_path)')
            else:
                WP_scores_pred(all_samples_median.cpu(), all_target.cpu(), mae_median, mae_iqr, dtw_median, dtw_iqr, 
                                mae_dtw_threshold, foldername)  
                
                
def linear_interpolation(array):
    for i in range(array.shape[0]):
        nan_mask = np.isnan(array[i])
        if np.any(nan_mask):
            not_nan_indices = np.where(~nan_mask)[0]
            not_nan_values = array[i, not_nan_indices]
            interp_func = interp1d(not_nan_indices, not_nan_values, kind='linear', bounds_error=False, fill_value="extrapolate")
            array[i, nan_mask] = interp_func(np.where(nan_mask)[0])
    return array


def WP_scores_pred(samples_median, c_target, mae_median, mae_iqr, dtw_median, dtw_iqr, 
                            mae_dtw_threshold, path):
    wp_num = 2
    WP_true = np.array(c_target[:, :, wp_num])
    WP_true = linear_interpolation(WP_true)
    WP_pred = np.array(samples_median[:, :, wp_num])
 
    rmse = np.sqrt(np.mean((WP_true - WP_pred) ** 2, axis=1))
    mae = np.mean(np.abs(WP_true - WP_pred), axis=1)
    dtw = np.array([fastdtw(WP_true[i], WP_pred[i], dist=euclidean)[0] for i in range(WP_true.shape[0])])

    ss_res = np.sum((WP_true - WP_pred) ** 2, axis=1)
    ss_tot = np.sum((WP_true - np.mean(WP_true, axis=1, keepdims=True)) ** 2, axis=1)
    r2 = 1 - (ss_res / ss_tot)
        
    scaled_test_mae = (mae - mae_median) / mae_iqr
    scaled_test_dtw = (dtw - dtw_median) / dtw_iqr

    a1 = 0.5
    a2 = 0.5
    combined_mae_dtw_scores = a1 * scaled_test_mae + a2 * scaled_test_dtw
    
    mae_dtw_labels = np.zeros_like(combined_mae_dtw_scores)
    
    for i, score in enumerate(combined_mae_dtw_scores):
        if score > mae_dtw_threshold:
            mae_dtw_labels[i] = 1
            # warning
        else:
            mae_dtw_labels[i] = 0
            # normal
    # print("combined_mae_dtw_scores: ", combined_mae_dtw_scores)
    # print("mae_dtw_labels: ", mae_dtw_labels)
    
    # data = pd.DataFrame({'combined_mae_dtw_scores': combined_mae_dtw_scores, 'mae_dtw_labels': mae_dtw_labels})
    # data.to_csv(path+'/scores_labels.csv')
    
    data = pd.DataFrame({'combined_mae_dtw_scores': combined_mae_dtw_scores})
    data.to_csv(path+'/scores.csv')
