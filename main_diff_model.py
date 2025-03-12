import numpy as np
import torch
import torch.nn as nn
from models.CTSDD import diff_enc, diff_dec

class DIFF_base(nn.Module):
    def __init__(self, target_dim, config, device, model_name):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.target_dim = target_dim  
        self.channels = config["diffusion"]["channels"]
        self.pred_length = config["model"]["pred_length"]
        self.obs_mask_rank = config["model"]["obs_mask_rank"]

        self.emb_time_dim = self.channels
        self.emb_feature_dim = self.channels
        self.is_unconditional = config["model"]["is_unconditional"] # False
        self.target_strategy = config["model"]["target_strategy"] # mix
        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim 
        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask
    
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim)  
        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim   # 145
        self.input_size = config["model"]["input_size"]

        self.diffmodel_enc = diff_enc(self.input_size, config_diff, self.device, inputdim=1)
        self.diffmodel_dec = diff_dec(self.input_size, config_diff, self.device, inputdim=1)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad": # Quadratic Schedule
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_randmask(self, observed_mask):   # observed_mask-[16, 36, 36] 
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask  # [16, 36, 36] 
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1) # [16, 1296] 
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()  # missing ratio  0.11
            sample_ratio = sample_ratio * self.obs_mask_rank
            num_observed = observed_mask[i].sum().item()  # 1153
            num_masked = round(num_observed * sample_ratio)  # 1153*0.11=127
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1  # 
     
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()   
        return cond_mask # [16, 36, 36] 
 
    def get_mid_wp_mask(self, observed_mask, is_train):  
        cond_mask = observed_mask
        if is_train == 1:
            cond_mask = self.get_randmask(observed_mask)
        wp_num = 2
        L = observed_mask.shape[2]
     
        center_start = (L - self.pred_length) // 2
  
        if (L - self.pred_length) % 2 != 0:
            center_start += 1
        pred_mask = torch.ones_like(observed_mask) #B, K, L
 
        pred_mask[:, wp_num, center_start:center_start+self.pred_length] = 0
        cond_mask = cond_mask * pred_mask 
        
        return cond_mask  # [16, 36, 36]
       
    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape # [16, 36, 36]
        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)  # [16, 36, 64]
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)  # [16, 36, 40, 64] 
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K,emb) (40, 64)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)   # [16, 144, 40, 36]

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)  # [16, 1, 40, 36]
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info  # [16, 145, 40, 36] 145=128+36+1(time_emb+feature_emb+cond_mask)
    
    def calc_residual(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1
    ):
        B, K, L = observed_data.shape  #

        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)  #
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device) # 
        current_alpha = self.alpha_torch[t].to(self.device)  #
        noise = torch.randn_like(observed_data) # x_0
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise 

        total_input = noisy_data.unsqueeze(1) #(B,1, K,L) 

        enc_input = (cond_mask * observed_data).unsqueeze(1)
        enc_output = self.diffmodel_enc(enc_input, side_info)
        predicted = self.diffmodel_dec(total_input, side_info, enc_output, t)  # (B,K,L)
        target_mask = observed_mask - cond_mask  
        residual_cond = (noise - predicted) * cond_mask 
        residual_target = (noise - predicted) * target_mask 
        return residual_cond, residual_target
    
    def calc_residual_valid(
        self, observed_data, cond_mask, observed_mask, side_info, is_train
    ):
        residual_cond_list, residual_target_list = [], []
        for t in range(self.num_steps):  # calculate loss for all t
            residual_cond, residual_target = self.calc_residual(observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t)
            residual_cond_list.append(residual_cond)
            residual_target_list.append(residual_target)
        residual_cond_all = torch.stack(residual_cond_list, dim=0)
        residual_target_all = torch.stack(residual_target_list, dim=0)
        return residual_cond_all, residual_target_all

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask): # input
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
        else:
            cond_obs = (cond_mask * observed_data).unsqueeze(1) 
            noisy_target = noisy_data.unsqueeze(1) 
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input # [16, 2, 40, 36]

    def impute(self, observed_data, cond_mask, side_info, n_samples):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):  # n_samples
            # generate noisy observation for unconditional model 
            if self.is_unconditional == True:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data) 
            # encoder
            enc_input = observed_data.unsqueeze(1)  # (B,1,K,L)
            enc_output = self.diffmodel_enc(enc_input, side_info)

            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional == True:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                else: 
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)
                    noisy_target = current_sample.unsqueeze(1) # [64,1, 40, 36]

                    diff_input = noisy_target #(B,1, K,L) 
                predicted = self.diffmodel_dec(diff_input, side_info, enc_output, torch.tensor([t]).to(self.device))

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted) # 

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise 

            imputed_samples[:, i] = current_sample.detach() 
        return imputed_samples
    
    def calc_residual_others(self, observed_data, cond_mask, observed_mask):
        B, K, L = observed_data.shape
        cond_data = cond_mask * observed_data
        predicted = self.Net(observed_data, cond_data, cond_mask)  # (B,K,L) 
        target_mask = observed_mask - cond_mask  
        residual_cond = (observed_data - predicted) * cond_mask 
        residual_target = (observed_data - predicted) * target_mask  
        return residual_cond, residual_target
    
    def calc_residual_valid_others(self, observed_data, cond_mask, observed_mask):
        residual_cond_list, residual_target_list = [], []
        for t in range(self.num_steps):  # calculate loss for all t
            residual_cond, residual_target = self.calc_residual_others(observed_data, cond_mask, observed_mask)
            residual_cond_list.append(residual_cond)
            residual_target_list.append(residual_target)
        residual_cond_all = torch.stack(residual_cond_list, dim=0)
        residual_target_all = torch.stack(residual_target_list, dim=0)
        return residual_cond_all, residual_target_all
    
    def impute_others(self, observed_data, cond_mask, observed_mask, n_samples):
        B, K, L = observed_data.shape
        cond_data = cond_mask * observed_data
        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)
        for i in range(n_samples):  # calculate loss for all t
            predicted = self.Net(observed_data, cond_data, cond_mask)
            imputed_samples[:, i] = predicted.detach()
        return imputed_samples

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
        ) = self.process_data(batch)
        if is_train == 0:
            cond_mask = gt_mask #
            if self.target_strategy == "random_mid_wp":
                cond_mask = self.get_mid_wp_mask(gt_mask, is_train)  
        elif self.target_strategy == "random_mid_wp":
            cond_mask = self.get_mid_wp_mask(observed_mask, is_train)   
        else:
            cond_mask = self.get_randmask(observed_mask) 

        side_info = self.get_side_info(observed_tp, cond_mask) # [16, 145, 36, 36]

        residual_func = self.calc_residual if is_train == 1 else self.calc_residual_valid
        residual_cond, residual_target = residual_func(observed_data, cond_mask, observed_mask, side_info, is_train)

        return residual_cond, residual_target  # [B, K, L]

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask 
  
            if self.target_strategy == "random_mid_wp":
                cond_mask = self.get_mid_wp_mask(gt_mask, is_train=0)   
                        
            target_mask = observed_mask - cond_mask

            side_info = self.get_side_info(observed_tp, cond_mask) 
     
            samples = self.impute(observed_data, cond_mask, side_info, n_samples)

                
            for i in range(len(cut_length)):  # to avoid double evaluation
                target_mask[i, ..., 0 : cut_length[i].item()] = 0

        return samples, observed_data, target_mask, observed_mask, observed_tp


class DIFF_wind(DIFF_base):
    def __init__(self, config, device, model_name, target_dim=40):
        super(DIFF_wind, self).__init__(target_dim, config, device, model_name)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float() #
        observed_mask = batch["observed_mask"].to(self.device).float() # 0-mask 1-not mask
        observed_tp = batch["timepoints"].to(self.device).float() 
        gt_mask = batch["gt_mask"].to(self.device).float() 
        cut_length = batch["cut_length"].to(self.device).long() 
        for_pattern_mask = batch["hist_mask"].to(self.device).float()  

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        for_pattern_mask = for_pattern_mask.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )

