import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_torch_trans(heads=8, layers=1, channels=64):
    dropout = 0.1
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu", dropout=dropout
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

def get_torch_trans_dec(heads=8, layers=1, channels=64):
    dropout = 0.1
    decoder_layers = nn.TransformerDecoderLayer(d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu", dropout=dropout)
    transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=layers)

    return transformer_decoder

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)  
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)  
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2) 
        return table


class diff_dec(nn.Module):
    def __init__(self, input_size, config, device, inputdim=2):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.channels = config["channels"]  
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )
        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock_dec(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    device = self.device,
                    input_size = self.input_size,
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, cond_info, enc_output, diffusion_step):
        B, inputdim, K, L = x.shape 
        
        x = x.reshape(B, inputdim, K * L) 
        x = self.input_projection(x)  
        x = F.relu(x) 
        x = x.reshape(B, self.channels, K, L) 

        diffusion_emb = self.diffusion_embedding(diffusion_step) 
        
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, enc_output, diffusion_emb) 
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L) 
        x = self.output_projection1(x)  # (B,channel,K*L) 
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x 
    
class ResidualBlock_dec(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, device, input_size):
        super().__init__()
        self.device = device
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(1, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)
        
        self.conv1d_tf = nn.Conv1d(2 * channels, 2 * channels, kernel_size=1)
        # LSTM
        self.hidden_size = 64
        self.num_directions = 2  
        self.num_layers = 2
        self.dropout1 = nn.Dropout(0.05)
        input_size_lstm = int(2*input_size*channels)
        output_size_lstm = int(input_size*channels)
        self.lstm = nn.LSTM(input_size=input_size_lstm, hidden_size=self.hidden_size, num_layers=self.num_layers, 
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.hidden_size*2, output_size_lstm)
    
        self.time_layer = get_torch_trans_dec(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans_dec(heads=nheads, layers=1, channels=channels)                 

    def forward_time(self, y, enc_output, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L) 
        enc_output = enc_output.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(2, 0, 1), enc_output.permute(2, 0, 1)).permute(1, 2, 0) 
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y  

    def forward_feature(self, y, enc_output, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        enc_output = enc_output.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1), enc_output.permute(2, 0, 1)).permute(1, 2, 0) 
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y
    
    def forward_lstm(self, y, base_shape):
        B, channel, K, L = base_shape 
        input_size = 2*channel*K
        y = y.reshape(B, input_size, L)
        y = y.transpose(1,2)
        h_0 = torch.randn(self.num_directions * self.num_layers, B, self.hidden_size).to(self.device)
        c_0 = torch.randn(self.num_directions * self.num_layers, B, self.hidden_size).to(self.device)
        output, (h_n, c_n) = self.lstm(y, (h_0, c_0))
        output = self.dropout1(output) 
        output = self.fc(output) 
        output =output.transpose(1,2) # output [B, c*K, L]
        return output

    def forward(self, x, cond_info, enc_output, diffusion_emb):
        # cond_info (B, *,K,L)
        B, channel, K, L = x.shape 
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)
        y = x
        if diffusion_emb != None:
            diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
            y = x + diffusion_emb 
        
        emb_t = cond_info[:, :channel, :, :].reshape(B, channel, K*L)
        emb_f = cond_info[:, channel:channel*2, :, :].reshape(B, channel, K*L)
        cond_mask = cond_info[:, -1:, :, :] 
        
        x_t = y + emb_t 
        x_f = y + emb_f  # (B,channel,K*L)
                
        y_t = self.forward_time(x_t, enc_output, base_shape)  
        y_f = self.forward_feature(x_f, enc_output, base_shape)  # (B,channel,K*L)
        
        y_tf = torch.cat((y_t, y_f), dim=1) # [B, 2*C, K*L]
        y_tf = self.conv1d_tf(y_tf) # [B, 2*C, K*L]
        
        y = y_tf
 
        y = self.forward_lstm(y, base_shape)  #[B, c*K, L]
        y = y.reshape(B, channel, K*L)
        y = self.output_projection(y) #[B, 2*c, K*L]
        
        residual, skip = torch.chunk(y, 2, dim=1) #[B, c, K*L]
        x = x.reshape(base_shape) 
        residual = residual.reshape(base_shape)  
        skip = skip.reshape(base_shape)  
        return (x + residual) / math.sqrt(2.0), skip
    
    
class diff_enc(nn.Module):
    def __init__(self, input_size, config, device, inputdim=1):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.channels = config["channels"]  
        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock_enc(
                    side_dim=config["side_dim"],
                    
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    device=self.device,
                    input_size=self.input_size,
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, cond_info):
        B, inputdim, K, L = x.shape
        
        x = x.reshape(B, inputdim, K * L) 
        x = self.input_projection(x)  
        x = F.relu(x) 
        x = x.reshape(B, self.channels, K, L)  
        
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb=None)  
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers)) 
        x = x.reshape(B, self.channels, K * L) 
        x = self.output_projection1(x)  # (B,channel,K*L)  
        x = F.relu(x)
        return x 
    
class ResidualBlock_enc(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, device, input_size):
        super().__init__()
        self.device = device
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(1, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.conv1d_tf = nn.Conv1d(2 * channels, 2 * channels, kernel_size=1)
        # LSTM
        self.hidden_size = 64
        self.num_directions = 2  
        self.num_layers = 2
        self.dropout1 = nn.Dropout(0.05)
        input_size_lstm = int(2*channels*input_size)
        output_size_lstm = int(channels*input_size)
        self.lstm = nn.LSTM(input_size=input_size_lstm, hidden_size=self.hidden_size, 
                            num_layers=self.num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.hidden_size*2, output_size_lstm)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
                    
    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0) 
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y 

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y
    
    def forward_lstm(self, y, base_shape):
        B, channel, K, L = base_shape 
        input_size = int(channel*2*K)
        y = y.reshape(B, input_size, L)
        y = y.transpose(1,2) # [B, L, input_size]
        
        h_0 = torch.randn(self.num_directions * self.num_layers, B, self.hidden_size).to(self.device)
        c_0 = torch.randn(self.num_directions * self.num_layers, B, self.hidden_size).to(self.device)
        output, (h_n, c_n) = self.lstm(y, (h_0, c_0))
        output = self.dropout1(output) 
 
        output = self.fc(output)  # [B, L, channels*input_size]
        output =output.transpose(1,2)
        return output
        
    def forward(self, x, cond_info, diffusion_emb):
        # cond_info (B, *,K,L)
        B, channel, K, L = x.shape 
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)
        y = x
        if diffusion_emb != None:   
            diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
            y = x + diffusion_emb 
        
        emb_t = cond_info[:, :channel, :, :].reshape(B, channel, K*L)
        emb_f = cond_info[:, channel:channel*2, :, :].reshape(B, channel, K*L)
        cond_mask = cond_info[:, -1, :, :].unsqueeze(1) 
        x_t = y + emb_t
        x_f = y + emb_f  # (B,channel,K*L)
                
        y_t = self.forward_time(x_t, base_shape)  
        y_f = self.forward_feature(x_f, base_shape)  # (B,channel,K*L)
        
        y_tf = torch.cat((y_t, y_f), dim=1) 
        y_tf = self.conv1d_tf(y_tf) # [B, 2*C, K*L]
      
        _, cond_dim, _, _ = cond_mask.shape
        cond_info = cond_mask.reshape(B, cond_dim, K * L) 
        cond_info = self.cond_projection(cond_info)  # (B,2 * channels,K*L) 
        y = y_tf + cond_info  
 
        y = self.forward_lstm(y, base_shape) # [B, C*K, L]
        y = y.reshape(B, channel, K*L)
        y = self.output_projection(y)
        
        residual, skip = torch.chunk(y, 2, dim=1) #[B, C*K, L]
        x = x.reshape(base_shape)  
        residual = residual.reshape(base_shape)  
        skip = skip.reshape(base_shape) 
        return (x + residual) / math.sqrt(2.0), skip
    
