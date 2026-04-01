import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class ConvSubsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvSubsampling, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x, lengths):
        # x: (batch, time, in_channels)
        x = x.unsqueeze(1) # (batch, 1, time, in_channels)
        
        # We need to swap time and channels for Conv2d -> (batch, 1, in_channels, time)
        x = x.permute(0, 1, 3, 2) 
        
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        
        # lengths adjustment for two stride=2 convs with kernel_size=3
        lengths = ((lengths - 1) // 2)
        lengths = ((lengths - 1) // 2)
        
        batch_size, channels, freq, time = x.size()
        x = x.view(batch_size, channels * freq, time)
        x = x.permute(0, 2, 1) # (batch, time, channels*freq)
        
        return x, lengths

class FeedForwardModule(nn.Module):
    def __init__(self, d_model, ffn_dim, dropout):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ffn_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.sequential(x)

class MultiHeadAttentionModule(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, key_padding_mask=None):
        x_norm = self.norm(x)
        out, _ = self.mha(query=x_norm, key=x_norm, value=x_norm, key_padding_mask=key_padding_mask)
        return self.dropout(out)

class DepthwiseConvModule(nn.Module):
    def __init__(self, d_model, conv_kernel):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        
        # Pointwise
        self.pointwise1 = nn.Conv1d(d_model, 2*d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        
        # Depthwise
        padding = (conv_kernel - 1) // 2
        self.depthwise = nn.Conv1d(d_model, d_model, kernel_size=conv_kernel, padding=padding, groups=d_model)
        
        self.norm_dw = nn.BatchNorm1d(d_model)
        self.swish = nn.SiLU()
        
        # Pointwise
        self.pointwise2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        
    def forward(self, x):
        # x: (batch, time, d_model)
        x_norm = self.norm(x)
        x_norm = x_norm.transpose(1, 2) # (batch, d_model, time)
        
        x_conv = self.pointwise1(x_norm)
        x_conv = self.glu(x_conv)
        
        x_conv = self.depthwise(x_conv)
        x_conv = self.norm_dw(x_conv)
        x_conv = self.swish(x_conv)
        
        x_conv = self.pointwise2(x_conv)
        
        return x_conv.transpose(1, 2) # (batch, time, d_model)

class SqueezeformerBlock(nn.Module):
    def __init__(self, d_model=256, num_heads=4, ffn_dim=1024, conv_kernel=31, dropout=0.1):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, ffn_dim, dropout)
        self.mha = MultiHeadAttentionModule(d_model, num_heads, dropout)
        self.conv = DepthwiseConvModule(d_model, conv_kernel)
        self.ffn2 = FeedForwardModule(d_model, ffn_dim, dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, key_padding_mask=None):
        x = x + 0.5 * self.ffn1(x)
        x = x + self.mha(x, key_padding_mask)
        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(x)
        x = self.norm(x)
        return x

class SqueezeformerEncoder(nn.Module):
    def __init__(self, input_dim=80, num_layers=16, d_model=256, num_heads=4, ffn_dim=1024, conv_kernel=31, dropout=0.1, vocab_size=2000):
        super().__init__()
        # input_dim = 80, we use conv_subsampling with out_channels=d_model
        # The output of conv_subsampling will be (d_model * ((80-1)//2 - 1)//2)
        # 80 -> 39 -> 19
        subsampled_dim = d_model * (((input_dim - 1) // 2 - 1) // 2)
        
        self.conv_subsampling = ConvSubsampling(in_channels=1, out_channels=d_model)
        self.input_proj = nn.Linear(subsampled_dim, d_model)
        
        self.pos_enc = PositionalEncoding(d_model, dropout)
        
        self.layers = nn.ModuleList([
            SqueezeformerBlock(d_model, num_heads, ffn_dim, conv_kernel, dropout)
            for _ in range(num_layers)
        ])
        
        self.final_layer_norm = nn.LayerNorm(d_model)
        self.ctc_decoder = nn.Linear(d_model, vocab_size)

    def forward(self, x, lengths):
        # x: (batch, time, input_dim)
        x, lengths = self.conv_subsampling(x, lengths)
        
        x = self.input_proj(x)
        x = self.pos_enc(x)
        
        batch_size, time = x.size(0), x.size(1)
        
        # Mask creation: lengths tells us valid lengths
        # key_padding_mask should be True for padded positions
        idx = torch.arange(time, device=x.device).unsqueeze(0).expand(batch_size, -1)
        key_padding_mask = idx >= lengths.unsqueeze(1)
        
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
            
        x = self.final_layer_norm(x)
        logits = self.ctc_decoder(x)
        return logits, lengths
