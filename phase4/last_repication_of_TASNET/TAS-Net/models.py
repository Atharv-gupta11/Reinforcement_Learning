import torch
import torch.nn as nn

__all__ = ['DSN', 'MambaDSN', 'SpatialAgent']

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


class DSN(nn.Module):
    """Deep Summarization Network"""
    def __init__(self, in_dim=1024, hid_dim=256, num_layers=1, cell='lstm'):
        super(DSN, self).__init__()
        assert cell in ['lstm', 'gru'], "cell must be either 'lstm' or 'gru'"
        if cell == 'lstm':
            self.rnn = nn.LSTM(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        else:
            self.rnn = nn.GRU(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hid_dim*2, 1)

    def forward(self, x):
        h, _ = self.rnn(x)
        # p = F.sigmoid(self.fc(h))
        p = torch.sigmoid(self.fc(h))
        # pdb.set_trace()
        return p


class MambaDSN(nn.Module):
    """HMS-Mamba Deep Summarization Network (replaces DSN)"""
    def __init__(self, in_dim=192, hid_dim=256, num_layers=1):
        super(MambaDSN, self).__init__()
        self.fc_in = nn.Linear(in_dim, hid_dim)
        
        if Mamba is not None:
            # Using real Mamba SSM layer for linear-time complexity
            self.mamba_layers = nn.ModuleList([
                Mamba(d_model=hid_dim, d_state=16, d_conv=4, expand=2) 
                for _ in range(num_layers)
            ])
        else:
            print("WARNING: mamba_ssm not found! Using standard GRU as fallback in MambaDSN.")
            self.mamba_layers = nn.GRU(hid_dim, hid_dim // 2, num_layers=num_layers, bidirectional=True, batch_first=True)
            
        self.fc_out = nn.Linear(hid_dim, 1)

    def forward(self, x):
        h = self.fc_in(x)
        if Mamba is not None:
            for layer in self.mamba_layers:
                h = layer(h)
        else:
            h, _ = self.mamba_layers(h)
        p = torch.sigmoid(self.fc_out(h))
        return p


class SpatialAgent(nn.Module):
    """HMS-Mamba Spatial/Feature Agent: Selects salient channels/features"""
    def __init__(self, in_dim=192, hid_dim=64):
        super(SpatialAgent, self).__init__()
        # Given the whole sequence, predict which features are salient
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, in_dim)
        
    def forward(self, x):
        # x shape: [batch, seq_len, in_dim]
        # Pool across time to get a global feature representation for the trial
        x_pooled, _ = torch.max(x, dim=1, keepdim=True)
        h = torch.relu(self.fc1(x_pooled))
        p = torch.sigmoid(self.fc2(h)) 
        return p
