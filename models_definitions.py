# models_definitions.py
import torch
import torch.nn as nn

class TSM(nn.Module):
    def __init__(self, n_segment, fold_div=3):
        super(TSM, self).__init__()
        self.n_segment, self.fold_div = n_segment, fold_div
    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        fold = c // self.fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]
        out[:, 1:, fold:2*fold] = x[:, :-1, fold:2*fold]
        out[:, :, 2*fold:] = x[:, :, 2*fold:]
        return out.view(nt, c, h, w)

class TSCAN(nn.Module):
    def __init__(self, frames=300):
        super(TSCAN, self).__init__()
        self.motion_conv1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.Tanh())
        self.tsm1 = TSM(frames)
        self.motion_conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.Tanh())
        self.tsm2 = TSM(frames)
        self.app_conv1 = nn.Conv2d(3, 32, 3, padding=1); self.app_conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.att_mask1 = nn.Conv2d(32, 1, 1); self.att_mask2 = nn.Conv2d(64, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(64, 1)
    def forward(self, diff, raw):
        b, c, t, h, w = diff.shape
        diff = diff.permute(0, 2, 1, 3, 4).reshape(b*t, c, h, w)
        raw = raw.permute(0, 2, 1, 3, 4).reshape(b*t, c, h, w)
        m1 = self.tsm1(self.motion_conv1(diff)); a1 = torch.tanh(self.app_conv1(raw))
        m1 = m1 * torch.sigmoid(self.att_mask1(a1))
        m2 = self.tsm2(self.motion_conv2(m1)); a2 = torch.tanh(self.app_conv2(a1))
        m2 = m2 * torch.sigmoid(self.att_mask2(a2))
        out = self.fc(self.dropout(self.avg_pool(m2).view(b*t, -1)))
        return out.view(b, t)

class ResidualBlock1D(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_c, out_c, 3, stride, 1), nn.BatchNorm1d(out_c), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(out_c, out_c, 3, 1, 1), nn.BatchNorm1d(out_c)
        )
        self.skip = nn.Sequential(nn.Conv1d(in_c, out_c, 1, stride), nn.BatchNorm1d(out_c)) if stride != 1 or in_c != out_c else nn.Sequential()
    def forward(self, x): return torch.relu(self.conv(x) + self.skip(x))

class HeartRateResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.prep = nn.Sequential(nn.Conv1d(1, 32, 7, 2, 3), nn.BatchNorm1d(32), nn.ReLU())
        self.layer1 = ResidualBlock1D(32, 64, 2)
        self.layer2 = ResidualBlock1D(64, 128, 2)
        self.layer3 = ResidualBlock1D(128, 256, 2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))
    def forward(self, x): return self.fc(torch.flatten(self.avgpool(self.layer3(self.layer2(self.layer1(self.prep(x))))), 1))