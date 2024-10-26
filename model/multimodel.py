import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from config.config import config

resnet_count = len(config['data_types'])


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.values = nn.Linear(embed_dim, embed_dim, bias=False)
        self.keys = nn.Linear(embed_dim, embed_dim, bias=False)
        self.queries = nn.Linear(embed_dim, embed_dim, bias=False)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        N, seq_length, _ = x.shape
        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)
        values = values.view(N, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(N, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(N, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        energy = torch.einsum("nhqd,nhkd->nhqk", [queries, keys])
        attention = torch.softmax(energy / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)), dim=3)
        out = torch.einsum("nhqk,nhvd->nhqd", [attention, values]).reshape(N, seq_length, -1)
        out = self.fc_out(out)
        return out


class MultiResNetAttentionModel(nn.Module):
    def __init__(self, unfreeze_layers=None):
        super(MultiResNetAttentionModel, self).__init__()

        unfreeze_layers = [] if unfreeze_layers is None else unfreeze_layers
        self.resnets = nn.ModuleList([self.create_resnet() for _ in range(resnet_count)])
        self.attention = MultiHeadAttention(embed_dim=512, num_heads=8)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(p=0.2)

        for resnet in self.resnets:
            for name, param in resnet.named_parameters():
                param.requires_grad = True if any(layer in name for layer in unfreeze_layers) else False

    @staticmethod
    def create_resnet():
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        resnet.fc = nn.Identity()
        return resnet

    def forward(self, x):
        batch_size = x.size(0)  # (batch_size, 18, H, W)
        x = x.view(batch_size, resnet_count, 3, *x.size()[2:])  # to (batch_size, data_types_count, 3, H, W)
        features = [self.resnets[i](x[:, i]) for i in range(resnet_count)]  # (batch_size, 512)
        features = torch.stack(features, dim=1)  # (batch_size, data_types_count, 512)

        attentive_features = self.attention(features)  # (batch_size, data_types_count, 512)
        combined_features = torch.mean(attentive_features, dim=1)  # (batch_size, 512)

        out = self.dropout(torch.relu(self.fc1(combined_features)))
        out = self.dropout(torch.relu(self.fc2(out)))
        out = self.fc3(out)
        return out


