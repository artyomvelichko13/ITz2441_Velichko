"""
Альтернативные архитектуры для сегментации
DeepLabV3+ и упрощённая версия SegFormer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== DEEPLABV3+ ====================

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP)
    Использует параллельные свёртки с разными dilation rates
    """
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        
        # Различные dilation rates
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Global average pooling
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Объединяющая свёртка
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        size = x.shape[-2:]
        
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.global_pool(x)
        feat5 = F.interpolate(feat5, size=size, mode='bilinear', align_corners=True)
        
        # Конкатенация всех признаков
        out = torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1)
        out = self.conv_out(out)
        
        return out


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ для семантической сегментации
    Использует ASPP и low-level features
    """
    def __init__(self, in_channels=3, num_classes=5):
        super().__init__()
        
        # Encoder - упрощённый ResNet-like
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Блоки с увеличивающимся числом каналов
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # ASPP
        self.aspp = ASPP(512, 256)
        
        # Low-level features processing
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(64, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        for _ in range(1, blocks):
            layers.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        size = x.shape[-2:]
        
        # Encoder
        x = self.conv1(x)
        x = self.pool(x)
        
        low_level_feat = self.layer1(x)
        x = self.layer2(low_level_feat)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # ASPP
        x = self.aspp(x)
        x = F.interpolate(x, size=(low_level_feat.shape[2], low_level_feat.shape[3]),
                         mode='bilinear', align_corners=True)
        
        # Обработка low-level features
        low_level_feat = self.low_level_conv(low_level_feat)
        
        # Объединение
        x = torch.cat([x, low_level_feat], dim=1)
        
        # Decoder
        x = self.decoder(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        
        return x


# ==================== SIMPLIFIED SEGFORMER ====================

class MLP(nn.Module):
    """Multi-Layer Perceptron для SegFormer"""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.act = nn.GELU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class EfficientSelfAttention(nn.Module):
    """Эффективный механизм self-attention"""
    def __init__(self, dim, num_heads=8, reduction_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.reduction_ratio = reduction_ratio
        
        if reduction_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=reduction_ratio, stride=reduction_ratio)
            self.norm = nn.LayerNorm(dim)
        
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        
        # Query
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Key, Value with spatial reduction
        if self.reduction_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        
        k, v = kv[0], kv[1]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer block для SegFormer"""
    def __init__(self, dim, num_heads, mlp_ratio=4, reduction_ratio=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads, reduction_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SimplifiedSegFormer(nn.Module):
    """
    Упрощённая версия SegFormer
    Transformer-based архитектура для сегментации
    """
    def __init__(self, in_channels=3, num_classes=5, embed_dims=[64, 128, 256, 512]):
        super().__init__()
        
        # Patch embedding для каждого stage
        self.patch_embed1 = nn.Sequential(
            nn.Conv2d(in_channels, embed_dims[0], 7, stride=4, padding=3),
            nn.BatchNorm2d(embed_dims[0])
        )
        
        self.patch_embed2 = nn.Sequential(
            nn.Conv2d(embed_dims[0], embed_dims[1], 3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dims[1])
        )
        
        self.patch_embed3 = nn.Sequential(
            nn.Conv2d(embed_dims[1], embed_dims[2], 3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dims[2])
        )
        
        self.patch_embed4 = nn.Sequential(
            nn.Conv2d(embed_dims[2], embed_dims[3], 3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dims[3])
        )
        
        # Transformer blocks для каждого stage
        self.block1 = TransformerBlock(embed_dims[0], num_heads=1, reduction_ratio=8)
        self.block2 = TransformerBlock(embed_dims[1], num_heads=2, reduction_ratio=4)
        self.block3 = TransformerBlock(embed_dims[2], num_heads=4, reduction_ratio=2)
        self.block4 = TransformerBlock(embed_dims[3], num_heads=8, reduction_ratio=1)
        
        # Decoder - MLP head
        self.decode_head = nn.Sequential(
            nn.Conv2d(sum(embed_dims), 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
    
    def forward(self, x):
        B = x.shape[0]
        size = x.shape[-2:]
        
        # Stage 1
        x1 = self.patch_embed1(x)
        B, C, H, W = x1.shape
        x1_flat = x1.flatten(2).transpose(1, 2)
        x1_trans = self.block1(x1_flat)
        x1_out = x1_trans.transpose(1, 2).reshape(B, C, H, W)
        
        # Stage 2
        x2 = self.patch_embed2(x1_out)
        B, C, H, W = x2.shape
        x2_flat = x2.flatten(2).transpose(1, 2)
        x2_trans = self.block2(x2_flat)
        x2_out = x2_trans.transpose(1, 2).reshape(B, C, H, W)
        
        # Stage 3
        x3 = self.patch_embed3(x2_out)
        B, C, H, W = x3.shape
        x3_flat = x3.flatten(2).transpose(1, 2)
        x3_trans = self.block3(x3_flat)
        x3_out = x3_trans.transpose(1, 2).reshape(B, C, H, W)
        
        # Stage 4
        x4 = self.patch_embed4(x3_out)
        B, C, H, W = x4.shape
        x4_flat = x4.flatten(2).transpose(1, 2)
        x4_trans = self.block4(x4_flat)
        x4_out = x4_trans.transpose(1, 2).reshape(B, C, H, W)
        
        # Upsampling всех features к одному размеру
        target_size = x1_out.shape[-2:]
        x1_up = x1_out
        x2_up = F.interpolate(x2_out, size=target_size, mode='bilinear', align_corners=True)
        x3_up = F.interpolate(x3_out, size=target_size, mode='bilinear', align_corners=True)
        x4_up = F.interpolate(x4_out, size=target_size, mode='bilinear', align_corners=True)
        
        # Объединение всех features
        features = torch.cat([x1_up, x2_up, x3_up, x4_up], dim=1)
        
        # Decoder
        out = self.decode_head(features)
        out = F.interpolate(out, size=size, mode='bilinear', align_corners=True)
        
        return out


# ==================== ТЕСТИРОВАНИЕ ====================

def test_models():
    """Тестирование всех моделей"""
    print("="*60)
    print("ТЕСТИРОВАНИЕ АРХИТЕКТУР")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    img_size = 256
    num_classes = 5
    
    # Создаём тестовый вход
    x = torch.randn(batch_size, 3, img_size, img_size).to(device)
    
    models_dict = {
        'DeepLabV3+': DeepLabV3Plus(in_channels=3, num_classes=num_classes),
        'SimplifiedSegFormer': SimplifiedSegFormer(in_channels=3, num_classes=num_classes)
    }
    
    for name, model in models_dict.items():
        model = model.to(device)
        model.eval()
        
        print(f"\n{name}:")
        print("-" * 60)
        
        # Считаем параметры
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Всего параметров: {total_params:,}")
        print(f"Обучаемых параметров: {trainable_params:,}")
        
        # Тест forward pass
        with torch.no_grad():
            output = model(x)
        
        print(f"Входной размер: {x.shape}")
        print(f"Выходной размер: {output.shape}")
        print(f"✓ Модель работает корректно")


if __name__ == '__main__':
    test_models()
