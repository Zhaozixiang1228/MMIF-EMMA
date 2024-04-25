import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange
import numbers
class Restormer_CNN_block(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(Restormer_CNN_block, self).__init__()
        self.embed=nn.Conv2d(in_dim, out_dim,kernel_size=3,stride=1, padding=1, bias=False,padding_mode="reflect")
        self.GlobalFeature = GlobalFeatureExtraction(dim=out_dim, num_heads = 8)
        self.LocalFeature = LocalFeatureExtraction(dim=out_dim)
        self.FFN=nn.Conv2d(out_dim*2, out_dim,kernel_size=3,stride=1, padding=1, bias=False,padding_mode="reflect")          
    def forward(self, x):
        x=self.embed(x)
        x1=self.GlobalFeature(x)
        x2=self.LocalFeature(x)
        out=self.FFN(torch.cat((x1,x2),1))
        return out
class GlobalFeatureExtraction(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expansion_factor=1.,  
                 qkv_bias=False,):
        super(GlobalFeatureExtraction, self).__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias,)
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.mlp = Mlp(in_features=dim,out_fratures=dim,
                       ffn_expansion_factor=ffn_expansion_factor,)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class LocalFeatureExtraction(nn.Module):
    def __init__(self,
                 dim=64,
                 num_blocks=2,
                 ):
        super(LocalFeatureExtraction, self).__init__()
        self.Extraction = nn.Sequential(*[ResBlock(dim,dim) for i in range(num_blocks)])
    def forward(self, x):
        return self.Extraction(x)
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True,padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True,padding_mode="reflect"),
        )
    def forward(self, x):
        out = self.conv(x)
        return out+x

class AttentionBase(nn.Module):
    def __init__(self,
                 dim,   
                 num_heads=8,
                 qkv_bias=False,):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim*3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim*3, dim*3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):

        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.proj(out)
        return out
    
class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, 
                 in_features, 
                 out_fratures,
                 ffn_expansion_factor = 2,
                 bias = False):
        super().__init__()
        hidden_features = int(in_features*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            in_features, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias,padding_mode="reflect")

        self.project_out = nn.Conv2d(
            hidden_features, out_fratures, kernel_size=1, bias=bias)
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
    
##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class Ufuser(nn.Module):
    def __init__(self):
        super(Ufuser, self).__init__()
        
        channel=[8,16,32,32]
        self.V_en_1 = Restormer_CNN_block(1, channel[0])
        self.V_en_2 = Restormer_CNN_block(channel[0], channel[1])
        self.V_en_3 = Restormer_CNN_block(channel[1], channel[2])
        self.V_en_4 = Restormer_CNN_block(channel[2], channel[3])

        self.I_en_1 = Restormer_CNN_block(1, channel[0])
        self.I_en_2 = Restormer_CNN_block(channel[0], channel[1])
        self.I_en_3 = Restormer_CNN_block(channel[1], channel[2])
        self.I_en_4 = Restormer_CNN_block(channel[2], channel[3])

        self.f_1 = Restormer_CNN_block(channel[0]*2, channel[0])
        self.f_2 = Restormer_CNN_block(channel[1]*2, channel[1])
        self.f_3 = Restormer_CNN_block(channel[2]*2, channel[2])
        self.f_4 = Restormer_CNN_block(channel[3]*2, channel[3])

        self.V_down1=nn.Conv2d(channel[0], channel[0], kernel_size=3, stride=2, padding=1, bias=False,padding_mode="reflect")
        self.V_down2=nn.Conv2d(channel[1], channel[1], kernel_size=3, stride=2, padding=1, bias=False,padding_mode="reflect")
        self.V_down3=nn.Conv2d(channel[2], channel[2], kernel_size=3, stride=2, padding=1, bias=False,padding_mode="reflect")
        

        self.I_down1=nn.Conv2d(channel[0], channel[0], kernel_size=3, stride=2, padding=1, bias=False,padding_mode="reflect")
        self.I_down2=nn.Conv2d(channel[1], channel[1], kernel_size=3, stride=2, padding=1, bias=False,padding_mode="reflect")
        self.I_down3=nn.Conv2d(channel[2], channel[2], kernel_size=3, stride=2, padding=1, bias=False,padding_mode="reflect")
        

        self.up4=nn.Sequential(
            nn.ConvTranspose2d(channel[3],channel[2], 4, 2, 1, bias=False),
            nn.ReLU()
        )
        self.up3=nn.Sequential(
            nn.ConvTranspose2d(channel[2],channel[1], 4, 2, 1, bias=False),
            nn.ReLU()
        )
        self.up2=nn.Sequential(
            nn.ConvTranspose2d(channel[1],channel[0], 4, 2, 1, bias=False),
            nn.ReLU()
        )

        self.de_1 = Restormer_CNN_block(channel[0]*2,channel[0])
        self.de_2 = Restormer_CNN_block(channel[1]*2,channel[1])
        self.de_3 = Restormer_CNN_block(channel[2]*2,channel[2])
        self.de_4 = Restormer_CNN_block(channel[3],channel[3])


        self.last = nn.Sequential(
            nn.Conv2d(channel[0], 1, kernel_size=3, stride=1, padding=1,padding_mode="reflect"),
            nn.Sigmoid()
        )

    def forward(self, i,v):
        i_1=self.I_en_1(i)
        i_2=self.I_en_2(self.I_down1(i_1))
        i_3=self.I_en_3(self.I_down2(i_2))
        i_4=self.I_en_4(self.I_down3(i_3))

        v_1=self.V_en_1(v)
        v_2=self.V_en_2(self.V_down1(v_1))
        v_3=self.V_en_3(self.V_down2(v_2))
        v_4=self.V_en_4(self.V_down3(v_3))

        f_1=self.f_1(torch.cat((i_1,v_1),1))
        f_2=self.f_2(torch.cat((i_2,v_2),1))
        f_3=self.f_3(torch.cat((i_3,v_3),1))
        f_4=self.f_4(torch.cat((i_4,v_4),1))

        out=self.up4(self.de_4(f_4))
        out=self.up3(self.de_3(torch.cat((out,f_3),1)))
        out=self.up2(self.de_2(torch.cat((out,f_2),1)))
        out=self.de_1(torch.cat((out,f_1),1))

        return self.last(out)

