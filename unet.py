import torch 
import torch.nn as nn
import math


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=8, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_
    
class Unet(nn.Module):
    def __init__(self, is_attention=True):
        super(Unet, self).__init__()
        self.embed_dim = 32 
        self.out_dim = self.embed_dim * 4
        
        self.temb_layer = self.get_temb_layer(self.embed_dim)
        
        self.down1 = nn.Sequential(nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.temb_proj1 = self.get_temb_projection(self.out_dim, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.down2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.temb_proj2 = self.get_temb_projection(self.out_dim, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.middle = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())
       
        if is_attention: 
            self.attn = AttnBlock(256)
        else:
            self.attn = None

        self.temb_proj3 = self.get_temb_projection(self.out_dim, 256)

        self.up1 = nn.Sequential(nn.ConvTranspose2d(256, 128, 2, stride=2), nn.ReLU())
        self.conv_up1 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.temb_proj4 = self.get_temb_projection(self.out_dim, 128)

        self.up2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 2, stride=2), nn.ReLU())
        self.conv_up2 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.temb_proj5 = self.get_temb_projection(self.out_dim, 64)

        self.final = nn.Conv2d(64, 1, 1)

    def get_temb_layer(self, in_channels):
        return nn.Sequential(nn.Linear(in_channels, in_channels * 4), nn.ReLU(), nn.Linear(in_channels * 4, in_channels * 4))
    
    def get_temb_projection(self, in_channels, out_channels):
        return nn.Linear(in_channels, out_channels)
    
    def get_timestep_embedding(self, timesteps, embedding_dim):
        """
        This matches the implementation in Denoising Diffusion Probabilistic Models:
        From Fairseq.
        Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        assert len(timesteps.shape) == 1

        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = emb.to(device=timesteps.device)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb

    def forward(self, x, t):
        t_emb = self.get_timestep_embedding(t, self.embed_dim)
        t_emb = self.temb_layer(t_emb)

        down1 = self.down1(x) + self.temb_proj1(t_emb)[:, :, None, None]
        pool1 = self.pool1(down1) 
        down2 = self.down2(pool1) + self.temb_proj2(t_emb)[:, :, None, None]
        pool2 = self.pool2(down2)

        middle = self.middle(pool2) + self.temb_proj3(t_emb)[:, :, None, None]
        if self.attn is not None:
            middle = self.attn(middle)

        up1 = self.up1(middle)
        up1 = torch.cat((up1, down2), dim=1)
        up1 = self.conv_up1(up1) + self.temb_proj4(t_emb)[:, :, None, None]

        up2 = self.up2(up1)
        up2 = torch.cat((up2, down1), dim=1)
        up2 = self.conv_up2(up2) + self.temb_proj5(t_emb)[:, :, None, None]

        return self.final(up2)

class UnetOld(nn.Module):
    def __init__(self, is_attention=True):
        super(UnetOld, self).__init__()
        self.embed_dim = 32 
        self.out_dim = self.embed_dim * 4
        
        self.temb_layer0 = self.get_temb_layer(self.embed_dim, self.out_dim)
        
        self.down1 = nn.Sequential(nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.temb_layer1 = self.get_temb_layer(self.out_dim, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.down2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.temb_layer2 = self.get_temb_layer(self.out_dim, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.middle = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())
        
        if is_attention:
            self.attn = AttnBlock(256)
        else: 
            self.attn = None

        self.temb_layer3 = self.get_temb_layer(self.out_dim, 256)

        self.up1 = nn.Sequential(nn.ConvTranspose2d(256, 128, 2, stride=2), nn.ReLU())
        self.conv_up1 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.temb_layer4 = self.get_temb_layer(self.out_dim, 128)

        self.up2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 2, stride=2), nn.ReLU())
        self.conv_up2 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.temb_layer5 = self.get_temb_layer(self.out_dim, 64)

        self.final = nn.Conv2d(64, 1, 1)

    def get_temb_layer(self, in_channels, out_channels):
        return nn.Sequential(nn.Linear(in_channels, in_channels * 4), nn.ReLU(), nn.Linear(in_channels * 4, out_channels))
    
    def get_timestep_embedding(self, timesteps, embedding_dim):
        """
        This matches the implementation in Denoising Diffusion Probabilistic Models:
        From Fairseq.
        Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        assert len(timesteps.shape) == 1

        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = emb.to(device=timesteps.device)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb

    def forward(self, x, t):
        t_emb = self.get_timestep_embedding(t, self.embed_dim)
        t_emb = self.temb_layer0(t_emb)

        down1 = self.down1(x) + self.temb_layer1(t_emb)[:, :, None, None]
        pool1 = self.pool1(down1) 
        down2 = self.down2(pool1) + self.temb_layer2(t_emb)[:, :, None, None]
        pool2 = self.pool2(down2)

        middle = self.middle(pool2) + self.temb_layer3(t_emb)[:, :, None, None]

        if self.attn is not None:
            middle = self.attn(middle)

        up1 = self.up1(middle)
        up1 = torch.cat((up1, down2), dim=1)
        up1 = self.conv_up1(up1) + self.temb_layer4(t_emb)[:, :, None, None]

        up2 = self.up2(up1)
        up2 = torch.cat((up2, down1), dim=1)
        up2 = self.conv_up2(up2) + self.temb_layer5(t_emb)[:, :, None, None]

        return self.final(up2)