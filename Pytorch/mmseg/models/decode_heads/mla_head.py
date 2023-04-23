import torch
import torch.nn as nn
import torch.nn.functional as F


from ..builder import HEADS
from .decode_head import BaseDecodeHead

from mmcv.cnn import build_norm_layer


class MLAHeads(nn.Module):
    def __init__(self, mlahead_channels=128, norm_cfg=None):
        super(MLAHeads, self).__init__()
        self.head2 = nn.Sequential(nn.Conv2d(mlahead_channels*2, mlahead_channels, 3, padding=1, bias=False),
                                   build_norm_layer(norm_cfg, mlahead_channels)[
            1], nn.ReLU(),
            nn.Conv2d(
                                       mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU())
        self.head3 = nn.Sequential(nn.Conv2d(mlahead_channels*2, mlahead_channels, 3, padding=1, bias=False),
                                   build_norm_layer(norm_cfg, mlahead_channels)[
            1], nn.ReLU(),
            nn.Conv2d(
                                       mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU())
        self.head4 = nn.Sequential(nn.Conv2d(mlahead_channels*2, mlahead_channels, 3, padding=1, bias=False),
                                   build_norm_layer(norm_cfg, mlahead_channels)[
            1], nn.ReLU(),
            nn.Conv2d(
                                       mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU())
        self.head5 = nn.Sequential(nn.Conv2d(mlahead_channels*2, mlahead_channels, 3, padding=1, bias=False),
                                   build_norm_layer(norm_cfg, mlahead_channels)[
            1], nn.ReLU(),
            nn.Conv2d(
                                       mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, mlahead_channels)[1], nn.ReLU())

    def forward(self, mla_p2, mla_p3, mla_p4, mla_p5):
        head2 = F.interpolate(self.head2(
            mla_p2), 4*mla_p2.shape[-1], mode='bilinear', align_corners=True)
        head3 = F.interpolate(self.head3(
            mla_p3), 4*mla_p3.shape[-1], mode='bilinear', align_corners=True)
        head4 = F.interpolate(self.head4(
            mla_p4), 4*mla_p4.shape[-1], mode='bilinear', align_corners=True)
        head5 = F.interpolate(self.head5(
            mla_p5), 4*mla_p5.shape[-1], mode='bilinear', align_corners=True)
        return torch.cat([head2, head3, head4, head5], dim=1)


@HEADS.register_module()
class MLAHead(BaseDecodeHead):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=1024, mlahead_channels=128,
                 norm_layer=nn.BatchNorm2d, norm_cfg=None, **kwargs):
        super(MLAHead, self).__init__(input_transform='multiple_select', **kwargs)
        self.img_size = img_size
        self.norm_cfg = norm_cfg
        self.BatchNorm = norm_layer
        self.mlahead_channels = mlahead_channels

        self.mlahead = MLAHeads(
                               mlahead_channels=self.mlahead_channels, norm_cfg=self.norm_cfg)
        self.cls = nn.Conv2d(4 * self.mlahead_channels,
                             self.num_classes, 3, padding=1)
            
        self.conv0 = nn.Sequential(nn.Conv2d(self.in_channels[0], self.in_channels[0]*2, 3, padding=1, bias=False),
                                   build_norm_layer(norm_cfg, self.in_channels[0]*2)[1], nn.ReLU())
        self.conv1 = nn.Sequential(nn.Conv2d(self.in_channels[1], self.in_channels[1], 3, padding=1, bias=False),
                                   build_norm_layer(norm_cfg, self.in_channels[1])[1], nn.ReLU())
        self.conv21 = nn.Sequential(nn.Conv2d(self.in_channels[2], self.in_channels[2], 3, padding=1, bias=False),
                                   build_norm_layer(norm_cfg, self.in_channels[2])[1], nn.ReLU())
        self.conv22 = nn.Sequential(nn.Conv2d(self.in_channels[2], self.in_channels[2]//2, 3, padding=1, bias=False),
                                   build_norm_layer(norm_cfg, self.in_channels[2]//2)[1], nn.ReLU())
        self.conv31 = nn.Sequential(nn.Conv2d(self.in_channels[3], self.in_channels[3], 3, padding=1, bias=False),
                                   build_norm_layer(norm_cfg, self.in_channels[3])[1], nn.ReLU())
        self.conv32 = nn.Sequential(nn.Conv2d(self.in_channels[3], self.in_channels[3]//2, 3, padding=1, bias=False),
                                   build_norm_layer(norm_cfg, self.in_channels[3]//2)[1], nn.ReLU())
        self.conv33 = nn.Sequential(nn.Conv2d(self.in_channels[3]//2, self.in_channels[3]//4, 3, padding=1, bias=False),
                                   build_norm_layer(norm_cfg, self.in_channels[3]//4)[1], nn.ReLU())
                                     

    def forward(self, inputs):
        inputs0 = self.conv0(inputs[0])
        inputs1 = F.interpolate(self.conv1(inputs[1]), size=(inputs[0].size()[-2], inputs[0].size()[-1]), mode='bilinear', align_corners=True)
        inputs2 = F.interpolate(self.conv21(inputs[2]), scale_factor=2, mode='bilinear', align_corners=True)
        inputs2 = F.interpolate(self.conv22(inputs2), size=(inputs[0].size()[-2], inputs[0].size()[-1]), mode='bilinear', align_corners=True) 
        inputs3 = F.interpolate(self.conv31(inputs[3]), scale_factor=2, mode='bilinear', align_corners=True) 
        inputs3 = F.interpolate(self.conv32(inputs3), scale_factor=2, mode='bilinear', align_corners=True) 
        inputs3 = F.interpolate(self.conv33(inputs3), size=(inputs[0].size()[-2], inputs[0].size()[-1]), mode='bilinear', align_corners=True) 
        inputs2 = inputs2 + inputs3
        inputs1 = inputs1 + inputs2
        inputs0 = inputs0 + inputs1
        
        x = self.mlahead(inputs0, inputs1, inputs2, inputs3)
        x = self.cls(x)
        
        return x

# @HEADS.register_module()
# class MLAHead(BaseDecodeHead):
#     """ Vision Transformer with support for patch or hybrid CNN input stage
#     """

#     def __init__(self, img_size=768, mlahead_channels=128,
#                  norm_layer=nn.BatchNorm2d, norm_cfg=None, **kwargs):
#         super(MLAHead, self).__init__(input_transform='multiple_select', **kwargs)
#         self.img_size = img_size
#         self.norm_cfg = norm_cfg
#         self.BatchNorm = norm_layer
#         self.mlahead_channels = mlahead_channels

#         self.mlahead = MLAHeads(
#                                mlahead_channels=self.mlahead_channels, norm_cfg=self.norm_cfg)
#         self.cls = nn.Conv2d(4 * self.mlahead_channels,
#                              self.num_classes, 3, padding=1)
            
#         self.conv1 = nn.Sequential(nn.Conv2d(self.in_channels[1], self.in_channels[1]//2, 3, padding=1, bias=False),
#                                    build_norm_layer(norm_cfg, self.in_channels[1]//2)[1], nn.ReLU())
#         self.conv2 = nn.Sequential(nn.Conv2d(self.in_channels[2], self.in_channels[2]//4, 3, padding=1, bias=False),
#                                    build_norm_layer(norm_cfg, self.in_channels[2]//4)[1], nn.ReLU())
#         self.conv3 = nn.Sequential(nn.Conv2d(self.in_channels[3], self.in_channels[3]//8, 3, padding=1, bias=False),
#                                    build_norm_layer(norm_cfg, self.in_channels[3]//8)[1], nn.ReLU())
                                     

#     def forward(self, inputs):
#         inputs0 = inputs[0]
#         print(inputs[3].size())
#         inputs1 = F.interpolate(self.conv1(inputs[1]), scale_factor=2, mode='bilinear', align_corners=True)
#         inputs2 = F.interpolate(self.conv2(inputs[2]), scale_factor=4, mode='bilinear', align_corners=True)
#         inputs3 = F.interpolate(self.conv3(inputs[3]), scale_factor=8, mode='bilinear', align_corners=True) 
#         inputs2 = inputs2 + inputs3
#         inputs1 = inputs1 + inputs2
#         inputs0 = inputs0 + inputs1
        
#         x = self.mlahead(inputs0, inputs1, inputs2, inputs3)
#         x = self.cls(x)
#         x = F.interpolate(x, size=self.img_size, mode='bilinear',
#                           align_corners=self.align_corners)
#         return x