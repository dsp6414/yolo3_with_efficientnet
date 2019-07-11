import torch
from vedanet.network.backbone import EfficientNet
from vedanet.network.backbone import Darknet53
# from modelsummary import summary

m = EfficientNet('efficientnet-b7')
# m = Darknet53()
i = torch.zeros(1,3,416,416)
o = m(i)
for oo in o:
    print(oo.shape)


