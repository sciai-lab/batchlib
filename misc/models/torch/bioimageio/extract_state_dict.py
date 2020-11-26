import torch
from unet import UNet2D

state = torch.load('./fg_and_boundaries_V2.torch')
state = state['model_state_dict']

net = UNet2D(in_channels=1,
             out_channels=2,
             f_maps=[32, 64, 128, 256, 512],
             testing=True)
net.load_state_dict(state)

torch.save(state, './unet-covid-if-v2-weights.torch')
print("Blub")
