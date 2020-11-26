import torch
import numpy as np
from imageio import volread
from unet import UNet2D

p = '../../../../data/test_data/naming_schemes/scheme5/WellH10_PointH10_0002_ChannelDAPI,WF_GFP,TRITC,WF_Cy5,DIA_Seq0713.tiff'
im = volread(p)
im = im[1]

im = np.array(im).astype('float32')
im -= im.mean()
im /= im.std()

with torch.no_grad():
    state = torch.load('./unet-covid-if-v2-weights.torch')
    net = UNet2D(in_channels=1,
                 out_channels=2,
                 f_maps=[32, 64, 128, 256, 512],
                 testing=True)
    net.load_state_dict(state)

    inp = im[None, None, None]
    np.save('./test_input.npy', inp)

    inp = torch.from_numpy(inp)
    out = net(inp)

    out = out.cpu().numpy()

    np.save('./test_output.npy', out)
