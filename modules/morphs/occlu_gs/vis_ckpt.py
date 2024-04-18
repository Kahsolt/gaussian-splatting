from argparse import ArgumentParser

import torch
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-L', '--load', required=True, help='path to ckpt')
  args = parser.parse_args()

  ckpt = torch.load(args.load, map_location='cpu')
  embeds = ckpt['view_spec_embed']['weight'].cpu().numpy()
  print('vmax:', embeds.max())
  print('vmin:', embeds.min())

  embeds = (embeds - embeds.min()) / (embeds.max() - embeds.min())
  for embed in embeds:
    im: Tensor = embed.reshape((545, 980))
    im3 = np.stack([im] * 3, axis=-1)
    plt.clf()
    plt.imshow(im3)
    plt.tight_layout()
    plt.show()
    plt.close()
