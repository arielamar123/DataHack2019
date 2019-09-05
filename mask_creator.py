from __future__ import print_function

import os
import argparse

import numpy as np

import torch.utils.data
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms

from PIL import Image
from skimage.io import imsave

from torch.autograd import Variable

import dss as dss

from guided_filter_pytorch.guided_filter import GuidedFilter


def create_mask(image_name):
    cudnn.benchmark = True
    netG = dss.network_dss(3, True, 8, 0.01, True)
    netG.load_state_dict(torch.load('/content/dss_latest.pth',map_location=torch.device('cpu')))
    dgf = GuidedFilter(8, 1e-2)
    img = transforms.ToTensor()(Image.open('/content/'+image_name).convert('RGB'))
    real_x = img.unsqueeze(0)
    real_A = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
    real_A.unsqueeze_(0)
    input_G = Variable(real_A, volatile=True)
    input_x = Variable(real_x, volatile=True)

    fake_B, side_out1, side_out2, side_out3, side_out4, side_out5, side_out6 = netG(input_G, input_x)
    input_x = input_x.sum(1, keepdim=True)
    image_B = dgf(input_x, fake_B).clamp(0, 1)

    image_B = image_B.data.cpu().mul(255).numpy().squeeze().astype(np.uint8)
    image_B[image_B >= 161] = 255
    image_B[image_B <= 161] = 0

    output_directory = os.path.dirname('/content/'+image_name)
    output_name = os.path.splitext(os.path.basename('/content/'+image_name))[0]
    save_path = os.path.join(output_directory, '/content/{}_labels.png'.format(output_name))
    imsave(save_path, image_B)
