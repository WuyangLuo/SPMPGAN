from utils import *
import argparse
import numpy as np
import datetime
from trainer import SPMPGAN_Trainer
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import shutil
import cv2

# GPU
os.environ["CUDA_VISIBLE_DEVICES"] = '2'


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the config file.')
parser.add_argument('--dataset_name', type=str, required=True,  help='name of dataset.')
parser.add_argument('--ckt_path', type=str, required=True, help='Path to the trained model.')
parser.add_argument('--image_path', type=str, required=True, help='Path to edited image.')
parser.add_argument('--segmap_path', type=str, required=True, help='Path to the segmantation map.')
parser.add_argument('--mask_path', type=str, required=True, help='Path to the mask.')

opts = parser.parse_args()
print_options(opts)



def get_transform(img, normalize=True):
    transform_list = []

    transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)(img)


def get_edges(t):
    edge = torch.cuda.ByteTensor(t.size()).zero_()
    edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
    edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
    edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
    edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
    
    return edge.float()

def test_results(dataset_name, ckt_path, image_path, segmap_path, mask_path):
    cfg = get_config(opts.config)
    if dataset_name == 'ADE20k-room' or dataset_name == 'ADE20k-landscape':
        cfg['lab_dim'] = 151
    elif dataset_name == 'cityscapes':
        cfg['lab_dim'] = 34
    
    trainer = SPMPGAN_Trainer(cfg)
    trainer.cuda()
    state_dict = torch.load(ckt_path)['netG']
    print('load trained model from ', ckt_path)
    trainer.netG.load_state_dict(state_dict)
    
    
    img = cv2.imread(image_path)
    lab = cv2.imread(segmap_path, 0)
    
    mask = cv2.imread(mask_path, 0)/255
    mask = mask.reshape((1,) + mask.shape).astype(np.float32)
    mask = torch.from_numpy(mask)

    img = get_transform(img)
    lab = get_transform(lab, normalize=False)
    lab = lab * 255.0
    masked_img = img * (1. - mask)

    img = img.unsqueeze(0).cuda()
    lab = lab.unsqueeze(0).cuda()
    mask = mask.unsqueeze(0).cuda()
    masked_img = masked_img.unsqueeze(0).cuda()
    
    # edgemap
    edge_map = get_edges(lab)
    edge_map = edge_map * mask

    # create one-hot label map
    lab_map = lab
    # lab_map = lab + 1
    bs, _, h, w = lab_map.size()
    nc = cfg['lab_dim']
    input_label = torch.cuda.FloatTensor(bs, nc, h, w).zero_()
    segmap = input_label.scatter_(1, lab_map.long(), 1.0)
    segmap = segmap * mask
    
    segmap_edge = torch.cat((segmap, edge_map), dim=1)

    trainer.eval()
    masked_fake = trainer.test(img, masked_img, segmap_edge, mask)
    
    save_dir = 'results/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    cv2.imwrite(os.path.join(save_dir, 'test_single_result.jpg'), tensor2im(masked_fake, no_fg=True))
    cv2.imwrite(os.path.join(save_dir, 'masked_img.jpg'), tensor2im(masked_img, no_fg=True))


test_results(opts.dataset_name, opts.ckt_path, opts.image_path, opts.segmap_path, opts.mask_path)

