import os
import cv2
import math
import numpy as np
from torch.utils.data import Dataset
import os.path
import random
import torchvision.transforms as transforms
import torch
from utils import make_dataset
from PIL import Image, ImageDraw


class Image_Editing_Dataset(Dataset):
    def __init__(self, cfg, dataset_root, split='train', dataset_name=''):
        self.split = split
        self.cfg = cfg
        self.dataset_name = dataset_name

        self.dir_img = os.path.join(dataset_root, self.split, 'images')
        self.dir_lab = os.path.join(dataset_root, self.split, 'labels')
        self.dir_ins = os.path.join(dataset_root, self.split, 'inst_map')
        name_list = os.listdir(self.dir_img)
        self.name_list = [n[:-4] for n in name_list if n.endswith('jpg')]
        
        if self.split == 'test':
            self.name_list.sort()
            
            self.predefined_mask_path = 'data/predefined_mask/'
            mask_list = os.listdir(self.predefined_mask_path)
            mask_list.sort()
            self.mask_list = mask_list[:len(self.name_list)]

    def __getitem__(self, index):
        name = self.name_list[index]
        # input data
        img = cv2.imread(os.path.join(self.dir_img, name + '.jpg'))
        lab = cv2.imread(os.path.join(self.dir_lab, name + '.png'), 0)

        if self.dataset_name == 'cityscapes':
            inst_map = Image.open(os.path.join(self.dir_ins, name + '.png'))
            inst_map = np.array(inst_map, dtype=np.int32)
        elif self.dataset_name == 'ADE20k-room':
            inst_map = cv2.imread(os.path.join(self.dir_ins, name + '.png'))
            inst_map = inst_map[:, :, 1]
        
        if self.split == 'train':
            # resize
            size = (self.cfg['crop_size'], self.cfg['crop_size'])
            h, w, _ = img.shape
            w_l = 0
            h_l = 0
            if w > 256:
                w_l = random.randint(0, w - 256)
            if h > 256:
                h_l = random.randint(0, h - 256)
            
            img = img[h_l:h_l+256, w_l:w_l+256]
            lab = lab[h_l:h_l+256, w_l:w_l+256]
            if self.dataset_name == 'ADE20k-room' or self.dataset_name == 'cityscapes':
                inst_map = inst_map[h_l:h_l+256, w_l:w_l+256]
            # flip
            if random.random() > 0.5:
                img = np.flip(img,axis=1).copy()
                lab = np.flip(lab,axis=1).copy()
                if self.dataset_name == 'ADE20k-room' or self.dataset_name == 'cityscapes':
                    inst_map = np.flip(inst_map,axis=1).copy()
        
        # select inst id
        if self.dataset_name == 'cityscapes':
            inst_ids = np.unique(inst_map)
            inst_ids = inst_ids.tolist()
            inst_ids = [i for i in inst_ids if i>=1000]  # filter out non-instance masks
        elif self.dataset_name == 'ADE20k-room':
            inst_ids = np.unique(inst_map)
            inst_ids = inst_ids.tolist()

        if self.dataset_name == 'ADE20k-room' or self.dataset_name == 'cityscapes':
            no_inst = False
            if len(inst_ids) == 0:
                no_inst = True
            else:
                selected_inst_id = random.choice(inst_ids)
        
        lab_ori = lab.copy()
        
        lab_ids = np.unique(lab)
        lab_ids = lab_ids.tolist()
        selected_lab_id = random.choice(lab_ids)
        
        img = get_transform(img)
        lab = get_transform(lab, normalize=False)
        lab = lab * 255.0
        
        if self.split == 'train':
            mask_type = index % 5
            if self.dataset_name == 'ADE20k-landscape':
                mask_type = index % 4
            # 
            if mask_type == 0:
                mask = brush_stroke_mask()
                mask = mask.reshape((1,) + mask.shape).astype(np.float32)
            elif mask_type == 1:
                mask = self.load_right_mask(self.cfg['crop_size'])
            elif mask_type == 2:
                mask = self.load_center_mask(self.cfg['crop_size'], split='train')
            elif mask_type == 3:
                mask = np.array(np.equal(lab_ori, selected_lab_id).astype(np.uint8))
                mask = mask.reshape((1,) + mask.shape).astype(np.float32)
            elif mask_type == 4:
                if not no_inst:
                    mask = np.zeros((256, 256), np.float32)
                    ys,xs = np.where(inst_map==selected_inst_id)
                    ymin, ymax, xmin, xmax = ys.min(), ys.max(), xs.min(), xs.max()
                    mask[ymin:ymax, xmin:xmax] = 1
                    mask = mask.reshape((1,) + mask.shape).astype(np.float32)
                else:
                    mask = brush_stroke_mask()
                    mask = mask.reshape((1,) + mask.shape).astype(np.float32)
        
        else:
            mask = cv2.imread(os.path.join(self.predefined_mask_path, self.mask_list[index]), 0) / 255
            mask = mask.reshape((1,) + mask.shape).astype(np.float32)
        
        mask = torch.from_numpy(mask)
        masked_img = img * (1. - mask)

        inst_map = inst_map.reshape((1,) + inst_map.shape).astype(np.float32)
        inst_map = torch.from_numpy(inst_map)
        
        return {'img': img, 'masked_img': masked_img, 'lab': lab, 'mask': mask, 'inst_map': inst_map, 'name': name}
        # 'mask_seam': mask_seam,

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.name_list)

    def load_center_mask(self, crop_size, split):   
        # rect
        height, width = crop_size, crop_size
        mask = np.ones((height, width), np.float32)
        if split == 'test':
            mask[64:192, 64:192] = 0.
            w1 = 64
            w2 = 64 + 128
            h1 = 64
            h2 = 64 + 128
        else:
            w1 = random.randint(32, 96)
            w2 = w1 + 128
            h1 = random.randint(32, 96)
            h2 = h1 + 128
            mask[h1:h2, w1:w2] = 0.  # edited region is 1, non-edited region is 0
        mask = mask.reshape((1,) + mask.shape).astype(np.float32)

        return mask

    def load_right_mask(self, img_shapes, mask_rate=0.5):
        height, width = img_shapes, img_shapes
        mask = np.zeros((height, width), np.float32)

        mask_length = int(width * mask_rate)  # masked length
        w1 = width - mask_length
        mask[:, w1:] = 1.  # edited region is 1, non-edited region is 0
        mask = mask.reshape((1,) + mask.shape).astype(np.float32)

        return mask

    def load_seam_mask(self, img_shapes, box):
        m = 16
        height, width = img_shapes, img_shapes
        mask1 = np.ones((height, width), np.float32)
        mask2 = np.zeros((height, width), np.float32)
        
        mask1[box[0]+m:box[1]-m, box[2]+m:box[3]-m] = 0.
        mask2[box[0]-m:box[1]+m, box[2]-m:box[3]+m] = 1.
        
        mask = mask1 * mask2
        mask = mask.reshape((1,) + mask.shape).astype(np.float32)

        return torch.from_numpy(mask)


def get_transform(img, normalize=True):
    transform_list = []

    transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)(img)

def brush_stroke_mask(H=256, W=256):
    min_num_vertex = 4
    max_num_vertex = 8
    mean_angle = 2 * math.pi / 5
    angle_range = 2 * math.pi / 15
    min_width = 50
    max_width = 140

    average_radius = math.sqrt(H * H + W * W) / 8
    mask = Image.new('L', (W, H), 0)

    num_vertex = random.randint(min_num_vertex, max_num_vertex)
    angle_min = mean_angle - random.uniform(0, angle_range)
    angle_max = mean_angle + random.uniform(0, angle_range)
    angles = []
    vertex = []
    for i in range(num_vertex):
        if i % 2 == 0:
            angles.append(2 * math.pi - random.uniform(angle_min, angle_max))
        else:
            angles.append(random.uniform(angle_min, angle_max))

    h, w = mask.size
    vertex.append((int(random.randint(0, w)), int(random.randint(0, h))))
    for i in range(num_vertex):
        r = np.clip(
            np.random.normal(loc=average_radius, scale=average_radius // 2),
            0, 2 * average_radius)
        new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
        new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
        vertex.append((int(new_x), int(new_y)))

    draw = ImageDraw.Draw(mask)
    width = int(random.uniform(min_width, max_width))
    draw.line(vertex, fill=1, width=width)
    for v in vertex:
        draw.ellipse((v[0] - width // 2,
                      v[1] - width // 2,
                      v[0] + width // 2,
                      v[1] + width // 2),
                     fill=1)

    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_TOP_BOTTOM)
    
    mask = np.asarray(mask, np.float32)

    return mask

def get_mask_edge(mask):
    edge = cv2.Canny(mask, 0, 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(20, 20))
    edge_mask = cv2.dilate(edge,kernel)
    
    return edge_mask