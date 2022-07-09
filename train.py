from utils import *
import argparse
import numpy as np
import datetime
from trainer import SPMPGAN_Trainer
from dataset import Image_Editing_Dataset
import torch
from torch.utils.data import DataLoader
import os
import shutil
import cv2
from metrics.fid_score import calculate_fid_given_paths


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument('--dataset_name', type=str, default='ADE20k-room', help="dataset name")
parser.add_argument("--resume", action="store_true")
parser.add_argument('--resume_dir', type=str, default='', help="outputs path")
opts = parser.parse_args()

print_options(opts)

# cudnn.benchmark = True

# GPU
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

# Load experiment setting
cfg = get_config(opts.config)
# datasets setting
if opts.dataset_name == 'ADE20k-room':
    cfg['lab_dim'] = 151
    cfg['max_epoch'] = 500
    cfg['test_freq'] = 20
elif opts.dataset_name == 'ADE20k-landscape':
    cfg['lab_dim'] = 151
    cfg['max_epoch'] = 500
    cfg['test_freq'] = 20
elif opts.dataset_name == 'cityscapes':
    cfg['lab_dim'] = 34
    cfg['max_epoch'] = 500
    cfg['test_freq'] = 20


trainer = SPMPGAN_Trainer(cfg)
trainer.cuda()

# print model information
trainer.print_networks()

# Setup dataset
dataset_root = os.path.join(cfg['dataset_dir'], opts.dataset_name)
train_dataset = Image_Editing_Dataset(cfg, dataset_root, split='train', dataset_name=opts.dataset_name)
train_loader = DataLoader(dataset=train_dataset, batch_size=cfg['batch_size'], shuffle=cfg['shuffle'], num_workers=cfg['worker'])
test_dataset = Image_Editing_Dataset(cfg, dataset_root, split='test', dataset_name=opts.dataset_name)
test_loader = DataLoader(dataset=test_dataset, batch_size=cfg['test_batch_size'], shuffle=False, num_workers=cfg['worker'])
print('train dataset containing ', len(train_loader), 'images')
print('test dataset containing ', len(test_loader), 'images')

# Setup logger and output folders
if opts.resume:
    checkpoint_directory = opts.resume_dir + 'checkpoints/'
    image_directory = opts.resume_dir + 'images/'
    result_directory = opts.resume_dir + 'results/'
    cur_epoch = trainer.resume(checkpoint_directory, ckpt_filename=None) + 1
    shutil.copy(opts.config, os.path.join(opts.resume_dir, 'config_resume.yaml'))  # copy config file to output folder
else:
    cur_epoch = 0
    output_directory = os.path.join(opts.output_path + "/outputs", opts.dataset_name,
                                    datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S'))
    checkpoint_directory, image_directory, result_directory = prepare_sub_folder(output_directory)
    shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))  # copy config file to output folder

cfg['checkpoints_dir'] = checkpoint_directory

best_fid = float("inf")
print('training start at %d th epoch' % (cur_epoch))
# Start training
for epoch in range(cur_epoch, cfg['max_epoch']):
    for i, data in enumerate(train_loader):  # inner loop within one epoch
        trainer.train()
        trainer.set_input(data)  # unpack data from dataset and apply preprocessing
        trainer.optimize_parameters()  # calculate loss functions, get gradients, update network weights

        if i % cfg['visual_img_freq'] == 0:
            results, img_name = trainer.visual_results()

            cur_img_dir = os.path.join(image_directory, 'epoch-%d_iter-%d_%s'%(epoch, i, img_name[0]))
            if not os.path.exists(cur_img_dir):
                os.makedirs(cur_img_dir)

            is_fg = ['mask', 'lab', 'mask_seam', 'edge_map']
            for name, img in results.items():
                no_fg = True
                if name in is_fg:
                    no_fg = False
                save_name = 'epoch-%d_iter-%d_%s_'%(epoch, i, name)+'.png'
                if name == 'lab':
                    lab = lab2im(img)
                    cv2.imwrite(os.path.join(cur_img_dir, save_name), lab)
                    # print('lab mean: ', lab2im(img).mean())
                    label_dir = os.path.join(cur_img_dir, 'label')
                    if not os.path.exists(label_dir):
                        os.makedirs(label_dir)
                    set = np.unique(lab)
                    for l in set:
                        cur_lab = np.array(np.equal(lab, l).astype(np.uint8)) * 255
                        cv2.imwrite(os.path.join(label_dir, str(l)+'.png'), cur_lab)
                elif name == 'att':
                    att_dir = os.path.join(cur_img_dir, 'att')
                    if not os.path.exists(att_dir):
                        os.makedirs(att_dir)
                    for i_att, att in enumerate(img):
                        hm = tensor2hm(att)
                        cv2.imwrite(os.path.join(att_dir, 'att_'+str(i_att)+'.png'), cv2.applyColorMap(hm, cv2.COLORMAP_JET))
                elif name == 'middle_avg':
                    cv2.imwrite(os.path.join(cur_img_dir, save_name), tensor2im(img[:,:3,:,:], no_fg=no_fg))
                elif name == 'masks':
                    masks_dir = os.path.join(cur_img_dir, 'masks')
                    if not os.path.exists(masks_dir):
                        os.makedirs(masks_dir)
                    for idx, m in enumerate(img):
                        hm = tensor2hm(m)
                        cv2.imwrite(os.path.join(masks_dir, 'att_'+str(idx)+'.png'), cv2.applyColorMap(hm, cv2.COLORMAP_JET))
                elif name in ['middle_avg_encoder', 'middle_avg_decoder']:
                    cv2.imwrite(os.path.join(cur_img_dir, save_name), tensor2im(img[:,:3,:,:], no_fg=no_fg))

                else:
                    cv2.imwrite(os.path.join(cur_img_dir, save_name), tensor2im(img, no_fg=no_fg))		

        if i % cfg['print_loss_freq'] == 0:  # print training losses and save logging information to the disk
            print('print losses at the {} epoch {} iter'.format(epoch, i))
            trainer.print_losses()

    if (epoch+1) % cfg['save_epoch_freq'] == 0:              # cache our model every <save_epoch_freq> epochs
        print('saving the model at the end of epoch %d' % (epoch))
        trainer.save_nets(epoch, cfg)

    print('saving the model')
    trainer.save_latest_nets(epoch, cfg)

    # updating learning rate
    # trainer.update_learning_rate()

    # compute FID score
    if epoch % cfg['test_freq'] == 0:
        with torch.no_grad():
            print('testing at %d epoch' % (epoch))
            trainer.eval()
            cur_save_dir = os.path.join(result_directory, str(epoch))
            if not os.path.exists(cur_save_dir):
                os.makedirs(cur_save_dir)
            for i, data in enumerate(test_loader):
                trainer.set_input(data)
                trainer.forward()
                results, img_name = trainer.visual_results()
                # if not os.path.exists(os.path.join(cur_save_dir, img_name[0]+'.png')):
                    # os.makedirs(os.path.join(cur_save_dir, img_name[0]+'.png'))
                cv2.imwrite(os.path.join(cur_save_dir, img_name[0]+'.png'), tensor2im(results['mask_fake_G3']))

            path_gt = os.path.join(cfg['dataset_dir'], opts.dataset_name, 'test', 'images')
            path_test = cur_save_dir
            print('path_gt: ', path_gt)
            print('path_test: ', path_test)

            # compute FID score
            path = [path_gt, path_test]
            fid_value = calculate_fid_given_paths(path, cfg['test_batch_size'])
            print('========FID==========: ', fid_value)

            if fid_value < best_fid:
                print('saving the current best model at the end of epoch %d' % (epoch))
                trainer.save_nets(epoch, cfg, suffix='_best_FID_'+str(fid_value))
                best_fid = fid_value

