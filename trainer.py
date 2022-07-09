from networks.generator import Generator
from networks.SNPatchDiscriminator import SNPatchDiscriminator
from utils import get_scheduler, weights_init, save_network, save_latest_network, get_model_list
import torch
import torch.nn as nn
import torch.nn.functional as F
import loss
import os

class SPMPGAN_Trainer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # setting basic params
        self.cfg = cfg
        if self.cfg['is_train']:
            self.model_names = ['netG', 'netD_1', 'netD_2', 'netD_3']
        else:  # during test time, only load G
            self.model_names = ['G']

        # Initiate the submodules and initialization params
        self.netG = Generator(self.cfg)
        self.netD_1 = SNPatchDiscriminator(self.cfg, num_layer=4)
        self.netD_2 = SNPatchDiscriminator(self.cfg, num_layer=5)
        self.netD_3 = SNPatchDiscriminator(self.cfg, num_layer=6)

        self.netG.apply(weights_init('gaussian'))
        self.netD_1.apply(weights_init('gaussian'))
        self.netD_2.apply(weights_init('gaussian'))
        self.netD_3.apply(weights_init('gaussian'))

        self.FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if torch.cuda.is_available() \
            else torch.ByteTensor

        # Setup the optimizers and schedulers
        if cfg['is_train']:
            lr = self.cfg['lr']
            beta1 = self.cfg['beta1']
            beta2 = self.cfg['beta2']
            # set optimizers
            self.optimizers = []
            G_params = list(self.netG.parameters())
            D_1_params = list(self.netD_1.parameters())
            D_2_params = list(self.netD_2.parameters())
            D_3_params = list(self.netD_3.parameters())

            self.optimizer_G = torch.optim.Adam(G_params, lr=lr, betas=(beta1, beta2))
            self.optimizer_D_1 = torch.optim.Adam(D_1_params, lr=lr*4, betas=(beta1, beta2))
            self.optimizer_D_2 = torch.optim.Adam(D_2_params, lr=lr*4, betas=(beta1, beta2))
            self.optimizer_D_3 = torch.optim.Adam(D_3_params, lr=lr*4, betas=(beta1, beta2))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_1)
            self.optimizers.append(self.optimizer_D_2)
            self.optimizers.append(self.optimizer_D_3)
            self.opt_names = ['optimizer_G', 'optimizer_D_1', 'optimizer_D_2', 'optimizer_D_3']

            # set schedulers
            # self.schedulers = [get_scheduler(optimizer, self.cfg) for optimizer in self.optimizers]
            # set criterion
            self.criterionGAN = loss.GANLoss(cfg['gan_mode']).cuda()
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionVGG = loss.VGGLoss()
            self.criterionFeat = torch.nn.L1Loss()

        self.G_losses = {}
        self.D_losses = {}

######################################################################################
    def set_input(self, input):
        # scatter_ require .long() type
        input['lab'] = input['lab'].long()
        self.masked_img = input['masked_img'].cuda()     # mask image
        self.gt = input['img'].cuda()        # real image
        # self.img_know = input['img_know'].cuda()
        self.mask = input['mask'].cuda()    # mask image
        self.lab = input['lab'].cuda()  # label image

        self.name = input['name']

        # create one-hot label map
        lab_map = self.lab
        bs, _, h, w = lab_map.size()
        nc = self.cfg['lab_dim']
        input_label = self.FloatTensor(bs, nc, h, w).zero_()
        self.segmap = input_label.scatter_(1, lab_map, 1.0)
        # print(' segmap ',self.lab.shape)
        
        self.segmap = self.segmap * self.mask

        self.inst_map = input['inst_map'].cuda()
        self.edge_map = self.get_edges(self.inst_map)
        self.edge_map = self.edge_map * self.mask
        
        self.segmap_edge = torch.cat((self.segmap, self.edge_map), dim=1)

        self.segmap_G1 = F.interpolate(self.segmap, size=(64, 64), mode='nearest')
        self.segmap_G2 = F.interpolate(self.segmap, size=(128, 128), mode='nearest')
        self.segmap_G3 = self.segmap


    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        
        return edge.float()

    def forward(self):
        gt_list, input_list, mask_fake_list, fake_list = self.netG(self.gt, self.masked_img, self.segmap_edge, self.mask)

        self.gt_G1, self.gt_G2, self.gt_G3 = gt_list
        self.input_G1, self.input_G2, self.input_G3 = input_list
        self.mask_fake_G1, self.mask_fake_G2, self.mask_fake_G3 = mask_fake_list
        self.fake_G1, self.fake_G2, self.fake_G3 = fake_list


    def test(self, gt, masked_img, segmap, mask):
        with torch.no_grad():
            _, _, mask_fake_list, _ = self.netG(gt, masked_img, segmap, mask)
            _, _, mask_fake_G3 = mask_fake_list
            
            return mask_fake_G3


    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        # Fake
        fake = torch.cat([self.segmap_G1, self.mask_fake_G1.detach()], dim=1)
        pred_fake = self.netD_1(fake)
        self.D_losses['loss_D_fake_G1'] = self.criterionGAN(pred_fake, False, for_discriminator=True) * self.cfg['lambda_gan']
        # Real
        real = torch.cat([self.segmap_G1, self.gt_G1], dim=1)
        pred_real = self.netD_1(real)
        self.D_losses['loss_D_real_G1'] = self.criterionGAN(pred_real, True, for_discriminator=True) * self.cfg['lambda_gan']

        # Fake
        fake = torch.cat([self.segmap_G2, self.mask_fake_G2.detach()], dim=1)
        pred_fake = self.netD_2(fake)
        self.D_losses['loss_D_fake_G2'] = self.criterionGAN(pred_fake, False, for_discriminator=True) * self.cfg['lambda_gan']
        # Real
        real = torch.cat([self.segmap_G2, self.gt_G2], dim=1)
        pred_real = self.netD_2(real)
        self.D_losses['loss_D_real_G2'] = self.criterionGAN(pred_real, True, for_discriminator=True) * self.cfg['lambda_gan']

        # Fake
        fake = torch.cat([self.segmap_G3, self.mask_fake_G3.detach()], dim=1)
        pred_fake = self.netD_3(fake)
        self.D_losses['loss_D_fake_G3'] = self.criterionGAN(pred_fake, False, for_discriminator=True) * self.cfg['lambda_gan']
        # Real
        real = torch.cat([self.segmap_G3, self.gt_G3], dim=1)
        pred_real = self.netD_3(real)
        self.D_losses['loss_D_real_G3'] = self.criterionGAN(pred_real, True, for_discriminator=True) * self.cfg['lambda_gan']

        return self.D_losses

    def compute_G_loss(self):
        """Calculate losses for the generator"""
        # L1 loss
        self.G_losses['L1_G1'] = torch.mean(torch.abs(self.fake_G1 - self.gt_G1)) * self.cfg['lambda_L1']
        self.G_losses['L1_G2'] = torch.mean(torch.abs(self.fake_G2 - self.gt_G2)) * self.cfg['lambda_L1']
        self.G_losses['L1_G3'] = torch.mean(torch.abs(self.fake_G3 - self.gt_G3)) * self.cfg['lambda_L1']
        
        # GAN loss
        fake_global = torch.cat([self.segmap_G1, self.mask_fake_G1], dim=1)
        pred_fake_global = self.netD_1(fake_global)
        self.G_losses['G_GAN_G1'] = self.criterionGAN(pred_fake_global, True, for_discriminator=False) * self.cfg['lambda_gan']

        fake_global = torch.cat([self.segmap_G2, self.mask_fake_G2], dim=1)
        pred_fake_global = self.netD_2(fake_global)
        self.G_losses['G_GAN_G2'] = self.criterionGAN(pred_fake_global, True, for_discriminator=False) * self.cfg['lambda_gan']

        fake_global = torch.cat([self.segmap_G3, self.mask_fake_G3], dim=1)
        pred_fake_global = self.netD_3(fake_global)
        self.G_losses['G_GAN_G3'] = self.criterionGAN(pred_fake_global, True, for_discriminator=False) * self.cfg['lambda_gan']

        # VGG loss
        if not self.cfg['no_vgg_loss']:
            self.G_losses['VGG_G1'] = self.criterionVGG(self.fake_G1, self.gt_G1) * self.cfg['lambda_vgg']
            self.G_losses['VGG_G2'] = self.criterionVGG(self.fake_G2, self.gt_G2) * self.cfg['lambda_vgg']
            self.G_losses['VGG_G3'] = self.criterionVGG(self.fake_G3, self.gt_G3) * self.cfg['lambda_vgg']

        return self.G_losses


    def optimize_parameters(self):
        self.forward()

        # update global D
        self.set_requires_grad(self.netD_1, True)  # enable backprop for D
        self.set_requires_grad(self.netD_2, True)
        self.set_requires_grad(self.netD_3, True)
        self.optimizer_D_1.zero_grad()  # set D's gradients to zero
        self.optimizer_D_2.zero_grad()
        self.optimizer_D_3.zero_grad()
        self.d_losses = self.compute_D_loss()
        d_loss = sum(self.d_losses.values()).mean()
        d_loss.backward()
        self.optimizer_D_1.step()
        self.optimizer_D_2.step()
        self.optimizer_D_3.step()

        # update G
        self.set_requires_grad(self.netD_1, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netD_2, False)
        self.set_requires_grad(self.netD_3, False)
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.g_losses = self.compute_G_loss()
        g_loss = sum(self.g_losses.values()).mean()
        g_loss.backward()
        self.optimizer_G.step()             # udpate G's weights

#########################################################################################################
########## util func #############
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def visual_results(self):
        return {'masked_img': self.masked_img, 'gt': self.gt, 'lab': self.lab, 'mask': self.mask,
                'gt_G1': self.gt_G1, 'gt_G2': self.gt_G2, 'gt_G3': self.gt_G3,
                'input_G1': self.input_G1, 'input_G2': self.input_G2, 'input_G3': self.input_G3,
                'mask_fake_G1': self.mask_fake_G1, 'mask_fake_G2': self.mask_fake_G2, 'mask_fake_G3': self.mask_fake_G3,
                'fake_G1': self.fake_G1, 'fake_G2': self.fake_G2, 'fake_G3': self.fake_G3, 
                'edge_map': self.edge_map, }, self.name

    def print_losses(self):
        print('G Losses')
        for v,k in self.G_losses.items():
            print(v, ': ', k)
            
        print('D Losses')
        for v,k in self.D_losses.items():
            print(v, ': ', k)


    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def save_nets(self, epoch, cfg, suffix=''):
        save_file = {}
        save_file['epoch'] = epoch
        for name in self.model_names:
            net = getattr(self, name)
            save_file[name] = net.cpu().state_dict()
            if torch.cuda.is_available():
                net.cuda()
        for name in self.opt_names:
            opt = getattr(self, name)
            save_file[name] = opt.state_dict()
        save_filename = '%03d_ckpt_%s.pth' % (epoch, suffix)
        save_path = os.path.join(cfg['checkpoints_dir'], save_filename)
        torch.save(save_file, save_path)

    def save_latest_nets(self, epoch, cfg):
        save_file = {}
        save_file['epoch'] = epoch
        for name in self.model_names:
            net = getattr(self, name)
            save_file[name] = net.cpu().state_dict()
            if torch.cuda.is_available():
                net.cuda()
        for name in self.opt_names:
            opt = getattr(self, name)
            save_file[name] = opt.state_dict()
        save_filename = 'latest_ckpt.pth'
        save_path = os.path.join(cfg['checkpoints_dir'], save_filename)
        torch.save(save_file, save_path)

    def print_networks(self):
        """Print the total number of parameters in the network and network architecture"""
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                # print network architecture
                print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)


    def resume(self, checkpoint_dir, ckpt_filename=None):
        if not ckpt_filename:
            ckpt_filename = 'latest_ckpt.pth'
        ckpt = torch.load(os.path.join(checkpoint_dir, ckpt_filename))
        cur_epoch = ckpt['epoch']
        for name in self.model_names:
            net = getattr(self, name)
            net.load_state_dict(ckpt[name])
            print('load model %s of epoch %d' % (name, cur_epoch))
        for name in self.opt_names:
            opt = getattr(self, name)
            opt.load_state_dict(ckpt[name])
            print('load opt %s of epoch %d' % (name, cur_epoch))
        return cur_epoch