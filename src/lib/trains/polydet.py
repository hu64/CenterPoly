from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from models.losses import FocalLoss
from models.losses import RegL1Loss, RegLoss, RegL1PolyLoss, AreaPolyLoss, NormRegL1Loss, RegWeightedL1Loss
from models.decode import polydet_decode
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import polydet_post_process
from utils.oracle_utils import gen_oracle_map
from .base_trainer import BaseTrainer
import cv2
from utils.image import get_affine_transform, affine_transform

class PolydetLoss(torch.nn.Module):
    def __init__(self, opt):
        super(PolydetLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_poly = RegL1PolyLoss()
        self.crit_dense_poly = torch.nn.L1Loss(reduction='sum')
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.area_poly = AreaPolyLoss()
        self.opt = opt

    def forward(self, outputs, batch):
        opt = self.opt
        hm_loss, off_loss, poly_loss, depth_loss, wh_loss, fg_loss = 0, 0, 0, 0, 0, 0
        # hm_loss, off_loss, poly_loss, depth_loss, border_hm_loss = 0, 0, 0, 0, 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])
                # output['fg'] = _sigmoid(output['fg'])
                # output['border_hm'] = _sigmoid(output['border_hm'])
            if opt.eval_oracle_hm:
                output['hm'] = batch['hm']
            # if opt.eval_oracle_fg:
            #     output['fg'] = batch['fg']
            if opt.eval_oracle_border_hm:
                output['border_hm'] = batch['border_hm']
            if opt.eval_oracle_offset:
                output['reg'] = torch.from_numpy(gen_oracle_map(
                    batch['reg'].detach().cpu().numpy(),
                    batch['ind'].detach().cpu().numpy(),
                    output['reg'].shape[3], output['reg'].shape[2])).to(opt.device)
            if opt.eval_oracle_poly:
                # output['poly'] = batch['poly']
                output['poly'] = torch.from_numpy(gen_oracle_map(
                    batch['poly'].detach().cpu().numpy(),
                    batch['ind'].detach().cpu().numpy(),
                    output['poly'].shape[3], output['poly'].shape[2])).to(opt.device)
            if opt.eval_oracle_pseudo_depth:
                output['pseudo_depth'] = torch.from_numpy(gen_oracle_map(
                    batch['pseudo_depth'].detach().cpu().numpy(),
                    batch['ind'].detach().cpu().numpy(),
                    output['pseudo_depth'].shape[3], output['pseudo_depth'].shape[2])).to(opt.device)
            # if opt.eval_oracle_wh:
            #     output['wh'] = torch.from_numpy(gen_oracle_map(
            #         batch['wh'].detach().cpu().numpy(),
            #         batch['ind'].detach().cpu().numpy(),
            #         output['wh'].shape[3], output['wh'].shape[2])).to(opt.device)

            # wh_loss += self.crit_reg(
            #     output['wh'], batch['reg_mask'],
            #     batch['ind'], batch['wh']) / opt.num_stacks

            depth_loss += self.crit_reg(output['pseudo_depth'], batch['reg_mask'],
                                          batch['ind'], batch['pseudo_depth']) / opt.num_stacks

            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            # fg_loss += self.crit(output['fg'], batch['fg']) / opt.num_stacks
            # border_hm_loss += self.crit(output['border_hm'], batch['border_hm']) / opt.num_stacks
            # area_loss += self.area_poly(output['poly'], batch['reg_mask'], batch['ind'], batch['instance'], batch['centers']) / opt.num_stacks
            # print(output['poly'].shape)
            if opt.cat_spec_poly:
                poly_loss += self.crit_poly(
                    output['poly'], batch['cat_spec_mask'],
                    batch['ind'], batch['cat_spec_poly']) / opt.num_stacks
            elif opt.dense_poly:
                mask_weight = batch['dense_poly_mask'].sum() + 1e-4
                poly_loss += (self.crit_dense_poly(output['poly'] * batch['dense_poly_mask'],
                                                   batch['dense_poly'] * batch['dense_poly_mask']) / mask_weight)\
                             / opt.num_stacks
            else:
                poly_loss += self.crit_poly(output['poly'], batch[
                    'reg_mask'], batch['ind'], batch['poly'], freq_mask = batch['freq_mask']) / opt.num_stacks
                # poly_loss += self.crit_poly(output['poly'], batch['reg_mask'],  # batch['freq_mask'],batch['ind'], batch['poly']) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks

        # import cv2
        # import os
        # write_depth = np.array(output['fg'][0, :, :, :].cpu().detach().squeeze(0).squeeze(0))
        # # print(write_depth.shape)
        # write_depth = (((write_depth - np.min(write_depth)) / np.max(write_depth)) * 255).astype(np.uint8)
        # count = 0
        # write_name = '/store/datasets/cityscapes/test_images/depth/depth' + str(count) + '.jpg'
        # while os.path.exists(write_name):
        #     count += 1
        #     write_name = '/store/datasets/cityscapes/test_images/depth/depth' + str(count) + '.jpg'
        # cv2.imwrite('/store/datasets/cityscapes/test_images/depth/depth' + str(count) + '.jpg', write_depth)
        # exit()

        # loss = opt.hm_weight * hm_loss + opt.off_weight * off_loss + opt.poly_weight * poly_loss + opt.depth_weight * depth_loss
        loss = opt.hm_weight * hm_loss + opt.off_weight * off_loss + opt.poly_weight * poly_loss \
               + opt.depth_weight * depth_loss # + fg_loss #  + opt.wh_weight * wh_loss #  + opt.border_hm_weight * border_hm_loss
        # loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'off_loss': off_loss, 'poly_loss': poly_loss, 'depth_loss': depth_loss, 'border_hm_loss': border_hm_loss}
        loss_stats = {'loss': loss, 'hm_l': hm_loss, 'off_l': off_loss, 'poly_l': poly_loss,
                      'depth_l': depth_loss,
                      #  'fg_l': fg_loss,
                      # 'wh_l': wh_loss,
                      }
        return loss, loss_stats


class PolydetTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(PolydetTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        # loss_states = ['loss', 'hm_loss', 'off_loss', 'poly_loss', 'depth_loss', 'border_hm_loss']
        # loss_states = ['loss', 'hm_l', 'off_l', 'poly_l', 'depth_l', 'wh_l']
        loss_states = ['loss', 'hm_l', 'off_l', 'poly_l', 'depth_l']
        loss = PolydetLoss(opt)
        return loss_states, loss

    def debug(self, batch, output, iter_id):
        opt = self.opt
        reg = output['reg'] if opt.reg_offset else None
        dets = polydet_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_poly=opt.cat_spec_poly, K=opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets[:, :, :4] *= opt.down_ratio
        dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
        dets_gt[:, :, :4] *= opt.down_ratio
        for i in range(1):
            debugger = Debugger(
                dataset=opt.dataset, ipynb=(opt.debug == 3), theme=opt.debugger_theme)
            img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
            img = np.clip(((
                                   img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
            pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
            gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm')
            debugger.add_blend_img(img, gt, 'gt_hm')
            debugger.add_img(img, img_id='out_pred')
            for k in range(len(dets[i])):
                if dets[i, k, 4] > opt.center_thresh:
                    debugger.add_coco_bbox(dets[i, k, :4], dets[i, k, -1],
                                           dets[i, k, 4], img_id='out_pred')

            debugger.add_img(img, img_id='out_gt')
            for k in range(len(dets_gt[i])):
                if dets_gt[i, k, 4] > opt.center_thresh:
                    debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1],
                                           dets_gt[i, k, 4], img_id='out_gt')

            if opt.debug == 4:
                debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
            else:
                debugger.show_all_imgs(pause=True)

    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = polydet_decode(
            output['hm'], output['poly'], output['pseudo_depth'], reg=reg,
            cat_spec_poly=self.opt.cat_spec_poly, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = polydet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])

        # fg = np.array(output['fg'].cpu().detach().squeeze(0).squeeze(0))
        # if np.max(fg) != 0:
        #     fg = (fg - np.min(fg)) / np.max(fg)
        # dets_out[0]['fg'] = fg
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]
        # results[str(batch['meta']['img_id'].cpu().numpy()[0])+'_fg'] = fg