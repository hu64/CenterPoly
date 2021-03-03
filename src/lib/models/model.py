from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from .networks.msra_resnet import get_pose_net
from .networks.dlav0 import get_pose_net as get_dlav0
from .networks.large_hourglass import get_large_hourglass_net, get_small_hourglass_net



_model_factory = {
  'res': get_pose_net, # default Resnet with deconv
  'dlav0': get_dlav0, # default DLAup
  'hourglass': get_large_hourglass_net,
  'smallhourglass': get_small_hourglass_net,
}


def create_model(arch, heads, head_conv):
  num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
  arch = arch[:arch.find('_')] if '_' in arch else arch
  get_model = _model_factory[arch]
  model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv)
  return model


def load_model(model, model_path, optimizer=None, resume=False,
               lr=None, lr_step=None):
  start_epoch = 0
  checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
  print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
  state_dict_ = checkpoint['state_dict']
  state_dict = {}

  # convert data_parallal to model
  for k in state_dict_:
    if k.startswith('module') and not k.startswith('module_list'):
      state_dict[k[7:]] = state_dict_[k]
    else:
      state_dict[k] = state_dict_[k]
  model_state_dict = model.state_dict()

  # check loaded parameters and created model parameters
  msg = 'If you see this, your model does not fully load the ' + \
        'pre-trained weight. Please make sure ' + \
        'you have correctly specified --arch xxx ' + \
        'or set the correct --num_classes for your own dataset.'
  for k in state_dict:
    if k in model_state_dict:
      if state_dict[k].shape != model_state_dict[k].shape:
        print('Skip loading parameter {}, required shape{}, ' \
              'loaded shape{}. {}'.format(
          k, model_state_dict[k].shape, state_dict[k].shape, msg))
        state_dict[k] = model_state_dict[k]
    else:
      print('Drop parameter {}.'.format(k) + msg)
  for k in model_state_dict:
    if not (k in state_dict):
      print('No param {}.'.format(k) + msg)
      state_dict[k] = model_state_dict[k]

  EXT_D = False
  if EXT_D:
    D_W = torch.load('../exp/cityscapes/polydet/resnet18_32pts_2/model_best.pth',
                     map_location=lambda storage, loc: storage)
    d_state_dict = D_W['state_dict']
    for k in d_state_dict:
      if 'depth' in k:
        print('depth: ', k)
        model_state_dict[k] = d_state_dict[k]
        state_dict[k] = d_state_dict[k]
  EXT_Poly = False
  if EXT_Poly:
    Poly_W = torch.load('../exp/cityscapes/polydet/newgt_pw10_lr2e4/model_best.pth',
                        map_location=lambda storage, loc: storage)
    poly_state_dict = Poly_W['state_dict']
    for k in poly_state_dict:
      if 'poly' in k or 'cnvs' in k:
        print('poly: ', k)
        model_state_dict[k] = poly_state_dict[k]
        state_dict[k] = poly_state_dict[k]

  loaded_state_dict = state_dict.copy()

  model.load_state_dict(state_dict, strict=False)

  # resume optimizer parameters
  if optimizer is not None and resume:
    if 'optimizer' in checkpoint:
      optimizer.load_state_dict(checkpoint['optimizer'])
      start_epoch = checkpoint['epoch']
      start_lr = lr
      for step in lr_step:
        if start_epoch >= step:
          start_lr *= 0.1
      for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr
      print('Resumed optimizer with start lr', start_lr)
    else:
      print('No optimizer parameters in checkpoint.')

  FREEZE_LAYERS = False
  if FREEZE_LAYERS:
    for name, param in model.named_parameters():
      if name in loaded_state_dict and not 'poly' in name and not 'depth' in name:
        # print('Freeze: ', name)
        param.requires_grad = False
        param.freeze = True
      else:
        print('Not freezing: ', name)
        param.freeze = False

  if optimizer is not None:
    return model, optimizer, start_epoch
  else:
    return model


def save_model(path, epoch, model, optimizer=None):
  if isinstance(model, torch.nn.DataParallel):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  data = {'epoch': epoch,
          'state_dict': state_dict}
  if not (optimizer is None):
    data['optimizer'] = optimizer.state_dict()
  torch.save(data, path)

