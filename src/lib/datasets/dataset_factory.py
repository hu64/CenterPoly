from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.ddd import DddDataset
from .sample.exdet import EXDetDataset
from .sample.ctdet import CTDetDataset
from .sample.polydet import PolydetDataset
from .sample.multi_pose import MultiPoseDataset

from .dataset.coco import COCO
from .dataset.pascal import PascalVOC
from .dataset.kitti import KITTI
from .dataset.kitti2d import KITTI2D
from .dataset.kitti_poly import KITTIPOLY
from .dataset.coco_hp import COCOHP
from .dataset.uadetrac import UADETRAC
from .dataset.uadetrac1on10 import UADETRAC1ON10
from .dataset.uadetrac1on10_b import UADETRAC1ON10_b
from .dataset.uav import UAV
from .dataset.cityscapes import CITYSCAPES
from .dataset.IDD import IDD


dataset_factory = {
  'coco': COCO,
  'pascal': PascalVOC,
  'kitti': KITTI,
  'kitti2d': KITTI2D,
  'kitti_poly': KITTIPOLY,
  'coco_hp': COCOHP,
  'uadetrac': UADETRAC,
  'uadetrac1on10': UADETRAC1ON10,
  'uadetrac1on10_b': UADETRAC1ON10_b,
  'uav': UAV,
  'cityscapes': CITYSCAPES,
  'IDD': IDD,

}

_sample_factory = {
  'exdet': EXDetDataset,
  'ctdet': CTDetDataset,
  'polydet': PolydetDataset,
  'ddd': DddDataset,
  'multi_pose': MultiPoseDataset
}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset
  
