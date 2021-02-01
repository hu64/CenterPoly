from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .exdet import ExdetDetector
from .ddd import DddDetector
from .ctdet import CtdetDetector
from .ctdetSpotNet2 import CtdetDetectorSpotNet2
from .ctdetMultiSpot import CtdetDetectorMultiSpot

from .ctdetVid import CtdetDetectorVid
from .ctdetSpotNetVid import CtdetDetectorSpotNetVid
from .polydet import CtdetWipeNetDetector
from .multi_pose import MultiPoseDetector

detector_factory = {
  'exdet': ExdetDetector, 
  'ddd': DddDetector,
  'ctdet': CtdetDetector,
  'ctdetSpotNet2': CtdetDetectorSpotNet2,
  'ctdetMultiSpot': CtdetDetectorMultiSpot,
  'ctdetVid': CtdetDetectorVid,
  'ctdetSpotNetVid': CtdetDetectorSpotNetVid,
  'ctdetWipeNet': CtdetWipeNetDetector,
  'multi_pose': MultiPoseDetector,
}
