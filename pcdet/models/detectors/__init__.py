from .PartA2_net import PartA2Net
from .second_net import SECONDNet
from .pointpillar import PointPillar
from .MVF import MVF

all_detectors = {
    'PartA2_net': PartA2Net,
    'second_net': SECONDNet,
    'PointPillar': PointPillar,
    'MVF': MVF
}