from .vfe_utils import MeanVoxelFeatureExtractor, PillarFeatureNetOld2, MVFFeatureNet,MVFFeatureNetDVP


vfe_modules = {
    'MeanVoxelFeatureExtractor': MeanVoxelFeatureExtractor,
    'PillarFeatureNetOld2': PillarFeatureNetOld2,
    'MVFFeatureNet': MVFFeatureNet,
    'MVFFeatureNetDVP' : MVFFeatureNetDVP
}