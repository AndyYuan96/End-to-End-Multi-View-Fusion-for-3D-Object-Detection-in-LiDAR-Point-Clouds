import torch
import torch.nn as nn
import torch.nn.functional as F
from ...config import cfg
from ..model_utils.pytorch_utils import Empty
import math
from scatter_max import scatter_max,scatter_mean

def scatter_nd(indices, updates, shape):
    """pytorch edition of tensorflow scatter_nd.
    this function don't contain except handle code. so use this carefully
    when indice repeats, don't support repeat add which is supported
    in tensorflow.
    """
    ret = torch.zeros(*shape, dtype=updates.dtype, device=updates.device)
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + shape[indices.shape[-1]:]
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    ret[slices] = updates.view(*output_shape)
    return ret

def dense(batch_size, spatial_shape, feature_dim, indices, features, channels_first=True):
    output_shape = [batch_size] + list(spatial_shape) + [feature_dim]
    res = scatter_nd(indices.long(), features, output_shape)
    if not channels_first:
        return res
    ndim = len(spatial_shape)
    trans_params = list(range(0, ndim + 1))
    trans_params.insert(1, ndim + 1)
    return res.permute(*trans_params).contiguous()

class ScatterMaxCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, input_index, output, output_index): 
        scatter_max(input, input_index, output, output_index, True)
        ctx.size = input.size()
        ctx.save_for_backward(output_index)

        return output
    
    @staticmethod
    def backward(ctx, output_grad):
        output_index = ctx.saved_tensors[0]
        grad_input = output_grad.new_zeros(ctx.size)
        # print("test grad")
        # print("max : ", output_index.max())
        # print("min : ", output_index.min())
        # print("points counts : ", ctx.size[0])
        grad_input.scatter_(0, output_index, output_grad)
       
        return grad_input, None, None, None

def scatterMax(input, input_index, voxel_nums, train):
    '''
        only accept two dimension tensor, and do maxpooing in first dimension
    '''
    output = input.new_full((voxel_nums, input.shape[1]), torch.finfo(input.dtype).min)
    output_index = input_index.new_empty((voxel_nums, input.shape[1]))

    if train:
        output = ScatterMaxCuda.apply(input, input_index, output, output_index)
    else:
        output = scatter_max(input, input_index, output, output_index, False)
    
    return output

def scatterMean(input, input_index, voxel_nums):
    output = input.new_full((voxel_nums, input.shape[1]), 0.0)
    input_mean = input.new_empty(input.shape)

    scatter_mean(input, input_index, output,input_mean)
    return input_mean


class VoxelFeatureExtractor(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def get_output_feature_dim(self):
        raise NotImplementedError

    def forward(self, **kwargs):
        raise NotImplementedError


class MeanVoxelFeatureExtractor(VoxelFeatureExtractor):
    def __init__(self, **kwargs):
        super().__init__()

    def get_output_feature_dim(self):
        return cfg.DATA_CONFIG.NUM_POINT_FEATURES['use']

    def forward(self, features, num_voxels, **kwargs):
        """
        :param features: (N, max_points_of_each_voxel, 3 + C)
        :param num_voxels: (N)
        :param kwargs:
        :return:
        """
        points_mean = features[:, :, :].sum(dim=1, keepdim=False) / num_voxels.type_as(features).view(-1, 1)
        return points_mean.contiguous()


def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.
    Args:
        actual_num ([type]): [description]
        max_num ([type]): [description]
    Returns:
        [type]: [description]
    """

    actual_num = torch.unsqueeze(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        """
        Pillar Feature Net Layer.
        The Pillar Feature Net could be composed of a series of these layers, but the PointPillars paper results only
        used a single PFNLayer.
        :param in_channels: <int>. Number of input channels.
        :param out_channels: <int>. Number of output channels.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param last_layer: <bool>. If last_layer, there is no concatenation of features.
        """

        super().__init__()
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        if use_norm:
            self.linear = nn.Linear(in_channels, self.units, bias=False)
            self.norm = nn.BatchNorm1d(self.units, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, self.units, bias=True)
            self.norm = Empty(self.units)

    def forward(self, inputs):
        x = self.linear(inputs)
        # x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        total_points, voxel_points, channels = x.shape
        x = self.norm(x.view(-1, channels)).view(total_points, voxel_points, channels)
        x = F.relu(x)

        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarFeatureNetOld2(VoxelFeatureExtractor):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=(64, ),
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1)):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """
        super().__init__()
        self.name = 'PillarFeatureNetOld2'
        assert len(num_filters) > 0
        num_input_features += 6
        if with_distance:
            num_input_features += 1
        self.with_distance = with_distance
        self.num_filters = num_filters
        # Create PillarFeatureNetOld layers
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, use_norm, last_layer=last_layer)
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]
        self.z_offset = self.vz / 2 + pc_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def forward(self, features, num_voxels, coords):
        """
        :param features: (N, max_points_of_each_voxel, 3 + C)
        :param num_voxels: (N)
        :param coors:
        :return:
        """
        dtype = features.dtype
        # Find distance of x, y, and z from cluster center
        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean

        # Find distance of x, y, and z from pillar center
        # f_center = features[:, :, :3]
        f_center = torch.zeros_like(features[:, :, :3])
        f_center[:, :, 0] = features[:, :, 0] - (coords[:, 3].to(dtype).unsqueeze(1) * self.vx + self.x_offset)
        f_center[:, :, 1] = features[:, :, 1] - (coords[:, 2].to(dtype).unsqueeze(1) * self.vy + self.y_offset)
        f_center[:, :, 2] = features[:, :, 2] - (coords[:, 1].to(dtype).unsqueeze(1) * self.vz + self.z_offset)

        # Combine together feature decorations
        features_ls = [features, f_cluster, f_center]
        if self.with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)
        features = torch.cat(features_ls, dim=-1)

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        # Forward pass through PFNLayers
        for pfn in self.pfn_layers:
            features = pfn(features)
               
        return features.squeeze()

class MVFFeatureNetDVP(VoxelFeatureExtractor):
    def __init__(self,
                bev_h, bev_w):
        super().__init__()
        self.name = 'MVFFeatureNetDVP'
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_size = self.bev_h * self.bev_w

        self.bev_FC = nn.Sequential(
                nn.Linear(10,64,bias=False),
                nn.BatchNorm1d(64,eps=1e-3,momentum=0.01),
                nn.ReLU(inplace=True)
            )
        
    def forward(self, input_dict):
        batch_size = input_dict['batch_size']
        bev_coordinate = input_dict['bev_coordinate']
        bev_local_coordinate = input_dict['bev_local_coordinate']
        intensity = input_dict['intensity']
        bev_mapping_pv = input_dict['bev_mapping_pv']
        # throw z position
        bev_mapping_vf = input_dict['bev_mapping_vf'][:,:3].contiguous()

        point_mean = scatterMean(bev_coordinate, bev_mapping_pv, bev_mapping_vf.shape[0])
        feature = torch.cat((bev_coordinate, intensity.unsqueeze(1), (bev_coordinate - point_mean), bev_local_coordinate),dim=1).contiguous()

        bev_fc_output = self.bev_FC(feature)
        bev_maxpool = scatterMax(bev_fc_output, bev_mapping_pv, bev_mapping_vf.shape[0], True)
        bev_dense = dense(batch_size, [self.bev_h, self.bev_w], 64, bev_mapping_vf, bev_maxpool)

        return bev_dense


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.deconv0 = self._make_layer(block, 64, layers[0])
        self.conv1 = self._make_layer(block, 128, layers[1], stride=2)
        self.conv2 = self._make_layer(block, 128, layers[2], stride=2)
        self.deconv1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)

        self.deconv1 =  nn.Sequential(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                                      nn.BatchNorm2d(64,eps=1e-3,momentum=0.01),
                                      nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 3, stride=4, output_padding=1),
                                      nn.BatchNorm2d(64,eps=1e-3,momentum=0.01),
                                      nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 3, 64, 3, stride=1, padding=1),
                                   nn.BatchNorm2d(64,eps=1e-3,momentum=0.01),
                                   nn.ReLU())

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-3, momentum=0.01),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        deconv0_x = self.deconv0(x)

        conv1_x = self.conv1(x)
        conv2_x = self.conv2(conv1_x)

        deconv1_x = self.deconv1(conv1_x)
        deconv2_x = self.deconv2(conv2_x)

        final_x = torch.cat((deconv0_x,deconv1_x,deconv2_x),dim=1)
        final_x = self.conv3(final_x)

        return final_x


class MVFFeatureNet(VoxelFeatureExtractor):
    def __init__(self,
                bev_h, bev_w,
                fv_h, fv_w,
                with_tower):
        super().__init__()
        self.name = 'MVFFeatureNet'
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fv_h = fv_h
        self.fv_w = fv_w
        self.with_tower = with_tower

        self.bev_size = self.bev_h * self.bev_w
        self.fv_size = self.fv_h * self.fv_w

        self.shared_FC = nn.Sequential(
            nn.Linear(7, 128, bias=False),
            nn.BatchNorm1d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True)
        )

        self.bev_FC = nn.Sequential(
            nn.Linear(3,64,bias=False),
            nn.BatchNorm1d(64,eps=1e-3,momentum=0.01),
            nn.ReLU(inplace=True)
        )

        self.fv_FC = nn.Sequential(
            nn.Linear(3,64,bias=False),
            nn.BatchNorm1d(64,eps=1e-3,momentum=0.01),
            nn.ReLU(inplace=True)
        )

        self.downsample_FC = nn.Sequential(
            nn.Linear(256, 64, bias=False),
            nn.BatchNorm1d(64,eps=1e-3,momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.fv_tower = None
        self.bev_tower = None
        
        if self.with_tower:
            self.fv_tower = ResNet(BasicBlock,[1,1,1])
            self.bev_tower = ResNet(BasicBlock,[1,1,1])
    
    def forward(self, input_dict):
        batch_size = input_dict['batch_size']
        
        bev_local_coordinate = input_dict['bev_local_coordinate']
        fv_local_coordiante = input_dict['fv_local_coordinate']
        intensity = input_dict['intensity']
        bev_mapping_pv = input_dict['bev_mapping_pv']
        bev_mapping_vf = input_dict['bev_mapping_vf']
        fv_mapping_pv = input_dict['fv_mapping_pv']
        fv_mapping_vf = input_dict['fv_mapping_vf']

        bev_fc_output = self.bev_FC(bev_local_coordinate)
        bev_maxpool = scatterMax(bev_fc_output, bev_mapping_pv, bev_mapping_vf.shape[0], True)
        
        fv_fc_output = self.fv_FC(fv_local_coordiante)
        fv_maxpool = scatterMax(fv_fc_output, fv_mapping_pv, fv_mapping_vf.shape[0], True)

        shared_fc_input = torch.cat((bev_local_coordinate, fv_local_coordiante, intensity.unsqueeze(1)), dim=1)
        shared_fc_output = self.shared_FC(shared_fc_input)

        bev_dense = dense(batch_size, [self.bev_h, self.bev_w], 64, bev_mapping_vf, bev_maxpool)
        fv_dense = dense(batch_size, [self.fv_h, self.fv_w], 64, fv_mapping_vf, fv_maxpool)

        bev_feature = None
        fv_feature = None

        if self.with_tower:
            bev_feature = self.bev_tower(bev_dense)
            fv_feature = self.fv_tower(fv_dense)
        else:
            bev_feature = bev_dense
            fv_feature = fv_dense
        
        # to (batch, h, w, c)
        fv_feature = fv_feature.permute(0,2,3,1).reshape(-1,64).contiguous()
        bev_feature = bev_feature.permute(0,2,3,1).reshape(-1,64).contiguous()
        
        # get each voxel's position in feature map
        # and then scatter those voxel
        bev_voxel_coordiante = bev_mapping_vf[:,0] * self.bev_size + bev_mapping_vf[:, 1] * self.bev_w + bev_mapping_vf[:, 2]
        fv_voxel_coordiante = fv_mapping_vf[:,0] * self.fv_size + fv_mapping_vf[:,1] * self.fv_w + fv_mapping_vf[:,2]
        
        # (64,M)
        bev_voxel_feature = torch.index_select(bev_feature,0,bev_voxel_coordiante)
        fv_voxel_feature = torch.index_select(fv_feature, 0, fv_voxel_coordiante)
        
        #bev_voxel_feature is (n1+n2+n3,3), bev_mapping_pv is (id + n1+id + n1+n2+id)
        bev_point_feature = torch.index_select(bev_voxel_feature, 0, bev_mapping_pv)
        fv_point_feature = torch.index_select(fv_voxel_feature, 0, fv_mapping_pv)

        final_point_feature = torch.cat((shared_fc_output, bev_point_feature, fv_point_feature),dim=1).contiguous()

        voxel_feature = scatterMax(final_point_feature, bev_mapping_pv, bev_mapping_vf.shape[0], True)
        
        final_voxel_feature = self.downsample_FC(voxel_feature)
        final_voxel_feature = dense(batch_size, [self.bev_h, self.bev_w], 64, bev_mapping_vf, final_voxel_feature)

        return final_voxel_feature






                






        






        



