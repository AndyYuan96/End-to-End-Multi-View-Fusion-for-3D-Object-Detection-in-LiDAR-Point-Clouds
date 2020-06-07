import numpy as np
from collections import defaultdict
import torch.utils.data as torch_data
from ..utils import box_utils, common_utils
from ..config import cfg
from .data_augmentation import augmentation_utils

import torch

class DatasetTemplate(torch_data.Dataset):
    def __init__(self):
        super().__init__()

    def get_infos(self, **kwargs):
        """generate data infos from raw data for the dataset"""
        raise NotImplementedError

    def create_groundtruth_database(self, **kwargs):
        """create groundtruth database for GT sampling augmentation"""
        raise NotImplementedError

    @staticmethod
    def generate_prediction_dict(input_dict, index, record_dict):
        """
        Generate the prediction dict for each sample, called by the post processing.
        Args:
            input_dict: provided by the dataset to provide dataset-specific information
            index: batch index of current sample
            record_dict: the predicted results of current sample from the detector,
                which currently includes these keys: {
                    'boxes': (N, 7 + C)  [x, y, z, w, l, h, heading_in_kitti] in LiDAR coords
                    'scores': (N)
                    'labels': (N）
                }
        Returns:
            predictions_dict: the required prediction dict of current scene for specific dataset
        """
        raise NotImplementedError

    @staticmethod
    def generate_annotations(input_dict, pred_dicts, class_names, save_to_file=False, output_dir=None):
        """
        Generate the annotation dict for each batch to be used for evaluation,
        and also (optionally) save the results to file.
        Args:
            input_dict: provided by the dataset to provide dataset-specific information
            pred_dicts: list of dict, each dict is provided by the function 'generate_prediction_dict'
            class_names: list of string, the names of all classes in order
            save_to_file: whether to save the results to file
            output_dir: output directory for saving the results
        Returns:
            list of dict, each dict is the predicted results for each scene
        """
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def forward(self, index):
        raise NotImplementedError

    def prepare_data(self, input_dict, has_label=True):
        """
        :param input_dict:
            sample_idx: string
            calib: object, calibration related
            points: (N, 3 + C1)
            gt_boxes_lidar: optional, (N, 7) [x, y, z, w, l, h, rz] in LiDAR coordinate, z is the bottom center
            gt_names: optional, (N), string
        :param has_label: bool
        :return:
            voxels: (N, max_points_of_each_voxel, 3 + C2), float
            num_points: (N), int
            coordinates: (N, 3), [idx_z, idx_y, idx_x]
            num_voxels: (N)
            voxel_centers: (N, 3)
            calib: object
            gt_boxes: (N, 8), [x, y, z, w, l, h, rz, gt_classes] in LiDAR coordinate, z is the bottom center
            points: (M, 3 + C)
        """
        sample_idx = input_dict['sample_idx']
        points = input_dict['points']
        calib = input_dict['calib']

        if has_label:
            gt_boxes = input_dict['gt_boxes_lidar'].copy()
            gt_names = input_dict['gt_names'].copy()

        if self.training:
            selected = common_utils.drop_arrays_by_name(gt_names, ['DontCare', 'Sign'])
            gt_boxes = gt_boxes[selected]
            gt_names = gt_names[selected]
            gt_boxes_mask = np.array([n in self.class_names for n in gt_names], dtype=np.bool_)

            if self.db_sampler is not None:
                road_planes = self.get_road_plane(sample_idx) \
                    if cfg.DATA_CONFIG.AUGMENTATION.DB_SAMPLER.USE_ROAD_PLANE else None
                sampled_dict = self.db_sampler.sample_all(
                    self.root_path, gt_boxes, gt_names, road_planes=road_planes,
                    num_point_features=cfg.DATA_CONFIG.NUM_POINT_FEATURES['total'], calib=calib
                )

                if sampled_dict is not None:
                    sampled_gt_names = sampled_dict['gt_names']
                    sampled_gt_boxes = sampled_dict['gt_boxes']
                    sampled_points = sampled_dict['points']
                    sampled_gt_masks = sampled_dict['gt_masks']

                    gt_names = np.concatenate([gt_names, sampled_gt_names], axis=0)
                    gt_boxes = np.concatenate([gt_boxes, sampled_gt_boxes])
                    gt_boxes_mask = np.concatenate([gt_boxes_mask, sampled_gt_masks], axis=0)

                    points = box_utils.remove_points_in_boxes3d(points, sampled_gt_boxes)
                    points = np.concatenate([sampled_points, points], axis=0)

            noise_per_object_cfg = cfg.DATA_CONFIG.AUGMENTATION.NOISE_PER_OBJECT
            if noise_per_object_cfg.ENABLED:
                gt_boxes, points = \
                    augmentation_utils.noise_per_object_v3_(
                    gt_boxes,
                    points,
                    gt_boxes_mask,
                    rotation_perturb=noise_per_object_cfg.GT_ROT_UNIFORM_NOISE,
                    center_noise_std=noise_per_object_cfg.GT_LOC_NOISE_STD,
                    num_try=100
                )

            gt_boxes = gt_boxes[gt_boxes_mask]
            gt_names = gt_names[gt_boxes_mask]

            gt_classes = np.array([self.class_names.index(n) + 1 for n in gt_names], dtype=np.int32)

            noise_global_scene = cfg.DATA_CONFIG.AUGMENTATION.NOISE_GLOBAL_SCENE
            if noise_global_scene.ENABLED:
                gt_boxes, points = augmentation_utils.random_flip(gt_boxes, points)
                gt_boxes, points = augmentation_utils.global_rotation(
                    gt_boxes, points, rotation=noise_global_scene.GLOBAL_ROT_UNIFORM_NOISE
                )
                gt_boxes, points = augmentation_utils.global_scaling(
                    gt_boxes, points, *noise_global_scene.GLOBAL_SCALING_UNIFORM_NOISE
                )
                # add according to pointpillars
                if "GLOBAL_LOC_NOISE_STD" in noise_global_scene.keys():
                    gt_boxes, points = augmentation_utils.global_translate(gt_boxes, points, *noise_global_scene.GLOBAL_LOC_NOISE_STD)

            pc_range = self.voxel_generator.point_cloud_range
            mask = box_utils.mask_boxes_outside_range(gt_boxes, pc_range)
            gt_boxes = gt_boxes[mask]
            gt_classes = gt_classes[mask]
            gt_names = gt_names[mask]

            # limit rad to [-pi, pi]
            gt_boxes[:, 6] = common_utils.limit_period(gt_boxes[:, 6], offset=0.5, period=2 * np.pi)

        points = points[:, :cfg.DATA_CONFIG.NUM_POINT_FEATURES['use']]
        if cfg.DATA_CONFIG[self.mode].SHUFFLE_POINTS:
            np.random.shuffle(points)
        
        if cfg.DATA_CONFIG.MASK_POINTS_BY_RANGE:
            points = common_utils.mask_points_by_range(points, cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
    
        example = {}

        if cfg.MODEL.NAME == 'MVF':
            # do my generate
            if cfg.MODEL.VFE.NAME == 'MVFFeatureNetDVP':
                points = torch.from_numpy(points)
                voxel_generator_cfg = cfg.DATA_CONFIG.VOXEL_GENERATOR

                bev_coordinate = torch.zeros((voxel_generator_cfg.MAX_POINT_NUMS, 3), dtype=torch.float32)
                bev_local_coordinate = torch.zeros((voxel_generator_cfg.MAX_POINT_NUMS, 3), dtype=torch.float32)
                intensity = torch.zeros((voxel_generator_cfg.MAX_POINT_NUMS), dtype=torch.float32)
                bev_mapping_pv = torch.zeros((voxel_generator_cfg.MAX_POINT_NUMS), dtype=torch.int32)
                bev_mapping_vf = torch.zeros((voxel_generator_cfg.MAX_NUMBER_OF_BEV_VOXEL, 3), dtype=torch.int32)

                self.mvf_feature.dynamicVoxelBEV(points, bev_coordinate, bev_local_coordinate, intensity, bev_mapping_pv, bev_mapping_vf)
                valid_point_nums = self.mvf_feature.getValidPointNums()
                valid_bev_voxel_nums = self.mvf_feature.getValidBEVVoxelNums()

                bev_coordinate = bev_coordinate[:valid_point_nums]
                bev_local_coordinate = bev_local_coordinate[:valid_point_nums]
                intensity = intensity[:valid_point_nums]
                bev_mapping_pv = bev_mapping_pv[:valid_point_nums]
                bev_mapping_vf = bev_mapping_vf[:valid_bev_voxel_nums]
                example.update({
                    'bev_coordinate' : bev_coordinate,
                    'bev_local_coordinate' : bev_local_coordinate,
                    'intensity' : intensity,
                    'bev_mapping_pv' : bev_mapping_pv,
                    'bev_mapping_vf' : bev_mapping_vf,
                    'calib' : input_dict['calib'],
                })

            else:
                points = torch.from_numpy(points)
                voxel_generator_cfg = cfg.DATA_CONFIG.VOXEL_GENERATOR
                bev_local_coordinate = torch.zeros((voxel_generator_cfg.MAX_POINT_NUMS, 3), dtype=torch.float32)
                fv_local_coordinate = torch.zeros((voxel_generator_cfg.MAX_POINT_NUMS, 3), dtype=torch.float32)
                intensity = torch.zeros((voxel_generator_cfg.MAX_POINT_NUMS), dtype=torch.float32)
                bev_mapping_pv = torch.zeros((voxel_generator_cfg.MAX_POINT_NUMS), dtype=torch.int32)
                bev_mapping_vf = torch.zeros((voxel_generator_cfg.MAX_NUMBER_OF_BEV_VOXEL, 2), dtype=torch.int32)
                fv_mapping_pv = torch.zeros((voxel_generator_cfg.MAX_POINT_NUMS), dtype=torch.int32)
                fv_mapping_vf = torch.zeros((voxel_generator_cfg.MAX_NUMBER_OF_FV_VOXEL, 2), dtype=torch.int32)
                self.mvf_feature.MVFVoxel(points, bev_local_coordinate, fv_local_coordinate, intensity, bev_mapping_pv, bev_mapping_vf, 
                                                fv_mapping_pv, fv_mapping_vf)
            
                            
                valid_point_nums = self.mvf_feature.getValidPointNums()
                valid_bev_voxel_nums = self.mvf_feature.getValidBEVVoxelNums()
                valid_fv_voxel_nums = self.mvf_feature.getValidFVVoxelNums()

                bev_local_coordinate = bev_local_coordinate[:valid_point_nums]
                fv_local_coordinate = fv_local_coordinate[:valid_point_nums]
                intensity = intensity[:valid_point_nums]
                bev_mapping_pv = bev_mapping_pv[:valid_point_nums]
                bev_mapping_vf = bev_mapping_vf[:valid_bev_voxel_nums]
                fv_mapping_pv = fv_mapping_pv[:valid_point_nums]
                fv_mapping_vf = fv_mapping_vf[:valid_fv_voxel_nums]

                # RECOVER TEST : PASS
                # tmp1 = torch.index_select(bev_mapping_vf, dim=0, index=bev_mapping_pv.squeeze().long())
                # tmp1 = tmp1.float()
                # tmp_coordinate = torch.zeros_like(bev_local_coordinate, dtype=torch.float32)
                # tmp_coordinate[:,0] = bev_local_coordinate[:,0] + tmp1[:,0] * (voxel_generator_cfg.BEV_RANGE[1] - voxel_generator_cfg.BEV_RANGE[0]) / voxel_generator_cfg.BEV_FEATURE_SIZE_XY[0] + voxel_generator_cfg.BEV_RANGE[0]
                # tmp_coordinate[:,1] = bev_local_coordinate[:,1] + tmp1[:,1] * (voxel_generator_cfg.BEV_RANGE[3] - voxel_generator_cfg.BEV_RANGE[2]) / voxel_generator_cfg.BEV_FEATURE_SIZE_XY[1] + voxel_generator_cfg.BEV_RANGE[2]
                # tmp_coordinate[:,2] = bev_local_coordinate[:,2] + voxel_generator_cfg.BEV_RANGE[4]

                # points.numpy().tofile("/home/ovo/code/tmp/origin.bin")
                # tmp_coordinate.numpy().tofile("/home/ovo/code/tmp/recover.bin")
                # while(1):
                #     pass

                example.update({
                    'bev_local_coordinate' : bev_local_coordinate,
                    'fv_local_coordinate' : fv_local_coordinate,
                    'intensity' : intensity,
                    'bev_mapping_pv' : bev_mapping_pv,
                    'bev_mapping_vf' : bev_mapping_vf,
                    'fv_mapping_pv' : fv_mapping_pv,
                    'fv_mapping_vf' : fv_mapping_vf,
                    'calib' : input_dict['calib'],
                })

        else:
            voxels, coordinates, num_points = \
                self.voxel_generator.generate(points, max_voxels=cfg.DATA_CONFIG[self.mode].MAX_NUMBER_OF_VOXELS)

            voxel_centers = (coordinates[:, ::-1] + 0.5) * self.voxel_generator.voxel_size \
                            + self.voxel_generator.point_cloud_range[0:3]
            
            example.update({
            'voxels': voxels,
            'num_points': num_points,
            'coordinates': coordinates,
            'voxel_centers': voxel_centers,
            'calib': input_dict['calib'],
            'points': points
            })

        if has_label:
            if not self.training:
                # for eval_utils
                selected = common_utils.keep_arrays_by_name(gt_names, self.class_names)
                gt_boxes = gt_boxes[selected]
                gt_names = gt_names[selected]
                gt_classes = np.array([self.class_names.index(n) + 1 for n in gt_names], dtype=np.int32)

            if 'TARGET_CONFIG' in cfg.MODEL.RPN.BACKBONE \
                and cfg.MODEL.RPN.BACKBONE.TARGET_CONFIG.GENERATED_ON == 'dataset':
                seg_labels, part_labels, bbox_reg_labels = \
                    self.generate_voxel_part_targets(voxel_centers, gt_boxes, gt_classes)
                example['seg_labels'] = seg_labels
                example['part_labels'] = part_labels
                if bbox_reg_labels is not None:
                    example['bbox_reg_labels'] = bbox_reg_labels

            gt_boxes = np.concatenate((gt_boxes, gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)

            example.update({
                'gt_boxes': gt_boxes
            })

        return example

    def generate_voxel_part_targets(self, voxel_centers, gt_boxes, gt_classes, generate_bbox_reg_labels=False):
        """
        :param voxel_centers: (N, 3) [x, y, z]
        :param gt_boxes: (M, 7) [x, y, z, w, l, h, ry] in LiDAR coords
        :return:
        """
        unet_target_cfg = cfg.MODEL.RPN.BACKBONE.TARGET_CONFIG

        MEAN_SIZE = unet_target_cfg.MEAN_SIZE
        GT_EXTEND_WIDTH = unet_target_cfg.GT_EXTEND_WIDTH

        extend_gt_boxes = common_utils.enlarge_box3d(gt_boxes, extra_width=GT_EXTEND_WIDTH)
        gt_corners = box_utils.boxes3d_to_corners3d_lidar(gt_boxes)
        extend_gt_corners = box_utils.boxes3d_to_corners3d_lidar(extend_gt_boxes)

        cls_labels = np.zeros(voxel_centers.shape[0], dtype=np.int32)
        reg_labels = np.zeros((voxel_centers.shape[0], 3), dtype=np.float32)
        bbox_reg_labels = np.zeros((voxel_centers.shape[0], 7), dtype=np.float32) if generate_bbox_reg_labels else None

        for k in range(gt_boxes.shape[0]):
            box_corners = gt_corners[k]
            fg_pt_flag = box_utils.in_hull(voxel_centers, box_corners)
            fg_voxels = voxel_centers[fg_pt_flag]
            cls_labels[fg_pt_flag] = gt_classes[k]

            # enlarge the bbox3d, ignore nearby points
            extend_box_corners = extend_gt_corners[k]
            fg_enlarge_flag = box_utils.in_hull(voxel_centers, extend_box_corners)
            ignore_flag = np.logical_xor(fg_pt_flag, fg_enlarge_flag)
            cls_labels[ignore_flag] = -1

            # part offset labels
            transformed_voxels = fg_voxels - gt_boxes[k, 0:3]
            transformed_voxels = common_utils.rotate_pc_along_z(transformed_voxels, -gt_boxes[k, 6])
            reg_labels[fg_pt_flag] = (transformed_voxels / gt_boxes[k, 3:6]) + np.array([0.5, 0.5, 0], dtype=np.float32)

            if generate_bbox_reg_labels:
                # rpn bbox regression target
                center3d = gt_boxes[k, 0:3].copy()
                center3d[2] += gt_boxes[k][5] / 2  # shift to center of 3D boxes
                bbox_reg_labels[fg_pt_flag, 0:3] = center3d - fg_voxels
                bbox_reg_labels[fg_pt_flag, 6] = gt_boxes[k, 6]  # dy

                cur_mean_size = MEAN_SIZE[cfg.CLASS_NAMES[gt_classes[k] - 1]]
                bbox_reg_labels[fg_pt_flag, 3:6] = (gt_boxes[k, 3:6] - np.array(cur_mean_size)) / cur_mean_size

        reg_labels = np.maximum(reg_labels, 0)
        return cls_labels, reg_labels, bbox_reg_labels

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        example_merged = defaultdict(list)
        for example in batch_list:
            for k, v in example.items():
                example_merged[k].append(v)
        ret = {}
        for key, elems in example_merged.items():
            if key in ['voxels', 'num_points', 'voxel_centers', 'seg_labels', 'part_labels', 'bbox_reg_labels']:
                ret[key] = np.concatenate(elems, axis=0)
            elif key in ['coordinates', 'points']:
                coors = []
                for i, coor in enumerate(elems):
                    coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                    coors.append(coor_pad)
                ret[key] = np.concatenate(coors, axis=0)
            elif key in ['gt_boxes']:
                max_gt = 0
                batch_size = elems.__len__()
                for k in range(batch_size):
                    max_gt = max(max_gt, elems[k].__len__())
                batch_gt_boxes3d = np.zeros((batch_size, max_gt, elems[0].shape[-1]), dtype=np.float32)
                for k in range(batch_size):
                    batch_gt_boxes3d[k, :elems[k].__len__(), :] = elems[k]
                ret[key] = batch_gt_boxes3d
            elif key in ['bev_local_coordinate', 'fv_local_coordinate', 'intensity', 'bev_coordinate']:
                ret[key] = torch.cat(elems, dim=0)
            elif key in ['bev_mapping_vf', 'fv_mapping_vf']:
                for idx, elem in enumerate(elems):
                    cur_elem = torch.cat((idx * torch.ones((elem.shape[0],1),dtype=torch.int32), elem), dim=1)
                    if idx == 0:
                        ret[key] = cur_elem
                    else:
                        ret[key] = torch.cat((ret[key], cur_elem), dim=0)
                
                ret[key] = ret[key].long()
            elif key in ['bev_mapping_pv', 'fv_mapping_pv']:
                cur_voxel_num = 0
                for idx, elem in enumerate(elems):
                    if idx == 0:
                        ret[key] = elem
                    else:
                        ret[key] = torch.cat((ret[key], elem + cur_voxel_num), dim=0)
                    if key == 'bev_mapping_pv':
                        cur_voxel_num += example_merged['bev_mapping_vf'][idx].shape[0]
                    else:
                        cur_voxel_num += example_merged['fv_mapping_vf'][idx].shape[0]
                ret[key] = ret[key].long()
            else:
                ret[key] = np.stack(elems, axis=0)
        ret['batch_size'] = batch_list.__len__()
        return ret
