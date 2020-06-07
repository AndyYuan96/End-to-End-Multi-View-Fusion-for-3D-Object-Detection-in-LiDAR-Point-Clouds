import torch 
from collections import namedtuple
from .detectors import all_detectors
from ..config import cfg

all_models = {
    **all_detectors
}


def build_network(dataset):
    model = all_models[cfg.MODEL.NAME](
        num_class=len(cfg.CLASS_NAMES),
        dataset=dataset,
    )

    return model


def model_fn_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, data):
        input_dict = example_convert_to_torch(data)
        ret_dict, tb_dict, disp_dict = model(input_dict)

        loss = ret_dict['loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func


def example_convert_to_torch(example, dtype=torch.float32):
    device = torch.cuda.current_device()
    example_torch = {}
    float_names = [
        'voxels', 'anchors', 'box_reg_targets', 'reg_weights', 'part_labels',
        'gt_boxes', 'voxel_centers', 'reg_src_targets', 'points',
    ]

    mvf_names = ['bev_local_coordinate', 'fv_local_coordinate', 'intensity','bev_mapping_vf', 'fv_mapping_vf', 'bev_mapping_pv', 'fv_mapping_pv', 'bev_coordinate']

    for k, v in example.items():
        if k in float_names:
            if torch.is_tensor(v):
                example_torch[k] = v
            else:
                try:
                    example_torch[k] = torch.tensor(v, dtype=torch.float32, device=device).to(dtype)
                except RuntimeError:
                    example_torch[k] = torch.zeros((v.shape[0], 1, 7), dtype=torch.float32, device=device).to(dtype)
        elif k in mvf_names:
            example_torch[k] = v.to(device=device)
        elif k in ['coordinates', 'box_cls_labels', 'num_points', 'seg_labels']:
            if torch.is_tensor(v):
                example_torch[k] = v
            else:
                example_torch[k] = torch.tensor(v, dtype=torch.int32, device=device)
        else:
            example_torch[k] = v
    
    return example_torch
