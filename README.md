

# End-to-End Multi-View Fusion for 3D Object Detection in LiDAR Point Clouds

​	This is an unofficial implementation of MVF based on [PCDet](https://github.com/sshaoshuai/PCDet)  and [PointCloudDynamicVoxel](https://github.com/AndyYuan96/PointCloudDynamicVoxel).



## How to use

​	Follow  PCDet and PointCloudDynamicVoxel install guide.

## Performance

```
DV+SV (same setting with PCDet's pointpillar config except dynamic voxel)
config : tools/mvf/mvf_pp_dv.yaml
model : output/mvf_pp_dv.pth
INFO  Car AP@0.70, 0.70, 0.70:
bbox AP:94.2958, 89.5560, 88.6962
bev  AP:89.6566, 86.9236, 84.3930
3d   AP:87.4800, 77.1988, 75.5659
aos  AP:94.27, 89.31, 88.33

MVF ([lr = 0.0015, epoch 100, only car](paper setting), batch 6)
config : tools/mvf/mvf_paper_car.yaml
model : output/mvf_paper_car.pth
INFO  Car AP@0.70, 0.70, 0.70:
bbox AP:90.7383, 89.9176, 88.7714
bev  AP:90.0564, 87.7798, 85.5016
3d   AP:88.0401, 78.4811, 76.5275
aos  AP:90.72, 89.77, 88.52
```



## NOTE

​	As paper don't say too much about voxelization in perspective view, I just implementation according to my understanding, and the result is a little lower than paper. For the resolution of perspective view, too small(16,64) or too large(64, 512) don't get good result, (16, 128) is relatively good. 