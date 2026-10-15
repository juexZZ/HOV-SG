#!/bin/bash

# (hovsg) jz4725@maxwell:/local_data/jz4725/HOV-SG$ ls ../Depth-Anything-3/metric_depth_results/
# 12thfloor           28_metfloor           28_metfloor_trav_12  2ndfloor_trav_1  pantry_2traverse
# 12thfloor_first150  28_metfloor_first150  2ndfloor             pantry1
# 12thfloor_trav_1    28_metfloor_trav_1    2ndfloor_first150    pantry2


python application/create_graph.py \
    main.dataset=scannet \
    main.split="" \
    main.scene_id=12thfloor_trav_1 \
    main.dataset_path=/local_data/jz4725/Depth-Anything-3/metric_depth_results \
    main.save_path=/local_data/jz4725/HOV-SG/data/scene_graphs \
    pipeline.skip_frames=10

python application/create_graph.py \
    main.dataset=scannet \
    main.split="" \
    main.scene_id=2ndfloor_trav_1 \
    main.dataset_path=/local_data/jz4725/Depth-Anything-3/metric_depth_results \
    main.save_path=/local_data/jz4725/HOV-SG/data/scene_graphs \
    pipeline.skip_frames=10

python application/create_graph.py \
    main.dataset=scannet \
    main.split="" \
    main.scene_id=28_metfloor_trav_1 \
    main.dataset_path=/local_data/jz4725/Depth-Anything-3/metric_depth_results \
    main.save_path=/local_data/jz4725/HOV-SG/data/scene_graphs \
    pipeline.skip_frames=10