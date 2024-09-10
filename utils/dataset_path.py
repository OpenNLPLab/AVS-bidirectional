path_s4_cvr = [
    "/workspace/data/AVSBench//Single-source/s4_meta_data.csv",
    "/workspace/data/AVSBench/Single-source/s4_data/visual_frames",
    "/workspace/data/AVSBench/Single-source/s4_data/audio_log_mel",
    "/workspace/data/AVSBench/Single-source/s4_data/gt_masks"
]
path_ms3_cvr = [
    "/workspace/data/AVSBench/Multi-sources/ms3_meta_data.csv",
    "/workspace/data/AVSBench/Multi-sources/ms3_data/visual_frames",
    "/workspace/data/AVSBench/Multi-sources//ms3_data/audio_log_mel",
    "/workspace/data/AVSBench/Multi-sources//ms3_data/gt_masks"
]
path_s4_lab = [
    "/mnt/SSD/avs_bidirectional_generation/AVSBench/Single-source/s4_meta_data.csv",
    "/mnt/SSD/avs_bidirectional_generation/AVSBench/Single-source/s4_data/visual_frames",
    "/mnt/SSD/avs_bidirectional_generation/AVSBench/Single-source/s4_data/audio_log_mel",
    "/mnt/SSD/avs_bidirectional_generation/AVSBench/Single-source/s4_data/gt_masks"
]
path_ms3_lab = [
    "/mnt/SSD/avs_bidirectional_generation/AVSBench/Multi-sources/ms3_meta_data.csv",
    "/mnt/SSD/avs_bidirectional_generation/AVSBench/Multi-sources/ms3_data/visual_frames",
    "/mnt/SSD/avs_bidirectional_generation/AVSBench/Multi-sources/ms3_data/audio_log_mel",
    "/mnt/SSD/avs_bidirectional_generation/AVSBench/Multi-sources/ms3_data/gt_masks"
]

def get_path(flag):
    path_ms3, path_s4 = [], []
    if flag == "lab":
        path_s4, path_ms3 = path_s4_lab, path_ms3_lab
    elif flag == "cvr":
        path_s4, path_ms3 = path_s4_cvr, path_ms3_cvr
        
    return path_s4, path_ms3
