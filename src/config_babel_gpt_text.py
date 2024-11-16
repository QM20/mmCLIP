import numpy as np
import pandas as pd
def load_sync_list(infilename):
    sync_list = []
    csv_file = pd.read_csv(infilename, header=0)
    print(infilename)
    for row_index in range(csv_file.shape[0]):
        trail_name = csv_file["trial_name"][row_index]
        sync_list.append(trail_name)
    return sync_list
# id_sub_act_try
exp_ids = [1]
sub_ids = [0]
act_ids = list(range(66))
act_ids.remove(0)
act_ids.remove(16)
act_ids.remove(23)
act_ids.remove(24)
act_ids.remove(42)
act_ids.remove(53)
try_ids = [0]
trials_list=[]
for exp_id in exp_ids:
    for sub_id in sub_ids:
        for act_id in act_ids:
            for try_id in try_ids:
                trial_name="S{:01d}_{:02d}_{:03d}_{:02d}".format(exp_id, sub_id, int(act_id), try_id)
                trials_list.append(trial_name)



test_class_list=[
    [2, 5, 7, 13, 14, 21, 26, 28, 30, 47],
    [3, 4, 6, 8, 12, 17, 19, 25, 31, 33],
    [1, 15, 18, 22, 27, 32, 37, 40, 43, 63],
]
all_list=act_ids
setting_list_babel=[
    {"exp_setting": "babel-human3d-kl-mbvitv3nosw-3group-seed2024",
    "dataset_list": ["ACCAD", "BioMotionLab_NTroje", "CMU", "EKUT", 'Eyes_Japan_Dataset', 'HumanEva',
                     'KIT', 'MPI_HDM05', 'MPI_Limits', 'MPI_mosh', 'SFU', 'SSM_synced', 'TCD_handMocap',
                     'TotalCapture', 'Transitions_mocap', "DFaust_67", "BMLmovi"],
    # babel parameters
    "if_use_babel": True, "if_use_gpt": True,
    "babel_train_data_location": ["./data/babel_data/4_td_hms_0-100_sigma0.15_0.7_0.15"],
    "babel_gpt_data_location": "./data/babel_data/6_prompts/",
    "label_dict_path": "./data/babel_data/1_segs_label_dict_framenone=1_cleancomma/",
    "if_use_img": False,
    "img_loss_ratio": 0.1,
    # humanml3d parameters
    "if_use_humanml3d": True,
    "humanml3d_gpt_location": "./data/humanml3d_data/6_prompts/",
    "humanml3d_train_data_location": ["./data/humanml3d_data/4_td_hms_0-100_sigma_1_0.15_1_1_0.2"],
    "humanml3d_text_paths": "./data/HumanML3D/HumanML3D/texts/",
    "humanml3d_cvs_paths": "./data/humanml3d_data/index.csv",
    ##local parameters
    "if_use_sim_local": False,
    "local_train_data_location": "N/A",
    "local_test_data_location": "./data/local_data/data_0505/hm_dra_modified_length_0-100",
    "gpt_data_location": "./data/local_data/gpt_data_0505/",
    "train_classes_real": "N/A",
    "train_ratio": 1,
    "test_ratio": 1,
    "train_order": "left_part",
    "test_order": "left_part",
    "train_sampling_gap": 500,
    ##t2motion parameters
    "if_use_t2m": False,
    ## model
    "if_range_aug": True,
    "model_type": "mmCLIP_gpt_multi_brach_property_v3",
    "trial_list": trials_list,
    "crop_size": (256, 224),
    "img_size": (224, 224),
    "test_sampling_gap": 32,
    "test_class_list": test_class_list,
    "if_use_hm": True,
    "if_use_hm_proj": False, "if_use_text_proj": False,
    "if_use_text_att": True, "if_use_hm_att": True, "if_use_img_att": False,
    "hm_type": [0, 1, 2],
    "batch_size": 16,
    "loss_type": "kl",
    "lr": 0.0001,
    "iteration_num": 50000,
    "gamma": 0.9,
    },

]