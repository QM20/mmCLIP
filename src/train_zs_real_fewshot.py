import sys
sys.path.append(".")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from model_gpt import mmCLIP_gpt_multi_brach_property_v3
from dataset import collate_fn, babel_dataset_gpt, local_dataset, local_dataset_fs, HumanML3DDataset
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
from config_zs_fewshot import setting_list
from torch.nn import functional as F
from peft import LoftQConfig, LoraConfig, get_peft_model
from torch.utils.data import DataLoader, ConcatDataset, Dataset

import random
import os
seed=2024
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

class CE_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.error_metric = nn.CrossEntropyLoss()
    def forward(self, prediction, label):
        loss = self.error_metric(prediction, label)
        return loss
class Cos_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.error_metric = nn.CosineSimilarity(dim=1, eps=1e-6)
    def forward(self, prediction, text_feature):
        cos = self.error_metric(prediction, text_feature)
        cosine_loss = (1 - cos).mean()
        return cosine_loss
class MSE_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.error_metric = nn.MSELoss()
    def forward(self, prediction, label):
        loss = self.error_metric(prediction, label)
        return loss
def gen_label(labels):
    num = len(labels)
    gt = np.zeros(shape=(num,num))
    for i, label in enumerate(labels):
        for k in range(num):
            if labels[k] == label:
                gt[i,k] = 1
    return gt
class KLLoss(nn.Module):
    def __init__(self):
        super().__init__()
        print('=========using KL Loss=and has temperature and * bz==========')
        self.error_metric = nn.KLDivLoss(reduction="batchmean")

    def forward(self, prediction, label):
        batch_size = prediction.shape[0]
        probs1 = F.log_softmax(prediction, 1)
        probs2 = F.softmax(label * 10, 1)
        loss = self.error_metric(probs1, probs2) #* batch_size
        return loss



np.set_printoptions(suppress=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device('cuda:0')

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            # print(name)
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )
if __name__=="__main__":
    exp_name = "zero-shot-mmCLIP-all-2024-fewshot"
    for setting_dict in setting_list:
        exp_setting=setting_dict["exp_setting"]
        print(exp_setting)
        if setting_dict["if_use_hm"]:
            hm_type=setting_dict["hm_type"]
            if setting_dict["model_type"]=="mmCLIP_gpt_v2":
                assert "Please provide a valid model_type"
            elif setting_dict["model_type"]=="mmCLIP_gpt_multi_branch":
                assert "Please provide a valid model_type"
            elif setting_dict["model_type"]=="mmCLIP_gpt_multi_head":
                assert "Please provide a valid model_type"
            elif setting_dict["model_type"]=="mmCLIP_gpt_multi_brach_property":
                assert "Please provide a valid model_type"
            elif setting_dict["model_type"]=="mmCLIP_gpt_multi_brach_property_v3":
                mmclip = mmCLIP_gpt_multi_brach_property_v3(proj_head_dim=64, if_use_hm_proj=setting_dict["if_use_hm_proj"],
                                       if_use_text_proj=setting_dict["if_use_text_proj"],
                                       if_use_text_att=setting_dict["if_use_text_att"],
                                       if_use_hm_att=setting_dict["if_use_hm_att"],
                                       if_use_hm=setting_dict["if_use_hm"],
                                       device=device,
                                       in_channels=len(hm_type)).to(device)
            elif setting_dict["model_type"]=="mmCLIP_gpt_multi_brach_property_with_image":
                assert "Please provide a valid model_type"
            else:
                assert "Please provide a valid model_type"

        else:
            assert "Please provide a valid model_type"

        if setting_dict["if_use_babel_pretrain"]:
            babel_checkpoint_path=setting_dict["epoch_name"]
            model_keys=torch.load(babel_checkpoint_path)
            # mmclip.load_state_dict(model_keys, strict=True)
            mmclip.load_state_dict(model_keys, strict=False)

        if setting_dict["if_freeze_heatmap_encoder"]:
            for param in mmclip.heatmap_encoder.parameters():
                param.requires_grad = False
        # print_trainable_parameters(mmclip)

        if setting_dict["if_lora_ft"]:
            config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=["qkv"],
                lora_dropout=0.1,
                bias="none",
                modules_to_save=["classifier"],
            )
            if setting_dict["model_type"]=="mmCLIP_gpt_multi_head":#get_peft_model work in-place
                lora_model = get_peft_model(mmclip.heatmap_encoder5, config)
                lora_model = get_peft_model(mmclip.heatmap_encoder1, config)
                lora_model = get_peft_model(mmclip.heatmap_encoder2, config)
                lora_model = get_peft_model(mmclip.heatmap_encoder3, config)
                lora_model = get_peft_model(mmclip.heatmap_encoder4, config)
            elif (setting_dict["model_type"]=="mmCLIP_gpt_multi_brach_property"
                  or setting_dict["model_type"]=="mmCLIP_gpt_multi_brach_property_with_image"):
                lora_model = get_peft_model(mmclip.heatmap_encoder, config)#do not need a return
            elif setting_dict["model_type"]=="mmCLIP_gpt_multi_brach_property_v3":
                lora_model = get_peft_model(mmclip.heatmap_encoder, config)#do not need a return
                lora_model = get_peft_model(mmclip.hm_self_attention, config)
                lora_model = get_peft_model(mmclip.text_self_attention, config)
                # for param in mmclip.text_self_attention.parameters():
                #     param.requires_grad = False
                # mmclip= get_peft_model(mmclip, config)#not include logitscale, hm_self_attn
            else:
                raise NotImplementedError
            # print_trainable_parameters(lora_model)
        optimizer = torch.optim.Adam([{'params': mmclip.parameters(), 'lr':setting_dict["lr"]}])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        loss_KL = KLLoss()
        loss_img=CE_Loss()
        loss_cos=Cos_loss()
        loss_mse=MSE_Loss()
        exp_folders_list=[]

        print_trainable_parameters(mmclip)

        if setting_dict["if_use_hm"]:
            train_classes_real = setting_dict["train_classes_real"]
            if setting_dict["train_classes_num"]:
                train_classes_real=random.sample(train_classes_real, setting_dict["train_classes_num"])

            setting_dict["used_train_classes_real"]=train_classes_real
            print("used_train_classes_real", train_classes_real)

            ds = local_dataset(trial_list=setting_dict["trial_list"], query_classes=train_classes_real,
                              data_location=setting_dict["local_train_data_location"],
                               gpt_data_location=setting_dict["gpt_data_location"],
                              crop_size=setting_dict["crop_size"],ratio=setting_dict["train_ratio"], order=setting_dict["train_order"],
                               img_size=setting_dict["img_size"], sampling_gap=setting_dict["train_sampling_gap"])
            if setting_dict["if_babel_cotrain"]:
                # ds_babel = babel_dataset(data_paths=setting_dict["babel_train_data_location"],
                #                    dataset_list=setting_dict["dataset_list"], crop_size=setting_dict["crop_size"],
                #                    img_size=setting_dict["img_size"])
                ds_babel = babel_dataset_gpt(data_paths=setting_dict["babel_train_data_location"],
                                       label_dict_path=setting_dict["label_dict_path"],
                                       dataset_list=setting_dict["dataset_list"],
                                       gpt_data_location=setting_dict["babel_gpt_data_location"],
                                       crop_size=setting_dict["crop_size"],
                                       img_size=setting_dict["img_size"],
                                       if_range_aug=setting_dict["if_range_aug"],
                                       if_use_gpt=setting_dict["if_use_gpt"],
                                       aug_ratio=setting_dict["aug_ratio"])
                ds = ConcatDataset([ds, ds_babel])
            if setting_dict["if_humanml3d_cotrain"]:
                ds_humanml3d=HumanML3DDataset(data_paths=setting_dict["humanml3d_train_data_location"],
                                              text_paths=setting_dict["humanml3d_text_paths"],
                                              csv_path=setting_dict["humanml3d_cvs_paths"],
                                              dataset_list=setting_dict["dataset_list"],
                                              crop_size=setting_dict["crop_size"],
                                              img_size=setting_dict["img_size"], aug_ratio=setting_dict["aug_ratio"],
                                              if_range_aug=setting_dict["if_range_aug"])
                ds = ConcatDataset([ds, ds_humanml3d])

            dl_train = DataLoader(ds, collate_fn=collate_fn, batch_size=setting_dict["batch_size"], shuffle=True, drop_last=True, num_workers=4, prefetch_factor=2)
            dl_iter_train = iter(dl_train)

            test_classes_real = setting_dict["test_classes_real"]
            print("used_test_classes_real", test_classes_real)
            ds_val = local_dataset(trial_list=setting_dict["trial_list"], query_classes=test_classes_real,
                                  data_location=setting_dict["local_test_data_location"],
                                   gpt_data_location=setting_dict["gpt_data_location"],
                                   crop_size=setting_dict["crop_size"],img_size=setting_dict["img_size"],
                                   ratio=setting_dict["test_ratio"], order=setting_dict["test_order"], sampling_gap=setting_dict["test_sampling_gap"])
            dl_val = DataLoader(ds_val, collate_fn=collate_fn, batch_size=10, shuffle=False, drop_last=False, num_workers=4, prefetch_factor=2)

            if setting_dict["if_few_shot"] or setting_dict["if_linear_prob"]:
                ds_fs=local_dataset_fs(trial_list=setting_dict["trial_list"], query_classes=test_classes_real,
                                  data_location=setting_dict["local_test_data_location"],
                                   gpt_data_location=setting_dict["gpt_data_location"],
                                   crop_size=setting_dict["crop_size"],img_size=setting_dict["img_size"],
                                   ratio=setting_dict["test_ratio"], order=setting_dict["test_order"], sampling_gap=setting_dict["fs_sampling_gap"],
                                start_index=setting_dict["fs_start_index"], few_shot_sample=setting_dict["fs_sample"])
                fs_hm_list=ds_fs.get_eval()


        else:
            train_classes_real = setting_dict["train_classes_real"]
            ds = local_dataset_pc(trial_list=setting_dict["trial_list"], query_classes=train_classes_real,
                               data_location=setting_dict["local_train_data_location"],
                               gpt_data_location=setting_dict["gpt_data_location"],
                               pc_crop_size=setting_dict["pc_crop_size"],ratio=setting_dict["train_ratio"], order=setting_dict["train_order"],
                               sampling_gap=setting_dict["train_sampling_gap"])
            dl_train = DataLoader(ds, collate_fn=collate_fn, batch_size=setting_dict["batch_size"], shuffle=True,
                                  drop_last=True, num_workers=4, prefetch_factor=2)
            dl_iter_train = iter(dl_train)

            test_classes_real = setting_dict["test_classes_real"]
            ds_val = local_dataset_pc(trial_list=setting_dict["trial_list"], query_classes=test_classes_real,
                                   data_location=setting_dict["local_test_data_location"],
                                   gpt_data_location=setting_dict["gpt_data_location"],
                                   pc_crop_size=setting_dict["pc_crop_size"],
                                   ratio=setting_dict["test_ratio"], order=setting_dict["test_order"], sampling_gap=setting_dict["test_sampling_gap"])
            dl_val = DataLoader(ds_val, collate_fn=collate_fn, batch_size=10, shuffle=False, drop_last=False,
                                num_workers=4, prefetch_factor=2)
            if setting_dict["if_few_shot"]:
                assert NotImplementedError

        iteration_num = setting_dict["iteration_num"]#10000
        iteration = 0
        if not os.path.isdir("./src/{}/".format(exp_name)):
            os.mkdir("./src/{}/".format(exp_name))
        if not os.path.isdir("./src/{}/{}/".format(exp_name, exp_setting)):
            os.mkdir("./src/{}/{}/".format(exp_name, exp_setting))
        if not os.path.isdir("./src/{}/{}/confusion_matrix/".format(exp_name, exp_setting)):
            os.mkdir("./src/{}/{}/confusion_matrix/".format(exp_name, exp_setting))
        log_file = open("./src/{}/{}/log_unseen.txt".format(exp_name, exp_setting), "w+")
        for key, value in setting_dict.items():
            log_file.writelines("{}:  {}\n".format(key, value))
        if not os.path.isdir("./src/{}/{}/checkpoint_unseen".format(exp_name, exp_setting)):
            os.mkdir("./src/{}/{}/checkpoint_unseen".format(exp_name, exp_setting))
        test_acc_list=[]
        while iteration <= iteration_num:
            if iteration % 200 == 0 or (iteration%50==0 and iteration<600):
                mmclip.eval()
                top1_correct = 0
                top2_correct = 0
                total = 0
                with torch.no_grad():
                    label_list = []
                    pred_list = []
                    eval_text_emd = mmclip.cal_text_features_2d(ds_val.inference_description_list)[test_classes_real]
                    eval_text_feature = eval_text_emd / eval_text_emd.norm(dim=-1, keepdim=True)
                    if setting_dict["if_few_shot"]:
                        fs_hm_emds = []
                        if setting_dict["if_use_hm"]:
                            for i, hms in enumerate(fs_hm_list):
                                fs_hm_array = torch.from_numpy(hms[:, hm_type, ...]).float().to(device)
                                fs_hm_emd, scores = mmclip.cal_hm_features(fs_hm_array)
                                fs_hm_emds.append(torch.mean(fs_hm_emd, dim=0))
                        else:
                            # fs_hm_emds_list=[]
                            # label_list=[]
                            # for i, hms in enumerate(fs_hm_list):
                            #     fs_hm_array = torch.from_numpy(hms[:, hm_type, ...]).float().to(device)
                            #     fs_hm_emd, scores = mmclip.cal_hm_features(fs_hm_array)
                            #     fs_hm_emd=fs_hm_emd[:,-1,:]
                            #     labels=[[i]*len(fs_hm_emds)]
                            #     fs_hm_emds_list.append(fs_hm_emd)
                            #     label_list.append(labels)
                            # classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
                            # classifier.fit(fs_hm_emd, train_labels)
                            assert NotImplementedError
                        fs_hm_emds = torch.stack(fs_hm_emds)
                    elif setting_dict["if_linear_prob"]:
                        assert NotImplementedError
                    tsne_hm_list=[]
                    tsne_label_list=[]
                    for i, (hms, _, _, labels) in tqdm(enumerate(dl_val), desc="Computing batch"):
                        if setting_dict["if_use_hm"]:
                            eval_hm_array = torch.from_numpy(hms[:, hm_type, ...]).float().to(device)
                        else:
                            eval_hm_array = torch.from_numpy(hms).float().to(device)
                        ## get features
                        eval_hm_emd, scores = mmclip.cal_hm_features(eval_hm_array)
                        ## normalize
                        eval_hm_feature = eval_hm_emd / eval_hm_emd.norm(dim=-1, keepdim=True)
                        ## get prob and class label
                        logit_scale = mmclip.logit_scale.exp()
                        logits_hm_text = logit_scale * eval_hm_feature[:,-1,:] @ eval_text_feature[:,-1,:].t()
                        probs = logits_hm_text.softmax(dim=1).detach().cpu()
                        if setting_dict["if_few_shot"]:
                            fs_hm_emds = fs_hm_emds / fs_hm_emds.norm(dim=-1, keepdim=True)
                            logits_hm_fs = logit_scale * eval_hm_feature[:,-1,:] @ fs_hm_emds[:,-1,:].t()
                            probs_hm_fs = logits_hm_fs.softmax(dim=1).detach().cpu()
                            probs = setting_dict["fs_text_weight"] * probs + (
                                        1 - setting_dict["fs_text_weight"]) * probs_hm_fs
                        if setting_dict["if_linear_prob"]:
                            assert "linear prob not implemented yet"
                        _, eval_pred_top1 = torch.topk(probs, k=1, dim=1)
                        if len(setting_dict["test_classes_real"])>=2:
                            _, eval_pred_top2 = torch.topk(probs, k=2, dim=1)
                        else:
                            eval_pred_top2=None
                        ## cal acc
                        for j in range(len(logits_hm_text)):
                            label_list.append(test_classes_real[labels[j][0]])
                            pred_list.append(test_classes_real[eval_pred_top1[j].numpy()[0]])
                            total += 1
                            if any([eval_label in eval_pred_top1[j].numpy() for eval_label in labels[j]]):
                                top1_correct += 1
                            if eval_pred_top2!=None:
                                if any([eval_label in eval_pred_top2[j].numpy() for eval_label in labels[j]]):
                                    top2_correct += 1
                            else:
                                top2_correct=0
                    cm = confusion_matrix(label_list, pred_list)
                    # cm=cm.astype('float')/cm.astype('float').sum(axis=1)[:,np.newaxis]
                    # np.savetxt("./src/{}/{}/confusion_matrix/{:05d}_cm.csv".
                    #            format(exp_name, exp_setting, iteration)
                    #            , cm, delimiter=",",fmt="%.2f")
                    plt.clf()
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_classes_real)
                    disp.plot()
                    plt.savefig(
                        "./src/{}/{}/confusion_matrix/{:05d}_cm.png".format(exp_name, exp_setting, iteration))
                    plt.close()


                test_acc_list.append(top1_correct / total)
                acc_quantile=np.quantile(test_acc_list, .90)
                print("top1 acc:{:5f}, top2 acc:{:5f}, 90 quantile acc:{:5f}, max acc:{:5f}"
                      .format(top1_correct / total,top2_correct / total, acc_quantile, max(test_acc_list)))
                log_file.writelines(
                    "Iteration {}, top1 acc:{:5f}, top2 acc:{:5f}, 90 quantile acc:{:5f}, max acc:{:5f} \n"
                    .format(iteration, top1_correct / total,top2_correct / total, acc_quantile, max(test_acc_list)))
                log_file.flush()
                if setting_dict["if_save_model"]:
                    torch.save(mmclip.heatmap_encoder.state_dict(),
                               "./src/{}/{}/checkpoint_unseen/{:05d}_checkpoint.pt"
                               .format(exp_name, exp_setting, iteration))
            try:
                hms, _, texts, _ = next(dl_iter_train)
            except StopIteration:
                print("new epoch")
                log_file.writelines("new epoch\n")
                dl_iter_train = iter(dl_train)
                hms, _, texts, _ = next(dl_iter_train)
                scheduler.step()
            if setting_dict["if_use_hm"]:
                hms = torch.from_numpy(hms[:, hm_type, ...]).float().to(device)##
            else:
                hms = torch.from_numpy(hms).float().to(device)
            mmclip.train()
            optimizer.zero_grad()

            hm_emds, scores = mmclip.cal_hm_features(hms)
            text_emds = mmclip.cal_text_features_2d(texts)
            logit_scale = mmclip.logit_scale.exp()
            hm_features = hm_emds / hm_emds.norm(dim=-1, keepdim=True)
            text_features = text_emds / text_emds.norm(dim=-1, keepdim=True)


            all_loss=0
            for i in range(hm_features.shape[1]):
                logits_hm_text = logit_scale * hm_features[:,i,:] @ text_features[:,i,:].t()
                if setting_dict["loss_type"] == "ce":
                    ground_truth = torch.arange(len(hms)).to(device)
                    loss_imgs = loss_img(logits_hm_text, ground_truth)
                    loss_text=loss_img(logits_hm_text.t(), ground_truth)
                    total_loss = (loss_imgs+loss_text)/2
                elif setting_dict["loss_type"] == "kl":
                    ground_truth = torch.tensor(gen_label(np.array(texts)[:,0]), dtype=hm_features.dtype, device=device)
                    loss_hm_text = loss_KL(logits_hm_text, ground_truth)
                    total_loss = loss_hm_text
                elif setting_dict["loss_type"] == "cos":
                    total_loss=loss_cos(text_features[:,i,:], hm_features[:,i,:])
                elif setting_dict["loss_type"] == "mse":
                    total_loss = loss_mse(text_emds[:,i,:], hm_emds[:,i,:])
                else:
                    assert "Please provide a valid loss function"
                all_loss+=total_loss
            all_loss.backward()
            optimizer.step()

            if iteration % 200 == 0:
                # for line in logits_per_image.softmax(dim=1).detach().cpu().numpy():
                #     print(line)
                print("iteration:{}, loss:{:5f}".format(iteration, total_loss.item()))
                log_file.writelines("iteration:{}, loss:{:5f}\n".format(iteration, total_loss.item()))
                log_file.flush()
            iteration += 1