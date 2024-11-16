import sys
sys.path.append(".")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from model_gpt import mmCLIP_gpt_multi_brach_property_v3
from dataset import collate_fn, babel_dataset_gpt, local_dataset, HumanML3DDataset
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
from config_babel_gpt_text import setting_list_babel
from torch.nn import functional as F
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
    gt = np.zeros(shape=(num, num))
    for i, label in enumerate(labels):
        for k in range(num):
            if labels[k] == label:
                gt[i, k] = 1
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
        loss = self.error_metric(probs1, probs2)  # * batch_size
        return loss
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

np.set_printoptions(suppress=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device('cuda:0')

if __name__ == "__main__":
    exp_name = "babel_0505_5set"
    for setting_dict in setting_list_babel:
        exp_setting = setting_dict["exp_setting"]
        print(exp_setting)
        hm_type = setting_dict["hm_type"]

        if not setting_dict["if_use_img"]:
            if setting_dict["model_type"]=="mmCLIP_gpt_multi_head":
                assert "Please provide a valid model_type"
            elif setting_dict["model_type"]=="mmCLIP_gpt_multi_brach_property":
                assert "Please provide a valid model_type"
            elif setting_dict["model_type"] == "mmCLIP_gpt_multi_brach_property_v3":
                mmclip = mmCLIP_gpt_multi_brach_property_v3(proj_head_dim=64,
                                                         if_use_hm_proj=setting_dict["if_use_hm_proj"],
                                                         if_use_text_proj=setting_dict["if_use_text_proj"],
                                                         if_use_text_att=setting_dict["if_use_text_att"],
                                                         if_use_hm_att=setting_dict["if_use_hm_att"],
                                                         if_use_hm=setting_dict["if_use_hm"],
                                                         device=device,
                                                         in_channels=len(hm_type)).to(device)
            else:
                assert "Please provide a valid model_type"
        else:
            if setting_dict["model_type"] == "mmCLIP_gpt_multi_head":
                assert "Please provide a valid model_type"
            elif setting_dict["model_type"]=="mmCLIP_gpt_multi_brach_property":
                assert "Please provide a valid model_type"
            else:
                assert "Please provide a valid model_type"
        print_trainable_parameters(mmclip)

        optimizer = torch.optim.Adam([{'params': mmclip.parameters(), 'lr': setting_dict['lr']}])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=setting_dict['gamma'])
        loss_KL = KLLoss()
        loss_ce = CE_Loss()
        loss_cos = Cos_loss()
        loss_mse = MSE_Loss()
        exp_folders_list = []

        ds= HumanML3DDataset(data_paths="temp")
        if setting_dict['if_use_babel']:
            ds_babel = babel_dataset_gpt(data_paths=setting_dict["babel_train_data_location"],
                                   label_dict_path=setting_dict["label_dict_path"],
                                   dataset_list=setting_dict["dataset_list"],
                                   gpt_data_location=setting_dict["babel_gpt_data_location"],
                                   crop_size=setting_dict["crop_size"],
                                   img_size=setting_dict["img_size"],
                                   if_range_aug=setting_dict['if_range_aug'],
                                   if_use_gpt=setting_dict["if_use_gpt"],
                                   if_use_img=setting_dict["if_use_img"])
            ds=ConcatDataset([ds, ds_babel])
        if setting_dict['if_use_sim_local']:
            train_classes_real = setting_dict["train_classes_real"]
            ds_local = local_dataset(trial_list=setting_dict["trial_list"], query_classes=train_classes_real,
                               data_location=setting_dict["local_train_data_location"],
                               gpt_data_location=setting_dict["gpt_data_location"],
                               crop_size=setting_dict["crop_size"], ratio=setting_dict["train_ratio"],
                               order=setting_dict["train_order"],
                               img_size=setting_dict["img_size"], sampling_gap=setting_dict["train_sampling_gap"],
                               if_range_aug=setting_dict["if_range_aug"])
            ds = ConcatDataset([ds, ds_local])
        if setting_dict["if_use_humanml3d"]:
            ds_humanml3d = HumanML3DDataset(data_paths=setting_dict["humanml3d_train_data_location"],
                                            text_paths=setting_dict["humanml3d_text_paths"],
                                            gpt_data_location=setting_dict["humanml3d_gpt_location"],
                                            csv_path=setting_dict["humanml3d_cvs_paths"],
                                            dataset_list=setting_dict["dataset_list"],
                                            crop_size=setting_dict["crop_size"],
                                            img_size=setting_dict["img_size"],
                                            if_use_gpt=setting_dict["if_use_gpt"],
                                            if_range_aug=setting_dict["if_range_aug"])
            ds = ConcatDataset([ds, ds_humanml3d])
        if setting_dict["if_use_t2m"]:
            pass

        dl_train = DataLoader(ds, collate_fn=collate_fn, batch_size=setting_dict["batch_size"], shuffle=True,
                              drop_last=True, num_workers=4, prefetch_factor=2)
        dl_iter_train = iter(dl_train)

        test_class_list = setting_dict["test_class_list"]
        ds_dl_val_list = []
        for test_class in test_class_list:
            ds_val = local_dataset(trial_list=setting_dict["trial_list"], query_classes=test_class,
                                   data_location=setting_dict["local_test_data_location"],
                                   gpt_data_location=setting_dict["gpt_data_location"],
                                   crop_size=setting_dict["crop_size"], img_size=setting_dict["img_size"],
                                   ratio=setting_dict["test_ratio"], order=setting_dict["test_order"],
                                   sampling_gap=setting_dict["test_sampling_gap"])
            dl_val = DataLoader(ds_val, collate_fn=collate_fn, batch_size=10, shuffle=False, drop_last=False,
                                num_workers=4,
                                prefetch_factor=2)
            ds_dl_val_list.append([ds_val, dl_val])

        iteration_num = setting_dict['iteration_num']
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
        test_acc_list = []
        avg_test_acc_list=[]
        for i in range(len(ds_dl_val_list)):
            test_acc_list.append([])

        while iteration <= iteration_num:
            if iteration%1000==1:
                scheduler.step()

            if iteration % 200 == 0 or (iteration % 50 == 0 and iteration < 600):
                mmclip.eval()
                top1_list=[]
                for i_ds, (ds_val,dl_val) in enumerate(ds_dl_val_list):
                    top1_correct = 0
                    top2_correct = 0
                    total = 0
                    with torch.no_grad():
                        label_list = []
                        pred_list = []
                        eval_text_emd = mmclip.cal_text_features_2d(ds_val.inference_description_list)[
                            test_class_list[i_ds]]
                        eval_text_feature = eval_text_emd / eval_text_emd.norm(dim=-1, keepdim=True)
                        for i, (hms, _, _, labels) in tqdm(enumerate(dl_val), desc="Computing batch"):
                            eval_hm_array = torch.from_numpy(hms[:, hm_type, ...]).float().to(device)
                            ## get features
                            eval_hm_emd, _ = mmclip.cal_hm_features(eval_hm_array)
                            ## normalize
                            eval_hm_feature = eval_hm_emd / eval_hm_emd.norm(dim=-1, keepdim=True)
                            ## get prob and class label
                            logit_scale = mmclip.logit_scale.exp()
                            logits_per_image = logit_scale * eval_hm_feature[:,-1,:] @ eval_text_feature[:,-1,:].t()
                            # logits_per_image = 100 * eval_hm_feature @ eval_text_feature.t()
                            probs = logits_per_image.softmax(dim=1).detach().cpu()
                            _, eval_pred_top1 = torch.topk(probs, k=1, dim=1)
                            if len(test_class_list[i_ds]) >= 2:
                                _, eval_pred_top2 = torch.topk(probs, k=2, dim=1)
                            else:
                                eval_pred_top2 = None
                            ## cal acc
                            for j in range(len(logits_per_image)):
                                label_list.append(test_class_list[i_ds][labels[j][0]])
                                pred_list.append(test_class_list[i_ds][eval_pred_top1[j].numpy()[0]])
                                total += 1
                                if any([eval_label in eval_pred_top1[j].numpy() for eval_label in labels[j]]):
                                    top1_correct += 1
                                if eval_pred_top2 != None:
                                    if any([eval_label in eval_pred_top2[j].numpy() for eval_label in labels[j]]):
                                        top2_correct += 1
                                else:
                                    top2_correct = 0
                        cm = confusion_matrix(label_list, pred_list)
                        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_class_list[i_ds])
                        disp.plot()
                        plt.savefig(
                            "./src/{}/{}/confusion_matrix/{:05d}_cm_{:02d}.png".format(exp_name, exp_setting, iteration, i_ds))
                        plt.close()
                        test_acc_list[i_ds].append(top1_correct / total)
                        acc_quantile = np.quantile(test_acc_list[i_ds], .90)
                        top1_list.append(top1_correct / total)
                        # print("{:02d}th list, top1 acc:{:5f}, top2 acc:{:5f}, 90 quantile acc:{:5f}, max acc:{:5f}"
                        #       .format(i_ds,top1_correct / total, top2_correct / total, acc_quantile, max(test_acc_list[i_ds])))
                        log_file.writelines(
                            "Iteration {}, {:02d}th list, top1 acc:{:5f}, top2 acc:{:5f}, 90 quantile acc:{:5f}, max acc:{:5f} \n"
                            .format(iteration, i_ds, top1_correct / total, top2_correct / total, acc_quantile,
                                    max(test_acc_list[i_ds])))

                avg_test_acc_list.append(np.mean(top1_list))
                avg_acc_quantile = np.quantile(avg_test_acc_list, .90)
                log_file.writelines("Iteration {}, top 1 avg {}, 90 quantile acc:{:5f}, max acc:{:5f} \n".format(iteration, np.mean(top1_list), avg_acc_quantile, max(avg_test_acc_list)))
                log_file.flush()
                torch.save(mmclip.state_dict(), "./src/{}/{}/checkpoint_unseen/{:05d}_checkpoint.pt".format(exp_name, exp_setting, iteration))

            try:
                hms, r_imgs, texts, _ = next(dl_iter_train)
            except StopIteration:
                print("new epoch")
                dl_iter_train = iter(dl_train)
                hms, r_imgs, texts, _ = next(dl_iter_train)
                # scheduler.step()

            hms = torch.from_numpy(hms[:, hm_type, ...]).float().to(device)  ##
            mmclip.train()
            optimizer.zero_grad()

            hm_emds, _ = mmclip.cal_hm_features(hms)
            text_emds = mmclip.cal_text_features_2d(texts)

            logit_scale = mmclip.logit_scale.exp()
            hm_features = hm_emds / hm_emds.norm(dim=-1, keepdim=True)
            text_features = text_emds / text_emds.norm(dim=-1, keepdim=True)



            all_loss = 0
            for i in range(hm_features.shape[1]):
                logits_hm_text = logit_scale * hm_features[:, i, :] @ text_features[:, i, :].t()
                if setting_dict["loss_type"] == "ce":
                    ground_truth = torch.arange(len(hms)).to(device)
                    loss_imgs = loss_ce(logits_hm_text, ground_truth)
                    loss_text = loss_ce(logits_hm_text.t(), ground_truth)
                    total_loss = (loss_imgs + loss_text) / 2
                    if setting_dict["if_use_img"] and i==hm_features.shape[1]-1:
                        r_imgs_embds = mmclip.cal_img_features(r_imgs)
                        r_img_features = r_imgs_embds / r_imgs_embds.norm(dim=-1, keepdim=True)
                        logits_per_hm_img = logit_scale * hm_features[:, i, :] @ r_img_features.t()
                        loss_hm_img = loss_ce(logits_per_hm_img, ground_truth)
                        total_loss+= setting_dict["img_loss_ratio"]*loss_hm_img
                elif setting_dict["loss_type"] == "kl":
                    ground_truth = torch.tensor(gen_label(np.array(texts)[:, 0]), dtype=hm_features.dtype,
                                                device=device)
                    loss_hm_text = loss_KL(logits_hm_text, ground_truth)
                    total_loss = loss_hm_text
                    if setting_dict["if_use_img"] and i==hm_features.shape[1]-1:
                        r_imgs_embds = mmclip.cal_img_features(r_imgs)
                        r_img_features = r_imgs_embds / r_imgs_embds.norm(dim=-1, keepdim=True)
                        logits_per_hm_img = logit_scale * hm_features[:, i, :] @ r_img_features.t()
                        loss_hm_img = loss_KL(logits_per_hm_img, ground_truth)
                        total_loss+= setting_dict["img_loss_ratio"]*loss_hm_img

                elif setting_dict["loss_type"] == "cos":
                    total_loss = loss_cos(text_features[:, i, :], hm_features[:, i, :])
                    if setting_dict["if_use_img"] and i==hm_features.shape[1]-1:
                        r_imgs_embds = mmclip.cal_img_features(r_imgs)
                        r_img_features = r_imgs_embds / r_imgs_embds.norm(dim=-1, keepdim=True)
                        loss_hm_img = loss_cos(r_img_features, hm_features[:, i, :])
                        total_loss+= setting_dict["img_loss_ratio"]*loss_hm_img
                elif setting_dict["loss_type"] == "mse":
                    total_loss = loss_mse(text_emds[:, i, :], hm_emds[:, i, :])
                else:
                    assert "Please provide a valid loss function"
                all_loss += total_loss
            all_loss.backward()
            optimizer.step()

            if iteration % 200 == 0:
                print("iteration:{}, loss:{:5f}".format(iteration, total_loss.item()))
                log_file.writelines("iteration:{}, loss:{:5f}\n".format(iteration, total_loss.item()))
            iteration += 1