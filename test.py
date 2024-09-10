# -*- coding: utf-8 -*-
'''
* @author [haodawei]
* @version [2023/8/17]
* @description [test under the MS3/S4 setting]
* @code [test.py]
'''
import os
import time
import torch
import logging

from config import cfg
from torchvggish.vggish import VGGish

from utils import pyutils
from utils.utility import logger, mask_iou, save_mask, save_mask_ms3, Eval_Fmeasure
from utils.system import setup_logging
from trainer import train_one_epoch
from datasets.get_loader import get_loader
from torch.optim import lr_scheduler
from model.AVSModel import AVSModel
import h5py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np


class audio_extractor(torch.nn.Module):
    def __init__(self, cfg, device):
        super(audio_extractor, self).__init__()
        self.audio_backbone = VGGish(cfg, device)

    def forward(self, audio):
        audio_fea = self.audio_backbone(audio)
        return audio_fea


if __name__ == "__main__":
    cfg.MODEL.TRAINED = "/mnt/SSD/avs_bidirectional_generation/SeqMotionAVS/experiments/S4_20230624-205910debug/checkpoints/S4_best.pth"
    # cfg.MODEL.TRAINED = "/workspace/avs_bidirectional_generation/SeqMotionAVS/experiments/MS3_20230625-063648debug/checkpoints/MS3_best.pth"
    # cfg.MODEL.TRAINED = "/workspace/avs_bidirectional_generation/SeqMotionAVS/experiments/MS3_20230624-184135debug/checkpoints/MS3_best.pth"
    #
    pyutils.set_seed(cfg.PARAM.SEED)

    save_mask_path = os.path.join(cfg.TRAIN.LOG_DIR, cfg.MODEL.TRAINED.split("/")[-3], "h5py")
    if not os.path.exists(save_mask_path): os.makedirs(save_mask_path, exist_ok=True)
    # logger = logging.getLogger(__name__)
    print('==> Config: {}'.format(cfg))

    # Model
    model = AVSModel(config=cfg)
    model = model.cuda().eval()
    model.load_state_dict(torch.load(cfg.MODEL.TRAINED))
    print("Load pretrained model: ".format(cfg.MODEL.TRAINED))

    param_count = sum(x.numel() / 1e6 for x in model.parameters())
    print("Model have {:.4f}Mb paramerters in total".format(param_count))

    # video backbone
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_backbone = audio_extractor(cfg, device).cuda().eval()

    # Data
    train_dataloader, val_dataloader = get_loader(cfg)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), cfg.PARAM.LR)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.PARAM.LR_DECAY_STEP, gamma=cfg.PARAM.LR_DECAY_RATE)
    avg_meter_miou = pyutils.AverageMeter('miou')
    avg_meter_F = pyutils.AverageMeter('F_score')
    TIME = []
    with torch.no_grad():
        for n_iter, batch_data in enumerate(val_dataloader):
            imgs, audio, mask, name_list = batch_data  # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 5, 1, 224, 224]

            imgs, audio, mask = imgs.cuda(), audio.cuda(), mask.cuda()
            B, T, C, H, W = imgs.shape
            imgs = imgs.view(B * T, C, H, W)
            mask = mask.view(B * T, H, W)
            audio = audio.view(-1, audio.shape[2], audio.shape[3], audio.shape[4])
            with torch.no_grad():
                audio_feature = audio_backbone(audio)

            # calculate time
            torch.cuda.synchronize()
            start_time = time.time()
            print(imgs.shape, audio_feature.shape)
            output, s_a, s_v, c_va = model(imgs, audio_feature)  # [bs*5, 1, 224, 224]
            torch.cuda.synchronize()
            end_time = time.time()
            TIME.append(end_time - start_time)
            # s_a, s_v, c_va = latent_code
            # hf_s_a = h5py.File(os.path.join(save_mask_path, "s_a_{}.h5".format(n_iter)), 'w')
            # hf_s_a.create_dataset('s_a', data=s_a.cpu().data.numpy())
            # hf_s_a.close()
            # hf_s_v = h5py.File(os.path.join(save_mask_path, "s_v_{}.h5".format(n_iter)), 'w')
            # hf_s_v.create_dataset('s_v', data=s_v.cpu().data.numpy())
            # hf_s_v.close()
            # hf_c_va = h5py.File(os.path.join(save_mask_path, "c_va_{}.h5".format(n_iter)), 'w')
            # hf_c_va.create_dataset('c_va', data=c_va.cpu().data.numpy())
            # hf_c_va.close()

            if cfg.TRAIN.TASK == "S4":
                save_mask(output, save_mask_path, name_list[0], name_list[1])
            elif cfg.TRAIN.TASK == "MS3":
                save_mask_ms3(output, save_mask_path, name_list[0])

            miou = mask_iou(output.squeeze(1), mask)

            F_score = Eval_Fmeasure(output.squeeze(1), mask)
            # if miou != 0.0:
            avg_meter_miou.add({'miou': miou})
            avg_meter_F.add({'F_score': F_score})
            print('n_iter: {}, iou: {}, F_score: {}'.format(n_iter, miou, F_score), ' ', name_list[0])

        miou = (avg_meter_miou.pop('miou'))
        F_score = (avg_meter_F.pop('F_score'))
        print('test miou:', miou.item())
        print('test F_score:', F_score)
    print(miou)
    # import numpy as np
    #
    print(np.mean(TIME))
