import os
import time
import torch
import logging

from config import cfg
from torchvggish.vggish import VGGish

from utils import pyutils
from utils.utility import mask_iou
from utils.system import setup_logging
from trainer import train_one_epoch
from datasets.get_loader import get_loader
from torch.optim import lr_scheduler
from model.AVSModel import AVSModel


class audio_extractor(torch.nn.Module):
    def __init__(self, cfg, device):
        super(audio_extractor, self).__init__()
        self.audio_backbone = VGGish(cfg, device)

    def forward(self, audio):
        audio_fea = self.audio_backbone(audio)
        return audio_fea


if __name__ == "__main__":
    pyutils.set_seed(cfg.PARAM.SEED)

    # Log directory
    if not os.path.exists(cfg.TRAIN.LOG_DIR):
        os.makedirs(cfg.TRAIN.LOG_DIR, exist_ok=True)
    # Logs
    log_dir = os.path.join(cfg.TRAIN.LOG_DIR,
                           '{}'.format(time.strftime(cfg.TRAIN.TASK + '_%Y%m%d-%H%M%S' + cfg.TRAIN.SAVE_NAME)))
    script_path = os.path.join(log_dir, 'scripts')
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    log_path = os.path.join(log_dir, 'log')

    # Save scripts
    if not os.path.exists(script_path): os.makedirs(script_path, exist_ok=True)
    # Checkpoints directory
    if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir, exist_ok=True)
    # Set logger
    if not os.path.exists(log_path): os.makedirs(log_path, exist_ok=True)

    setup_logging(filename=os.path.join(log_path, 'log.txt'))
    logger = logging.getLogger(__name__)
    logger.info('==> Config: {}'.format(cfg))

    # Model
    model = AVSModel(config=cfg)
    model.cuda().train()  # single gpu
    if cfg.MODEL.TRAINED:
        # MS3--load pretrained model
        state_dict = torch.load(cfg.MODEL.TRAINED)
        model.load_state_dict(state_dict)

    # import pdb; pdb.set_trace()
    param_count = sum(x.numel() / 1e6 for x in model.parameters())
    logger.info("Model have {:.4f}Mb paramerters in total".format(param_count))

    # video backbone
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_backbone = audio_extractor(cfg, device).cuda().eval()

    # Data
    train_dataloader, val_dataloader = get_loader(cfg)
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), cfg.PARAM.LR)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.PARAM.LR_DECAY_STEP, gamma=cfg.PARAM.LR_DECAY_RATE)
    avg_meter_miou = pyutils.AverageMeter('miou')

    # Train
    best_epoch, max_miou, miou_list = 0, 0, []
    for epoch in range(cfg.PARAM.EPOCHS):
        model = train_one_epoch(cfg, epoch, train_dataloader, audio_backbone, model, optimizer)
        scheduler.step()

        # Validation:
        model.eval()
        with torch.no_grad():
            for n_iter, batch_data in enumerate(val_dataloader):
                imgs, audio, mask, _ = batch_data  # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 5, 1, 224, 224]

                imgs, audio, mask = imgs.cuda(), audio.cuda(), mask.cuda()
                B, T, C, H, W = imgs.shape
                imgs = imgs.view(B * T, C, H, W)
                mask = mask.view(B * T, H, W)
                audio = audio.view(-1, audio.shape[2], audio.shape[3], audio.shape[4])
                with torch.no_grad():
                    audio_feature = audio_backbone(audio)

                output, _, _, _ = model(imgs, audio_feature)  # [bs*5, 1, 224, 224]

                miou = mask_iou(output.squeeze(1), mask)
                avg_meter_miou.add({'miou': miou})

            miou = (avg_meter_miou.pop('miou'))
            if miou > max_miou:
                model_save_path = os.path.join(checkpoint_dir, '%s_best.pth' % (cfg.TRAIN.TASK))
                # torch.save(model.module.state_dict(), model_save_path)
                torch.save(model.state_dict(), model_save_path)
                best_epoch = epoch
                logger.info('save best model to %s' % model_save_path)

            miou_list.append(miou)
            max_miou = max(miou_list)

            val_log = 'Epoch: {:03d}, Miou: {:.6f}, maxMiou: {:.6f}, LR: {:.6f}'.format(epoch + 1, miou, max_miou,
                                                                                        optimizer.param_groups[0]['lr'])

            logger.info(val_log)

        model.train()
    logger.info('best val Miou {:.6f} at peoch: {:03d}'.format(max_miou, best_epoch))
