import torch
from tqdm import tqdm
from utils import pyutils
from loss import IouSemanticAwareLoss, structure_loss_ms3, A_MaskedV_SimmLoss, structure_loss_s4, F1_IoU_BCELoss


def train_one_epoch(cfg, epoch, train_dataloader, audio_backbone, model, optimizer):
    avg_total_loss, avg_sup_loss, avg_latent_loss = pyutils.AvgMeter(), pyutils.AvgMeter(), pyutils.AvgMeter()
    progress_bar = tqdm(train_dataloader, ncols=100, desc='Epoch[{:03d}/{:03d}]'.format(epoch + 1, cfg.PARAM.EPOCHS))
    if cfg.TRAIN.TASK == "S4":
        structure_loss, sim_loss_rate = structure_loss_s4, 0
    elif cfg.TRAIN.TASK == "MS3":
        structure_loss, sim_loss_rate = structure_loss_ms3, cfg.PARAM.SIM_LOSS
    for n_iter, batch_data in enumerate(progress_bar):
        optimizer.zero_grad()
        imgs, audio, mask, _ = batch_data  # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]

        imgs, audio, mask = imgs.cuda(), audio.cuda(), mask.cuda()
        B, T, C, H, W = imgs.shape
        imgs = imgs.view(B * T, C, H, W)
        if cfg.TRAIN.TASK == "MS3":
            mask = mask.view(B * T, 1, H, W)
        else:
            mask = mask.view(B, 1, H, W)
        audio = audio.view(-1, audio.shape[2], audio.shape[3], audio.shape[4])  # [B*T, 1, 96, 64]
        with torch.no_grad():
            audio_feature = audio_backbone(audio)  # [B*T, 128]

        output, visual_feat_list, audio_feat_list, latent_loss = model(imgs, audio_feature, mask)  # [bs*5, 1, 224, 224]
        sup_loss = structure_loss(output, mask)
        sa_loss = A_MaskedV_SimmLoss(output, audio_feat_list, visual_feat_list, 'avg', kl_flag=True)
        loss = sup_loss + sim_loss_rate * sa_loss + 1.0 * latent_loss
        avg_total_loss.update(loss.data)
        avg_sup_loss.update(sup_loss.data)
        avg_latent_loss.update(latent_loss.data)
        # import pdb; pdb.set_trace()
        loss.backward()
        optimizer.step()
        progress_bar.set_postfix(
            loss=f"{avg_total_loss.show():.3f}|{avg_sup_loss.show():.3f}|{avg_latent_loss.show():.3f}")

    return model
