import torch
import torch.nn as nn
from model.fusion_module.tpavi import TPAVIModule
from model.fusion_module.avcorr import AVCorr
from model.decoder.get_decoder import get_decoder
from model.encoder.get_encoder import get_encoder
from model.neck.get_neck import get_neck
from utils.torch_utils import torch_L2normalize
import torch.nn.functional as F
from model.blocks.base_blocks import BasicConv2d
from config import cfg
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns


class AVSModel(nn.Module):
    def __init__(self, config=None):
        super(AVSModel, self).__init__()
        self.cfg = config
        self.tpavi_stages = self.cfg.PARAM.TPAVI_STAGES
        self.neck_channel = self.cfg.MODEL.NECK_CHANNEL

        self.encoder, in_channel_list = get_encoder(self.cfg)
        self.neck = get_neck(self.cfg, in_channel_list)
        self.audio_neck = nn.Linear(128, self.cfg.MODEL.NECK_CHANNEL)
        self.decoder = get_decoder(self.cfg)
        self.fusion_modules = nn.ModuleList()
        self.corr_modules = nn.ModuleList()
        self.corr_atten = nn.ModuleList()
        if self.cfg.MODEL.FUSION.lower() == "tpavi":
            fusion_module = TPAVIModule
        elif self.cfg.MODEL.FUSION.lower() == "av_corr":
            fusion_module = AVCorr
        for i in self.tpavi_stages:
            self.fusion_modules.append(fusion_module(in_channels=self.neck_channel))
            # self.corr_modules.append(CorrealationBlock(self.neck_channel*5, None, False, scale=2))
            self.corr_atten.append(nn.Conv2d(1, 1, 1))
        self.mask_encoder = nn.Sequential(
            BasicConv2d(in_planes=128, out_planes=64, kernel_size=1, stride=1, padding=0, norm=False),
            BasicConv2d(in_planes=64, out_planes=64, kernel_size=3, stride=1, padding=1, norm=False),
            BasicConv2d(in_planes=64, out_planes=128, kernel_size=1, stride=1, padding=0, norm=False)
        )
        self.maksed_visual_to_audio = nn.Linear(7*7, 1)

    def visual_corr_attention(self, feature_map_list):
        # perform visual correlation
        corr_feat_list = []
        for scale in range(len(feature_map_list)):
            # the for loop controls the corr scale
            BT, C, H, W = feature_map_list[scale].shape
            B, T = BT // 5, 5
            feat = feature_map_list[scale]
            feat_reshape = feat.view(B, T, *feat.shape[1:4])
            corr_list = []
            for i in range(T):
                if i < 4:
                    B_curr, T_curr, C_curr, H_curr, W_curr = feat_reshape.shape
                    feat1 = feat_reshape[:, i, ...].view(B_curr, C_curr, -1)
                    feat2 = feat_reshape[:, i+1, ...].view(B_curr, C_curr, -1)
                else:
                    feat1 = feat_reshape[:, i, ...].view(B_curr, C_curr, -1)
                    feat2 = feat_reshape[:, i-1, ...].view(B_curr, C_curr, -1)
                # frame-wise correlation operation
                # feat1, feat2 = torch_L2normalize(feat1, d=1), torch_L2normalize(feat2, d=1)
                corr = torch.bmm(feat1.transpose(1, 2), feat2).view(B_curr, H_curr, W_curr, H_curr, W_curr).clamp(min=0)
                corr_list.append(corr.unsqueeze(1))
            corr_cat = torch.cat(corr_list, dim=1).view(BT, 1, H_curr, W_curr, H_curr, W_curr)
            corr_feat_list.append(torch.sigmoid(self.corr_atten[scale](corr_cat.mean(-1).mean(-1))))
        
        # perform visual correlation attention
        corred_feat_list = []
        for scale in range(len(feature_map_list)):
            feat = feature_map_list[scale]
            atten = corr_feat_list[scale]
            corred_feat_list.append(feat + feat*atten)

        return corred_feat_list

    def audio_visual_corr_attention(self, visual_feature_list, audio_feature):
        fused_feature_list, fused_audio_list = [], []
        for i, block in enumerate(self.fusion_modules):
            visual_feat, audio_feat = block(visual_feature_list[i], audio_feature)
            fused_feature_list.append(visual_feat)
            fused_audio_list.append(audio_feat)
        
        return fused_feature_list, fused_audio_list

    def forward(self, x, audio_feature, gt=None):
        feature_map_list = self.encoder(x)
        feature_map_list = self.neck(feature_map_list)

        # mode one: fusion first, then correlation
        fused_feature_list, fused_audio_list = self.audio_visual_corr_attention(feature_map_list, audio_feature)
        for_decode_feature_list = self.visual_corr_attention(fused_feature_list)
        pred = self.decoder(for_decode_feature_list)  # torch.Size([5, 1, 224, 224])
        # pred = self.decoder(fused_feature_list)   # AVSBench with bidirectional generation

        # mode two: fusion first, then correlation
        # corred_feat_list = self.visual_corr_attention(feature_map_list)
        # fused_feature_list, fused_audio_list = self.audio_visual_corr_attention(corred_feat_list, audio_feature)
        # pred = self.decoder(fused_feature_list)

        latent_loss = 0
        #if gt is None:
        if gt is not None:
            # visual_feat_tiny_stage = fused_feature_list[-1]
            visual_feat_tiny_stage = for_decode_feature_list[-1]  # torch.Size([5, 128, 7, 7])
            BT, C, H, W = visual_feat_tiny_stage.shape
            if self.cfg.TRAIN.TASK == "MS3":
                downsample_pred_masks = nn.AdaptiveAvgPool2d((H, W))(gt)
            elif self.cfg.TRAIN.TASK == "S4":
                downsample_pred_masks = nn.AdaptiveAvgPool2d((H, W))(pred)  # torch.Size([5, 1, 7, 7])

            masked_v_map = torch.mul(visual_feat_tiny_stage, downsample_pred_masks)  # torch.Size([5, 128, 7, 7])
            decoded_masked_v_map = self.mask_encoder(masked_v_map)  # torch.Size([5, 128, 7, 7])
            bt, c, h, w = decoded_masked_v_map.shape

            generated_audio_feature = self.maksed_visual_to_audio(decoded_masked_v_map.view(bt, c, h*w))  # torch.Size([5, 128, 1])

            # implicit audio generation
            normed_audio_feature = F.normalize(audio_feature, dim=-1)  # torch.Size([5, 128])
            # ax = sns.heatmap(normed_audio_feature.cpu(), vmax=0.02)
            # plt.show()

            normed_generated_audio_feature = F.normalize(generated_audio_feature.squeeze(), dim=-1)   # torch.Size([5, 128])
            # ax = sns.heatmap(normed_generated_audio_feature.cpu(), vmax=0.02)
            # plt.show()

            # diff_heatmap(normed_audio_feature, normed_generated_audio_feature)
            latent_loss = F.pairwise_distance(normed_audio_feature, normed_generated_audio_feature, p=2).mean()

        # latent_loss = 0
        # for fused_audio in fused_audio_list:
        #     import pdb; pdb.set_trace()
        #     latent_loss += F.kl_div(audio_feature, fused_audio, reduction='mean')
        return pred, fused_feature_list, fused_audio_list, latent_loss


def plot_heatmap(uniform_data):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Plot a heatmap for a numpy array
    ax = sns.heatmap(uniform_data, vmax=0.03)
    plt.show()


def plot_corr(res):
    from string import ascii_letters
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    # import palettable

    # 生成随机数
    # rs = np.random.RandomState(33)  # 类似np.random.seed，即每次括号中的种子33不变，每次可获得相同的随机数
    # d = pd.DataFrame(data=res.normal(size=(5, 128)),  # normal生成高斯分布的概率密度随机数，需要在变量rs下使用
    #                  columns=list(ascii_letters[5:]))

    d = pd.DataFrame(res)
    # corr函数计算相关性矩阵(correlation matrix)
    dcorr = d.corr(method='pearson')  # 默认为'pearson'检验，可选'kendall','spearman'

    plt.figure(figsize=(11, 9), dpi=160)
    sns.heatmap(data=dcorr, vmax=0.3,)
    plt.show()


def diff_heatmap(tensor1, tensor2):
    import torch
    import seaborn as sns
    import matplotlib.pyplot as plt

    # 生成两个5x128的张量
    # tensor1 = torch.rand(5, 128)
    # tensor2 = torch.rand(5, 128)

    # 计算两个张量之间的差异
    diff_tensor = torch.abs(tensor1.cpu() - tensor2.cpu())

    # 绘制热力图
    sns.heatmap(diff_tensor, cmap='coolwarm', vmax=0.02)
    plt.show()


if __name__ == "__main__":
    # imgs = torch.randn(10, 3, 224, 224).cuda()
    # gt = torch.randn(2, 1, 224, 224).cuda()
    # audio = torch.randn(10, 128).cuda()
    # # model = Pred_endecoder(channel=256)
    # model = AVSModel(cfg)
    # # output = model(imgs)
    # model.cuda()
    # output = model(imgs, audio, gt)
    # pdb.set_trace()
    import numpy as np
    rs = np.random.RandomState(33)
    print(type(rs))
