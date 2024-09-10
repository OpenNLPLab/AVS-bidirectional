import torch


class Correlation:

    @classmethod
    def multilayer_correlation(cls, query_feats, support_feats):
        eps = 1e-5

        corrs = []
        for idx, (query_feat, support_feat) in enumerate(zip(query_feats, support_feats)):
            bsz, ch, hb, wb = support_feat.size()
            support_feat = support_feat.view(bsz, ch, -1)
            # support_feat = support_feat / (support_feat.norm(dim=1, p=2, keepdim=True) + eps)

            bsz, ch, ha, wa = query_feat.size()
            query_feat = query_feat.view(bsz, ch, -1)
            # query_feat = query_feat / (query_feat.norm(dim=1, p=2, keepdim=True) + eps)

            corr = torch.bmm(query_feat.transpose(1, 2), support_feat).view(bsz, ha, wa, hb, wb)
            corr = corr.clamp(min=0)
            corrs.append(corr)

        corr_l4 = corrs[2].unsqueeze(1)
        corr_l3 = corrs[1].unsqueeze(1)
        corr_l2 = corrs[0].unsqueeze(1)

        return [corr_l4, corr_l3, corr_l2]
