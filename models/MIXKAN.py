import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.module import T

from layers.Invertible import RevIN
from layers.kanlayers import KANLayer, TaylorKANLayer, WaveKANLayer
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.KANLinear import KANLinear


class Down_sample(nn.Module):
    def __init__(self, configs):
        super(Down_sample, self).__init__()
        self.configs = configs
        self.down_pool = torch.nn.AvgPool1d(configs.down_sampling_window)

    def forward(self, x_enc, x_mark_enc):
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = self.down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc_mark_ori is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        if x_mark_enc_mark_ori is not None:
            x_mark_enc = x_mark_sampling_list
        else:
            x_mark_enc = x_mark_enc

        return x_enc, x_mark_enc


class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()

        # self.down_sampling_layers = torch.nn.ModuleList(
        #     [
        #         nn.Sequential(
        #             torch.nn.Linear(
        #                 configs.seq_len // (configs.down_sampling_window ** i),
        #                 configs.seq_len // (configs.down_sampling_window ** (i + 1)),
        #             ),
        #             nn.GELU(),
        #             torch.nn.Linear(
        #                 configs.seq_len // (configs.down_sampling_window ** (i + 1)),
        #                 configs.seq_len // (configs.down_sampling_window ** (i + 1)),
        #             ),
        #
        #         )
        #         for i in range(configs.down_sampling_layers)
        #     ]
        # )

        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                   KANLinear(configs.seq_len // (configs.down_sampling_window ** i),
                             configs.seq_len // (configs.down_sampling_window ** (i + 1))),
                   KANLinear(configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                             configs.seq_len // (configs.down_sampling_window ** (i + 1))),
                )
                for i in range(configs.down_sampling_layers)
            ]
        )

    def forward(self, season_list):

        # mixing high->low
        out_high = season_list[0]
        out_low = season_list[1]
        # out_season_list = [out_high.permute(0, 2, 1)]
        out_season_list = [out_high]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high)

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()

        # self.up_sampling_layers = torch.nn.ModuleList(
        #     [
        #         nn.Sequential(
        #             torch.nn.Linear(
        #                 configs.seq_len // (configs.down_sampling_window ** (i + 1)),
        #                 configs.seq_len // (configs.down_sampling_window ** i),
        #             ),
        #             nn.GELU(),
        #             torch.nn.Linear(
        #                 configs.seq_len // (configs.down_sampling_window ** i),
        #                 configs.seq_len // (configs.down_sampling_window ** i),
        #             ),
        #         )
        #         for i in reversed(range(configs.down_sampling_layers))
        #     ])

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    KANLinear(configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                              configs.seq_len // (configs.down_sampling_window ** i)),
                    KANLinear(configs.seq_len // (configs.down_sampling_window ** i),
                              configs.seq_len // (configs.down_sampling_window ** i)),
                )
                for i in reversed(range(configs.down_sampling_layers))
            ])

    def forward(self, trend_list):

        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low)

        out_trend_list.reverse()
        return out_trend_list


class decompose_and_fusion_layer(nn.Module):
    def __init__(self, configs):
        super(decompose_and_fusion_layer, self).__init__()
        self.seq_len = configs.seq_len
        self.pre_len = configs.pred_len
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.down_sampling_window = configs.down_sampling_window
        self.layer_norm = RevIN(self.d_model)
        self.dropout = nn.Dropout(self.dropout)
        self.channel_independence = configs.channel_independence
        self.decompsition = series_decomp(configs.moving_avg)
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)
        # Mxing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)

        # self.out_cross_layer = nn.Sequential(
        #     nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
        #     nn.GELU(),
        #     nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
        # )
        self.out_cross_layer = torch.nn.ModuleList(
            [
                nn.Sequential(
                    KANLinear(configs.seq_len // (configs.down_sampling_window ** i),
                              configs.seq_len // (configs.down_sampling_window ** i))
                ) for i in range(configs.down_sampling_layers + 1)
            ]
        )

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        x_kan_list = []
        for x in x_list:
            b, T, c = x.size()
            season, trend = self.decompsition(x)
            # b, T, c -> b*c, T
            x_kan_shape = x.permute(0, 2, 1).reshape(-1, T)
            season_kan_shape = season.permute(0, 2, 1).reshape(-1, T)
            trend_kan_shape = trend.permute(0, 2, 1).reshape(-1, T)
            # season_list.append(season.permute(0, 2, 1))
            # trend_list.append(trend.permute(0, 2, 1))
            x_kan_list.append(x_kan_shape)
            season_list.append(season_kan_shape)
            trend_list.append(trend_kan_shape)

        # bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list)
        # top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        # for ori, out_season, out_trend, length in zip(x_kan_list, out_season_list, out_trend_list,
        #                                               length_list):
        #     out = out_season + out_trend
        #     out = ori + self.out_cross_layer(out)
        #     out_list.append(out[:, :length, :])

        for i, (ori, out_season, out_trend, length) in enumerate(
                zip(x_kan_list, out_season_list, out_trend_list, length_list)):
            out = out_season + out_trend
            out = ori + self.out_cross_layer[i](out)  #
            out_list.append(out[:, :length])
        return out_list


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.down_sampling_window = configs.down_sampling_window
        self.down_sampling_layers = configs.down_sampling_layers
        self.channel_independence = configs.channel_independence
        self.num_experts = 4
        self.down_sample_layers = Down_sample(configs)
        self.e_layers = configs.e_layers
        self.fusion_layers = nn.ModuleList([decompose_and_fusion_layer(configs)
                                            for _ in range(configs.e_layers)])
        self.preprocess = series_decomp(configs.moving_avg)
        self.var_num = configs.var_num

        self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)

        self.normalize_layers = torch.nn.ModuleList(
            [
                RevIN(self.configs.var_num, affine=True)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        self.predict_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    configs.pred_len,
                )
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        self.gates = nn.ModuleList([
            nn.Linear(self.seq_len // (2 ** i), self.num_experts)
            for i in range(self.down_sampling_layers + 1)
        ])

        self.experts = nn.ModuleList([
            nn.ModuleList([
                KANLinear(self.seq_len // (2 ** i), self.seq_len // (2 ** i)),
                KANLinear(self.seq_len // (2 ** i), self.seq_len // (2 ** i)),
                KANLinear(self.seq_len // (2 ** i), self.seq_len // (2 ** i)),
                KANLinear(self.seq_len // (2 ** i), self.seq_len // (2 ** i)),
            ])
            for i in range(self.down_sampling_layers + 1)
        ])

        self.projection_layer = nn.Linear(configs.d_model, 1, bias=True)

    def future_multi_mixing(self, B, d_model, enc_out_list, x_list):
        dec_out_list = []
        for i, enc_out in zip(range(len(x_list)), enc_out_list):
            b_dmode_c, T = enc_out.size()
            b_c = b_dmode_c // self.d_model
            expert_outputs = torch.stack([expert(enc_out) for expert in self.experts[i]], dim=-1)  # (B, Lo, E)
            score = F.softmax(self.gates[i](enc_out), dim=-1)
            prediction = torch.einsum("BLE,BE->BL", expert_outputs, score).reshape(b_c, d_model, -1).permute(0, 2, 1)

            dec_out = self.predict_layers[i](prediction.permute(0, 2, 1)).permute(0, 2, 1)
            dec_out = self.projection_layer(dec_out)
            dec_out = dec_out.reshape(B, self.configs.channel, self.pred_len).permute(0, 2, 1).contiguous()

            dec_out_list.append(dec_out)

        return dec_out_list

    def forward(self, x_enc, x_mark_enc):

        x_enc, x_mark_enc = self.down_sample_layers(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
            B, T, N = x.size()
            x = self.normalize_layers[i](x, 'norm')
            if self.channel_independence == 1:
                x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_mark = x_mark.repeat(N, 1, 1)
            x_list.append(x)
            x_mark_list.append(x_mark)

        enc_out_list = []
        for i, x, x_mark in zip(range(len(x_list)), x_list, x_mark_list):
            enc_out = self.enc_embedding(x, x_mark)  # [B,T,C]
            enc_out_list.append(enc_out)

        for i in range(self.e_layers):
            enc_out_list_kan = self.fusion_layers[i](enc_out_list)

        dec_out_list = self.future_multi_mixing(B, self.d_model, enc_out_list_kan, x_list)

        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out
