import torch
import torch.nn as nn
from mpmath import clsin
from torch.nn import functional as F
from torch.nn.modules.module import T

from layers.Invertible import RevIN
from layers.Autoformer_EncDec import series_decomp, AdaptiveConvDecomp, AdaptiveAttentionDecomp, GatedDecomp
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

class KANLinear_for_Predict(nn.Module):
    def __init__(self, configs):
        super(KANLinear_for_Predict, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channel = configs.channel
        self.d_model = configs.d_model
        self.down_sampling_layers = configs.down_sampling_layers
        self.down_sampling_window = configs.down_sampling_window
        self.num_experts = 4

        self.gates = nn.ModuleList([
            nn.LSTM(self.seq_len // (2 ** i), self.num_experts, batch_first=True)
            for i in range(self.down_sampling_layers + 1)
        ])

        self.predict_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    configs.pred_len,
                )
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        self.experts = nn.ModuleList([
            nn.ModuleList([
                KANLinear(self.configs.seq_len // (2 ** i), self.configs.seq_len // (2 ** i)),
                KANLinear(self.configs.seq_len // (2 ** i), self.configs.seq_len // (2 ** i)),
                KANLinear(self.configs.seq_len // (2 ** i), self.configs.seq_len // (2 ** i)),
                KANLinear(self.configs.seq_len // (2 ** i), self.configs.seq_len // (2 ** i)),
            ])
            for i in range(self.configs.down_sampling_layers + 1)
        ])

        self.projection_layer = nn.Linear(configs.d_model, 1, bias=True)


    def forward(self, B, d_model, enc_out_list, x_list):
        dec_out_list = []
        for i, enc_out in zip(range(len(x_list)), enc_out_list):
            b_dmode_c, T = enc_out.size()
            b_c = b_dmode_c // self.d_model
            expert_outputs = torch.stack([expert(enc_out) for expert in self.experts[i]], dim=-1)  # (B, Lo, E)
            gate_out, _ = self.gates[i](enc_out)
            score = F.softmax(gate_out, dim=-1)
            # score= F.softmax(self.gates[i](enc_out), dim=-1)
            prediction = torch.einsum("BLE,BE->BL", expert_outputs, score).reshape(b_c, d_model, -1).permute(0, 2, 1)

            dec_out = self.predict_layers[i](prediction.permute(0, 2, 1)).permute(0, 2, 1)
            dec_out = self.projection_layer(dec_out)
            dec_out = dec_out.reshape(B, self.configs.channel, self.pred_len).permute(0, 2, 1).contiguous()

            dec_out_list.append(dec_out)

        return dec_out_list

class AdaptiveDecomp(nn.Module):
    def __init__(self, configs):
        super(AdaptiveDecomp, self).__init__()
        self.configs = configs
        self.input_dim = configs.d_model
        self.trend_conv = nn.Conv1d(in_channels=configs.d_model, out_channels=configs.d_model, kernel_size=3, padding=1)
        self.season_conv = nn.Conv1d(in_channels=configs.d_model, out_channels=configs.d_model, kernel_size=3, padding=1)

        # Smooth trend extraction
        self.avg_pool = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)

    def decompose(self, x):
        # Use convolution to extract trend
        trend = self.trend_conv(x.permute(0, 2, 1))
        # Smooth the trend to get a more stable component
        trend = self.avg_pool(trend)

        # Extract seasonal component by subtracting the trend from the original input
        season = x.permute(0, 2, 1) - trend

        # Return to original dimension
        return season.permute(0, 2, 1), trend.permute(0, 2, 1)

    def forward(self, enc_out_list):
        season_list = []
        trend_list = []
        x_kan_list = []

        for x in enc_out_list:
            b, T, c = x.size()
            season, trend = self.decompose(x)

            # b, T, c -> b*c, T
            x_kan_shape = x.permute(0, 2, 1).reshape(-1, T)
            season_kan_shape = season.permute(0, 2, 1).reshape(-1, T)
            trend_kan_shape = trend.permute(0, 2, 1).reshape(-1, T)

            x_kan_list.append(x_kan_shape)
            season_list.append(season_kan_shape)
            trend_list.append(trend_kan_shape)

        return x_kan_list, season_list, trend_list


class AdaptiveKANLinearFusion(nn.Module):
    def __init__(self, configs):
        super(AdaptiveKANLinearFusion, self).__init__()
        self.configs = configs
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)

        # Define output cross layers for each downsampling level
        self.out_cross_layer = torch.nn.ModuleList(
            [
                nn.Sequential(
                    KANLinear(configs.seq_len // (configs.down_sampling_window ** i),
                              configs.seq_len // (configs.down_sampling_window ** i))
                ) for i in range(configs.down_sampling_layers + 1)
            ]
        )

        # Add an attention mechanism to enhance season and trend interactions
        self.interaction_attention = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=configs.seq_len // (configs.down_sampling_window ** i), num_heads=2)
            for i in range(configs.down_sampling_layers + 1)
        ])

        # Add learnable weights for season and trend components
        self.season_weight_layer = nn.ModuleList([
            nn.Linear(configs.seq_len // (configs.down_sampling_window ** i),
                      configs.seq_len // (configs.down_sampling_window ** i))
            for i in range(configs.down_sampling_layers + 1)
        ])

        # Residual connections
        self.residual_layers = torch.nn.ModuleList([
            nn.Linear(configs.seq_len // (configs.down_sampling_window ** i),
                      configs.seq_len // (configs.down_sampling_window ** i))
            for i in range(configs.down_sampling_layers + 1)
        ])

    def forward(self, x_kan_list, season_list, trend_list):
        length_list = []
        for x in x_kan_list:
            _, T = x.size()
            length_list.append(T)

        out_season_list = self.mixing_multi_scale_season(season_list)
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []

        for i, (ori, out_season, out_trend, length) in enumerate(
                zip(x_kan_list, out_season_list, out_trend_list, length_list)):
            # Align dimensions if necessary
            if out_season.size(-1) != out_trend.size(-1):
                out_season = self.align_layer[i](out_season)

            # Concatenate season and trend, and apply attention mechanism
            combined = torch.stack([out_season, out_trend], dim=0)  # Shape: (2, batch_size, seq_len)
            if i < len(self.interaction_attention):
                combined, _ = self.interaction_attention[i](combined, combined, combined)
            out_season, out_trend = combined[0], combined[1]

            # Learnable weighted fusion
            season_weight = torch.sigmoid(self.season_weight_layer[i](out_season))
            trend_weight = 1 - season_weight
            out = season_weight * out_season + trend_weight * out_trend

            # Residual connection
            out = out + self.residual_layers[i](ori)

            # Combine with original input
            out = ori + self.out_cross_layer[i](out)
            out_list.append(out[:, :length])

        return out_list



class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channel = configs.channel
        self.d_model = configs.d_model
        self.down_sampling_layers = configs.down_sampling_layers
        self.down_sampling_window = configs.down_sampling_window
        self.num_experts = 4

        self.embed = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)

        self.normalize_layers = torch.nn.ModuleList(
            [
                RevIN(self.configs.var_num, affine=True)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        self.decompose = AdaptiveDecomp(self.configs)
        self.fusion = AdaptiveKANLinearFusion(self.configs)
        self.down_sample = Down_sample(self.configs)
        self.season_mixing = MultiScaleSeasonMixing(self.configs)
        self.trend_mixing = MultiScaleTrendMixing(self.configs)
        self.predict = KANLinear_for_Predict(self.configs)

    def forward(self, x, x_mark):
        x_enc, x_mark_enc = self.down_sample(x, x_mark)

        #normalize and embedding
        x_list = []
        x_mark_list = []
        for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
            B, T, N = x.size()
            x = self.normalize_layers[i](x, 'norm')
            x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            x_mark = x_mark.repeat(N, 1, 1)
            x_list.append(x)
            x_mark_list.append(x_mark)

        enc_out_list = []
        for i, x, x_mark in zip(range(len(x_list)), x_list, x_mark_list):
            enc_out = self.embed(x, x_mark)  # [B,T,C]
            enc_out_list.append(enc_out)

        x_kan_list, season_list, trend_list = self.decompose(enc_out_list)

        fusion_out = self.fusion(x_kan_list, season_list, trend_list)

        prediction_list = self.predict(B, self.d_model, fusion_out, x_list)

        dec_out = torch.stack(prediction_list, dim=-1).sum(-1)
        dec_out = self.normalize_layers[0](dec_out, 'denorm')

        return dec_out
