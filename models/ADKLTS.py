import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize
from layers.KANLinear import KANLinear
import math

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
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),

                )
                for i in range(configs.down_sampling_layers)
            ]
        )

    def forward(self, season_list):

        # mixing high->low
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

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
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(configs.down_sampling_layers))
            ])

    def forward(self, trend_list):

        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class AdaptiveDecomp(nn.Module):
    def __init__(self, configs):
        super(AdaptiveDecomp, self).__init__()
        self.configs = configs

        # Multi-level moving average for trend extraction (cascading)
        self.kernel_sizes = [5, 15, 25]  # Different window sizes for different trend scales
        self.moving_avgs = nn.ModuleList([
            nn.AvgPool1d(kernel_size=ks, stride=1, padding=0) for ks in self.kernel_sizes
        ])

        # Convolution layer for trend enhancement
        self.conv1 = nn.Conv1d(in_channels=configs.d_model, out_channels=configs.d_model, kernel_size=3, padding=1)

        # Linear layer for trend refinement
        self.linear = nn.Linear(configs.d_model, configs.d_model)

    def forward(self, x):
        # Cascading decomposition for trend extraction
        trend = x
        residual = x

        # Iteratively apply moving averages to extract trends
        for avg_pool, ks in zip(self.moving_avgs, self.kernel_sizes):
            # `ks` is an integer, which directly represents the kernel size used
            front = residual[:, 0:1, :].repeat(1, (ks - 1) // 2, 1)
            end = residual[:, -1:, :].repeat(1, (ks - 1) // 2, 1)
            x_padded = torch.cat([front, residual, end], dim=1)
            trend_component = avg_pool(x_padded.permute(0, 2, 1)).permute(0, 2, 1)

            # Enhance trend using convolution
            trend_component = self.conv1(trend_component.permute(0, 2, 1)).permute(0, 2, 1)

            # Refine trend component using a linear layer
            trend_component = self.linear(trend_component)

            # Update trend and residual
            trend = trend + trend_component
            residual = residual - trend_component

        return residual, trend


class Decompose(nn.Module):
    def __init__(self, configs):
        super(Decompose, self).__init__()
        self.configs = configs
        self.kernel_size = 25
        self.decompsition = series_decomp(self.kernel_size)
        self.autodecompsition = AdaptiveDecomp(configs)

    def forward(self, x_list):
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.autodecompsition(x)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))
        return season_list, trend_list


class LD(nn.Module):
    def __init__(self, kernel_size=25):
        super(LD, self).__init__()
        # Define a shared convolution layers for all channels
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, stride=1, padding=int(kernel_size // 2),
                              padding_mode='replicate', bias=True)
        # Define the parameters for Gaussian initialization
        kernel_size_half = kernel_size // 2
        sigma = 1.0  # 1 for variance
        weights = torch.zeros(1, 1, kernel_size)
        for i in range(kernel_size):
            weights[0, 0, i] = math.exp(-((i - kernel_size_half) / (2 * sigma)) ** 2)

        # Set the weights of the convolution layer
        self.conv.weight.data = F.softmax(weights, dim=-1)
        self.conv.bias.data.fill_(0.0)

    def forward(self, inp):
        # Permute the input tensor to match the expected shape for 1D convolution (B, N, T)
        inp = inp.permute(0, 2, 1)
        # Split the input tensor into separate channels
        input_channels = torch.split(inp, 1, dim=1)

        # Apply convolution to each channel
        conv_outputs = [self.conv(input_channel) for input_channel in input_channels]

        # Concatenate the channel outputs
        out = torch.cat(conv_outputs, dim=1)
        out = out.permute(0, 2, 1)
        return out

class Fusion_KAN(nn.Module):
    def __init__(self, configs):
        super(Fusion_KAN, self).__init__()
        self.configs = configs
        self.multi_scale_season_mixing = MultiScaleSeasonMixing(configs)
        self.multi_scale_trend_mixing = MultiScaleTrendMixing(configs)
        self.num_layers = configs.num_layers

        # Replacing out_cross_layer with KANLinear for enhanced interaction
        # self.out_kan_linear = KANLinear(
        #     in_features=configs.d_model,
        #     out_features=configs.d_model
        # )
        self.out_kan_linear_layers = nn.ModuleList([
            KANLinear(
                in_features=configs.d_model,
                out_features=configs.d_model
            ) for _ in range(self.num_layers)  # num_layers = KANLinear
        ])

    def forward(self, x_list, season_list, trend_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

            # bottom-up season mixing
        out_season_list = self.multi_scale_season_mixing(season_list)
        # top-down trend mixing
        out_trend_list = self.multi_scale_trend_mixing(trend_list)

        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list, length_list):
            # Combine season and trend components
            out = out_season + out_trend

            # Apply KANLinear layer for enhanced interaction
            B, T, C = out.size()
            out_2d = out.reshape(B * T, C)  # Reshape to 2D for KANLinear
            # out_2d = self.out_kan_linear(out_2d)
            for kan_linear_layer in self.out_kan_linear_layers:
                out_2d = kan_linear_layer(out_2d)
            out = out_2d.reshape(B, T, C)  # Reshape back to original dimensions

            # Residual connection with the original input
            out = ori + out
            out_list.append(out[:, :length, :])

        return out_list

class Fusion_Linear(nn.Module):
    def __init__(self, configs):
        super(Fusion_Linear, self).__init__()
        self.configs = configs
        self.multi_scale_season_mixing = MultiScaleSeasonMixing(configs)
        self.multi_scale_trend_mixing = MultiScaleTrendMixing(configs)

        # Replacing KANLinear with a standard Linear layer for interaction
        self.out_linear = nn.Linear(
            in_features=configs.d_model,
            out_features=configs.d_model
        )

    def forward(self, x_list, season_list, trend_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # bottom-up season mixing
        out_season_list = self.multi_scale_season_mixing(season_list)
        # top-down trend mixing
        out_trend_list = self.multi_scale_trend_mixing(trend_list)

        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list, length_list):
            # Combine season and trend components
            out = out_season + out_trend

            # Apply Linear layer for interaction
            B, T, C = out.size()
            out = self.out_linear(out)

            # Residual connection with the original input
            out = ori + out
            out_list.append(out[:, :length, :])

        return out_list

class Fusion_SelfAttention(nn.Module):
    def __init__(self, configs):
        super(Fusion_SelfAttention, self).__init__()
        self.configs = configs
        self.multi_scale_season_mixing = MultiScaleSeasonMixing(configs)
        self.multi_scale_trend_mixing = MultiScaleTrendMixing(configs)

        # Adding self-attention mechanism for interaction
        self.self_attention = nn.MultiheadAttention(
            embed_dim=configs.d_model,
            num_heads=4
        )

    def forward(self, x_list, season_list, trend_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # bottom-up season mixing
        out_season_list = self.multi_scale_season_mixing(season_list)
        # top-down trend mixing
        out_trend_list = self.multi_scale_trend_mixing(trend_list)

        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list, length_list):
            # Combine season and trend components
            out = out_season + out_trend

            # Apply self-attention mechanism for interaction
            B, T, C = out.size()
            out = out.permute(1, 0, 2)  # Reshape to (T, B, C) for MultiheadAttention
            out, _ = self.self_attention(out, out, out)
            out = out.permute(1, 0, 2)  # Reshape back to (B, T, C)

            # Residual connection with the original input
            out = ori + out
            out_list.append(out[:, :length, :])

        return out_list

class Prediction(nn.Module):
    def __init__(self, configs):
        super(Prediction, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channel = configs.channel
        self.d_model = configs.d_model
        self.down_sampling_layers = configs.down_sampling_layers
        self.down_sampling_window = configs.down_sampling_window
        self.num_experts = configs.experts

        self.gates = nn.ModuleList([
            nn.LSTM(self.seq_len // (2 ** i), self.num_experts, batch_first=True)
            for i in range(self.down_sampling_layers + 1)
        ])

        self.experts = nn.ModuleList([
            nn.ModuleList([
                KANLinear(self.configs.seq_len // (2 ** i), self.configs.pred_len),
                KANLinear(self.configs.seq_len // (2 ** i), self.configs.pred_len),
                KANLinear(self.configs.seq_len // (2 ** i), self.configs.pred_len),
                KANLinear(self.configs.seq_len // (2 ** i), self.configs.pred_len),
                # KANLinear(self.configs.seq_len // (2 ** i), self.configs.pred_len),
                # KANLinear(self.configs.seq_len // (2 ** i), self.configs.pred_len),
                # KANLinear(self.configs.seq_len // (2 ** i), self.configs.pred_len),
                # KANLinear(self.configs.seq_len // (2 ** i), self.configs.pred_len),
            ])
            for i in range(self.configs.down_sampling_layers + 1)
        ])

        self.projection_layer = nn.Linear(configs.d_model, 1, bias=True)

    def forward(self, B, enc_out_list, x_list):
        dec_out_list = []
        for i, enc_out in zip(range(len(x_list)), enc_out_list):
            b_c, T, d_model = enc_out.size()
            enc_out = enc_out.permute(0, 2, 1).contiguous().reshape(b_c*d_model, T)
            expert_outputs = torch.stack([expert(enc_out) for expert in self.experts[i]], dim=-1)  # (B, Lo, E)
            gate_out, _ = self.gates[i](enc_out)
            score = F.softmax(gate_out, dim=-1)
            # score= F.softmax(self.gates[i](enc_out), dim=-1)
            prediction = torch.einsum("BLE,BE->BL", expert_outputs, score).reshape(b_c, self.d_model, -1).permute(0, 2, 1)

            dec_out = self.projection_layer(prediction)
            dec_out = dec_out.reshape(B, self.configs.channel, self.pred_len).permute(0, 2, 1).contiguous()

            dec_out_list.append(dec_out)

        return dec_out_list

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.embed = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
        self.decompose = Decompose(configs)
        self.fusion_kan = Fusion_KAN(configs)
        self.fusion_linear = Fusion_Linear(configs)
        self.fusion_attention = Fusion_SelfAttention(configs)
        self.down_sample = Down_sample(configs)
        self.prediction = Prediction(configs)

        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.configs.channel, affine=True, non_norm=True if configs.use_norm == 0 else False)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence

    def forward(self, x_enc, x_mark_enc):
        x_enc, x_mark_enc = self.down_sample(x_enc, x_mark_enc)

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

        season_list, trend_list = self.decompose(enc_out_list)
        dec_out_list = self.fusion_kan(enc_out_list, season_list, trend_list)
        dec_out_list = self.prediction(B, dec_out_list, x_list)
        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out
