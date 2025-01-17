# coding: utf-8
import numpy as np
import torch
import torch.nn as nn
from attention_modules import BertConfig, BertEncoder, BertPooler
from modules import (
    Conv_1d,
    Conv_2d,
    Conv_H,
    Conv_V,
    HarmonicSTFT,
    MelSpecBatchNorm,
    Res_2d,
    Res_2d_mp,
    ResSE_1d,
)


class FCN(nn.Module):
    """
    Choi et al. 2016
    Automatic tagging using deep convolutional neural networks.
    Fully convolutional network.
    """

    def __init__(
        self,
        sample_rate=16000,
        n_fft=512,
        f_min=0.0,
        f_max=8000.0,
        n_mels=96,
        n_class=50,
        n_stems=1,
    ):
        super(FCN, self).__init__()

        # Spectrogram
        self.spec = MelSpecBatchNorm(
            sample_rate=sample_rate,
            n_fft=n_fft,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            n_stems=n_stems,
        )

        # FCN
        self.layer1 = Conv_2d(n_stems, 64, pooling=(2, 4))
        self.layer2 = Conv_2d(64, 128, pooling=(2, 4))
        self.layer3 = Conv_2d(128, 128, pooling=(2, 4))
        self.layer4 = Conv_2d(128, 128, pooling=(3, 5))
        self.layer5 = Conv_2d(128, 64, pooling=(4, 4))

        # Dense
        self.dense = nn.Linear(64, n_class)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Spectrogram
        x = self.spec(x)

        # FCN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        # Dense
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.dense(x)
        x = nn.Sigmoid()(x)

        return x


class Musicnn(nn.Module):
    """
    Pons et al. 2017
    End-to-end learning for music audio tagging at scale.
    This is the updated implementation of the original paper. Referred to the Musicnn code.
    https://github.com/jordipons/musicnn
    """

    def __init__(
        self,
        sample_rate=16000,
        n_fft=512,
        f_min=0.0,
        f_max=8000.0,
        n_mels=96,
        n_class=50,
        dataset="mtat",
        n_stems=1,
    ):
        super(Musicnn, self).__init__()

        # Spectrogram
        self.spec = MelSpecBatchNorm(
            sample_rate=sample_rate,
            n_fft=n_fft,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            n_stems=n_stems,
        )

        # Pons front-end
        m1 = Conv_V(n_stems, 204, (int(0.7 * 96), 7))
        m2 = Conv_V(1, 204, (int(0.4 * 96), 7))
        m3 = Conv_H(1, 51, 129)
        m4 = Conv_H(1, 51, 65)
        m5 = Conv_H(1, 51, 33)
        self.layers = nn.ModuleList([m1, m2, m3, m4, m5])

        # Pons back-end
        backend_channel = 512 if dataset == "msd" else 64
        self.layer1 = Conv_1d(561, backend_channel, 7, 1, 1)
        self.layer2 = Conv_1d(backend_channel, backend_channel, 7, 1, 1)
        self.layer3 = Conv_1d(backend_channel, backend_channel, 7, 1, 1)

        # Dense
        dense_channel = 500 if dataset == "msd" else 200
        self.dense1 = nn.Linear((561 + (backend_channel * 3)) * 2, dense_channel)
        self.bn = nn.BatchNorm1d(dense_channel)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.dense2 = nn.Linear(dense_channel, n_class)

    def forward(self, x):
        # Spectrogram
        x = self.spec(x)

        # Pons front-end
        out = []
        for layer in self.layers:
            out.append(layer(x))
        out = torch.cat(out, dim=1)

        # Pons back-end
        length = out.size(2)
        res1 = self.layer1(out)
        res2 = self.layer2(res1) + res1
        res3 = self.layer3(res2) + res2
        out = torch.cat([out, res1, res2, res3], 1)

        mp = nn.MaxPool1d(length)(out)
        avgp = nn.AvgPool1d(length)(out)

        out = torch.cat([mp, avgp], dim=1)
        out = out.squeeze(2)

        out = self.relu(self.bn(self.dense1(out)))
        out = self.dropout(out)
        out = self.dense2(out)
        out = nn.Sigmoid()(out)

        return out


class CRNN(nn.Module):
    """
    Choi et al. 2017
    Convolution recurrent neural networks for music classification.
    Feature extraction with CNN + temporal summary with RNN
    """

    def __init__(
        self,
        sample_rate=16000,
        n_fft=512,
        f_min=0.0,
        f_max=8000.0,
        n_mels=96,
        n_class=50,
        n_stems=1,
    ):
        super(CRNN, self).__init__()

        # Spectrogram
        self.spec = MelSpecBatchNorm(
            sample_rate=sample_rate,
            n_fft=n_fft,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            n_stems=n_stems,
        )

        # CNN
        self.layer1 = Conv_2d(n_stems, 64, pooling=(2, 2))
        self.layer2 = Conv_2d(64, 128, pooling=(3, 3))
        self.layer3 = Conv_2d(128, 128, pooling=(4, 4))
        self.layer4 = Conv_2d(128, 128, pooling=(4, 4))

        # RNN
        self.layer5 = nn.GRU(128, 32, 2, batch_first=True)

        # Dense
        self.dropout = nn.Dropout(0.5)
        self.dense = nn.Linear(32, 50)

    def forward(self, x):
        # Spectrogram
        x = self.spec(x)

        # CCN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # RNN
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        x, _ = self.layer5(x)
        x = x[:, -1, :]

        # Dense
        x = self.dropout(x)
        x = self.dense(x)
        x = nn.Sigmoid()(x)

        return x


class SampleCNN(nn.Module):
    """
    Lee et al. 2017
    Sample-level deep convolutional neural networks for music auto-tagging using raw waveforms.
    Sample-level CNN.
    """

    def __init__(
        self,
        n_class=50,
        n_stems=1,
    ):
        super(SampleCNN, self).__init__()

        if n_stems != 1:
            raise NotImplementedError("Only single stem input is supported")

        self.layer1 = Conv_1d(n_stems, 128, shape=3, stride=3, pooling=1)
        self.layer2 = Conv_1d(128, 128, shape=3, stride=1, pooling=3)
        self.layer3 = Conv_1d(128, 128, shape=3, stride=1, pooling=3)
        self.layer4 = Conv_1d(128, 256, shape=3, stride=1, pooling=3)
        self.layer5 = Conv_1d(256, 256, shape=3, stride=1, pooling=3)
        self.layer6 = Conv_1d(256, 256, shape=3, stride=1, pooling=3)
        self.layer7 = Conv_1d(256, 256, shape=3, stride=1, pooling=3)
        self.layer8 = Conv_1d(256, 256, shape=3, stride=1, pooling=3)
        self.layer9 = Conv_1d(256, 256, shape=3, stride=1, pooling=3)
        self.layer10 = Conv_1d(256, 512, shape=3, stride=1, pooling=3)
        self.layer11 = Conv_1d(512, 512, shape=1, stride=1, pooling=1)
        self.dropout = nn.Dropout(0.5)
        self.dense = nn.Linear(512, n_class)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = x.squeeze(-1)
        x = self.dropout(x)
        x = self.dense(x)
        x = nn.Sigmoid()(x)
        return x


class SampleCNNSE(nn.Module):
    """
    Kim et al. 2018
    Sample-level CNN architectures for music auto-tagging using raw waveforms.
    Sample-level CNN + residual connections + squeeze & excitation.
    """

    def __init__(
        self,
        n_class=50,
        n_stems=1,
    ):
        super(SampleCNNSE, self).__init__()

        if n_stems != 1:
            raise NotImplementedError("Only single stem input is supported")

        self.layer1 = ResSE_1d(n_stems, 128, shape=3, stride=3, pooling=1)
        self.layer2 = ResSE_1d(128, 128, shape=3, stride=1, pooling=3)
        self.layer3 = ResSE_1d(128, 128, shape=3, stride=1, pooling=3)
        self.layer4 = ResSE_1d(128, 256, shape=3, stride=1, pooling=3)
        self.layer5 = ResSE_1d(256, 256, shape=3, stride=1, pooling=3)
        self.layer6 = ResSE_1d(256, 256, shape=3, stride=1, pooling=3)
        self.layer7 = ResSE_1d(256, 256, shape=3, stride=1, pooling=3)
        self.layer8 = ResSE_1d(256, 256, shape=3, stride=1, pooling=3)
        self.layer9 = ResSE_1d(256, 256, shape=3, stride=1, pooling=3)
        self.layer10 = ResSE_1d(256, 512, shape=3, stride=1, pooling=3)
        self.layer11 = ResSE_1d(512, 512, shape=1, stride=1, pooling=1)
        self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(512, 512)
        self.bn = nn.BatchNorm1d(512)
        self.dense2 = nn.Linear(512, n_class)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = x.squeeze(-1)
        x = nn.ReLU()(self.bn(self.dense1(x)))
        x = self.dropout(x)
        x = self.dense2(x)
        x = nn.Sigmoid()(x)
        return x


class ShortChunkCNN(nn.Module):
    """
    Short-chunk CNN architecture.
    So-called vgg-ish model with a small receptive field.
    Deeper layers, smaller pooling (2x2).
    """

    def __init__(
        self,
        n_channels=128,
        sample_rate=16000,
        n_fft=512,
        f_min=0.0,
        f_max=8000.0,
        n_mels=128,
        n_class=50,
        n_stems=1,
    ):
        super(ShortChunkCNN, self).__init__()

        # Spectrogram
        self.spec = MelSpecBatchNorm(
            sample_rate=sample_rate,
            n_fft=n_fft,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            n_stems=n_stems,
        )

        # CNN
        self.layer1 = Conv_2d(n_stems, n_channels, pooling=2)
        self.layer2 = Conv_2d(n_channels, n_channels, pooling=2)
        self.layer3 = Conv_2d(n_channels, n_channels * 2, pooling=2)
        self.layer4 = Conv_2d(n_channels * 2, n_channels * 2, pooling=2)
        self.layer5 = Conv_2d(n_channels * 2, n_channels * 2, pooling=2)
        self.layer6 = Conv_2d(n_channels * 2, n_channels * 2, pooling=2)
        self.layer7 = Conv_2d(n_channels * 2, n_channels * 4, pooling=2)

        # Dense
        self.dense1 = nn.Linear(n_channels * 4, n_channels * 4)
        self.bn = nn.BatchNorm1d(n_channels * 4)
        self.dense2 = nn.Linear(n_channels * 4, n_class)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Spectrogram
        x = self.spec(x)

        # CNN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = x.squeeze(2)

        # Global Max Pooling
        if x.size(-1) != 1:
            x = nn.MaxPool1d(x.size(-1))(x)
        x = x.squeeze(2)

        # Dense
        x = self.dense1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = nn.Sigmoid()(x)

        return x


class ShortChunkCNN_Res(nn.Module):
    """
    Short-chunk CNN architecture with residual connections.
    """

    def __init__(
        self,
        n_channels=128,
        sample_rate=16000,
        n_fft=512,
        f_min=0.0,
        f_max=8000.0,
        n_mels=128,
        n_class=50,
        n_stems=1,
    ):
        super(ShortChunkCNN_Res, self).__init__()

        # Spectrogram
        self.spec = MelSpecBatchNorm(
            sample_rate=sample_rate,
            n_fft=n_fft,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            n_stems=n_stems,
        )

        # CNN
        self.layer1 = Res_2d(n_stems, n_channels, stride=2)
        self.layer2 = Res_2d(n_channels, n_channels, stride=2)
        self.layer3 = Res_2d(n_channels, n_channels * 2, stride=2)
        self.layer4 = Res_2d(n_channels * 2, n_channels * 2, stride=2)
        self.layer5 = Res_2d(n_channels * 2, n_channels * 2, stride=2)
        self.layer6 = Res_2d(n_channels * 2, n_channels * 2, stride=2)
        self.layer7 = Res_2d(n_channels * 2, n_channels * 4, stride=2)

        # Dense
        self.dense1 = nn.Linear(n_channels * 4, n_channels * 4)
        self.bn = nn.BatchNorm1d(n_channels * 4)
        self.dense2 = nn.Linear(n_channels * 4, n_class)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Spectrogram
        x = self.spec(x)

        # CNN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = x.squeeze(2)

        # Global Max Pooling
        if x.size(-1) != 1:
            x = nn.MaxPool1d(x.size(-1))(x)
        x = x.squeeze(2)

        # Dense
        x = self.dense1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = nn.Sigmoid()(x)

        return x


class CNNSA(nn.Module):
    """
    Won et al. 2019
    Toward interpretable music tagging with self-attention.
    Feature extraction with CNN + temporal summary with Transformer encoder.
    """

    def __init__(
        self,
        n_channels=128,
        sample_rate=16000,
        n_fft=512,
        f_min=0.0,
        f_max=8000.0,
        n_mels=128,
        n_class=50,
        n_stems=1,
    ):
        super(CNNSA, self).__init__()

        # Spectrogram
        self.spec = MelSpecBatchNorm(
            sample_rate=sample_rate,
            n_fft=n_fft,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            n_stems=n_stems,
        )

        # CNN
        self.layer1 = Res_2d(n_stems, n_channels, stride=2)
        self.layer2 = Res_2d(n_channels, n_channels, stride=2)
        self.layer3 = Res_2d(n_channels, n_channels * 2, stride=2)
        self.layer4 = Res_2d(n_channels * 2, n_channels * 2, stride=(2, 1))
        self.layer5 = Res_2d(n_channels * 2, n_channels * 2, stride=(2, 1))
        self.layer6 = Res_2d(n_channels * 2, n_channels * 2, stride=(2, 1))
        self.layer7 = Res_2d(n_channels * 2, n_channels * 2, stride=(2, 1))

        # Transformer encoder
        bert_config = BertConfig(
            vocab_size=256,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=8,
            intermediate_size=1024,
            hidden_act="gelu",
            hidden_dropout_prob=0.4,
            max_position_embeddings=700,
            attention_probs_dropout_prob=0.5,
        )
        self.encoder = BertEncoder(bert_config)
        self.pooler = BertPooler(bert_config)
        self.vec_cls = self.get_cls(256)

        # Dense
        self.dropout = nn.Dropout(0.5)
        self.dense = nn.Linear(256, n_class)

    def get_cls(self, channel):
        np.random.seed(0)
        single_cls = torch.Tensor(np.random.random((1, channel)))
        vec_cls = torch.cat([single_cls for _ in range(64)], dim=0)
        vec_cls = vec_cls.unsqueeze(1)
        return vec_cls

    def append_cls(self, x):
        batch, _, _ = x.size()
        part_vec_cls = self.vec_cls[:batch].clone()
        part_vec_cls = part_vec_cls.to(x.device)
        return torch.cat([part_vec_cls, x], dim=1)

    def forward(self, x):
        # Spectrogram
        x = self.spec(x)

        # CNN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = x.squeeze(2)

        # Get [CLS] token
        x = x.permute(0, 2, 1)
        x = self.append_cls(x)

        # Transformer encoder
        x = self.encoder(x)
        x = x[-1]
        x = self.pooler(x)

        # Dense
        x = self.dropout(x)
        x = self.dense(x)
        x = nn.Sigmoid()(x)

        return x


class HarmonicCNN(nn.Module):
    """
    Won et al. 2020
    Data-driven harmonic filters for audio representation learning.
    Trainable harmonic band-pass filters, short-chunk CNN.
    """

    def __init__(
        self,
        n_channels=128,
        sample_rate=16000,
        n_fft=512,
        f_min=0.0,
        f_max=8000.0,
        n_mels=128,
        n_class=50,
        n_stems=1,
        n_harmonic=6,
        semitone_scale=2,
        learn_bw="only_Q",
    ):
        super(HarmonicCNN, self).__init__()

        if n_stems != 1:
            raise NotImplementedError("Only single stem input is supported")

        # Harmonic STFT
        self.hstft = HarmonicSTFT(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_harmonic=n_harmonic,
            semitone_scale=semitone_scale,
            learn_bw=learn_bw,
        )
        self.hstft_bn = nn.BatchNorm2d(n_harmonic)

        # CNN
        self.layer1 = Conv_2d(n_harmonic * n_stems, n_channels, pooling=2)
        self.layer2 = Res_2d_mp(n_channels, n_channels, pooling=2)
        self.layer3 = Res_2d_mp(n_channels, n_channels, pooling=2)
        self.layer4 = Res_2d_mp(n_channels, n_channels, pooling=2)
        self.layer5 = Conv_2d(n_channels, n_channels * 2, pooling=2)
        self.layer6 = Res_2d_mp(n_channels * 2, n_channels * 2, pooling=(2, 3))
        self.layer7 = Res_2d_mp(n_channels * 2, n_channels * 2, pooling=(2, 3))

        # Dense
        self.dense1 = nn.Linear(n_channels * 2, n_channels * 2)
        self.bn = nn.BatchNorm1d(n_channels * 2)
        self.dense2 = nn.Linear(n_channels * 2, n_class)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Spectrogram
        x = self.hstft_bn(self.hstft(x))

        # CNN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = x.squeeze(2)

        # Global Max Pooling
        if x.size(-1) != 1:
            x = nn.MaxPool1d(x.size(-1))(x)
        x = x.squeeze(2)

        # Dense
        x = self.dense1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = nn.Sigmoid()(x)

        return x


class ShortChunkCNNMultiStem(nn.Module):
    """
    Short-chunk CNN architecture, but
    don't combine stem features until last layer
    """

    def __init__(
        self,
        n_channels=64,
        sample_rate=16000,
        n_fft=512,
        f_min=0.0,
        f_max=8000.0,
        n_mels=128,
        n_class=50,
        n_stems=1,
    ):
        super(ShortChunkCNNMultiStem, self).__init__()

        # Spectrogram
        self.spec = MelSpecBatchNorm(
            sample_rate=sample_rate,
            n_fft=n_fft,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            n_stems=n_stems,
        )

        self.n_stems = n_stems
        # CNN
        self.layer1 = nn.ModuleList()
        self.layer2 = nn.ModuleList()
        self.layer3 = nn.ModuleList()
        self.layer4 = nn.ModuleList()
        self.layer5 = nn.ModuleList()
        self.layer6 = nn.ModuleList()
        self.layer7 = nn.ModuleList()
        for stem in range(n_stems):
            self.layer1.append(Conv_2d(1, n_channels, pooling=2))
            self.layer2.append(Conv_2d(n_channels, n_channels, pooling=2))
            self.layer3.append(Conv_2d(n_channels, n_channels * 2, pooling=2))
            self.layer4.append(Conv_2d(n_channels * 2, n_channels * 2, pooling=2))
            self.layer5.append(Conv_2d(n_channels * 2, n_channels * 2, pooling=2))
            self.layer6.append(Conv_2d(n_channels * 2, n_channels * 2, pooling=2))
            self.layer7.append(Conv_2d(n_channels * 2, n_channels * 4, pooling=2))

        # Dense
        self.dense1 = nn.Linear(n_channels * 4 * n_stems, n_channels * 4)
        self.bn = nn.BatchNorm1d(n_channels * 4)
        self.dense2 = nn.Linear(n_channels * 4, n_class)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Spectrogram
        x = self.spec(x)
        # x is (N, stems, H, W)

        stem_features = []
        for stem in range(self.n_stems):
            # CNN
            x2 = x[:, stem, ...].unsqueeze(1)
            x2 = self.layer1[stem](x2)
            x2 = self.layer2[stem](x2)
            x2 = self.layer3[stem](x2)
            x2 = self.layer4[stem](x2)
            x2 = self.layer5[stem](x2)
            x2 = self.layer6[stem](x2)
            x2 = self.layer7[stem](x2)
            # x2 is (N, C, H, W)
            x2 = x2.squeeze(2)
            # x2 is (N, C, W)

            # Global Max Pooling
            if x2.size(-1) != 1:
                x2 = nn.MaxPool1d(x2.size(-1))(x2)
            x2 = x2.squeeze(2)
            # x2 is (N, C)
            stem_features.append(x2)

        x = torch.cat(stem_features, 1)
        # x is (N, n_stems * C)

        # Dense
        x = self.dense1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = nn.Sigmoid()(x)

        return x


class ShortChunkCNNMultiStem_Res(nn.Module):
    """
    Short-chunk CNN architecture with residual connections.
    """

    def __init__(
        self,
        n_channels=128,
        sample_rate=16000,
        n_fft=512,
        f_min=0.0,
        f_max=8000.0,
        n_mels=128,
        n_class=50,
        n_stems=1,
    ):
        super(ShortChunkCNNMultiStem_Res, self).__init__()

        # Spectrogram
        self.spec = MelSpecBatchNorm(
            sample_rate=sample_rate,
            n_fft=n_fft,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            n_stems=n_stems,
        )

        self.n_stems = n_stems
        # CNN
        self.layer1 = nn.ModuleList()
        self.layer2 = nn.ModuleList()
        self.layer3 = nn.ModuleList()
        self.layer4 = nn.ModuleList()
        self.layer5 = nn.ModuleList()
        self.layer6 = nn.ModuleList()
        self.layer7 = nn.ModuleList()
        for stem in range(n_stems):
            self.layer1.append(Res_2d(1, n_channels, stride=2))
            self.layer2.append(Res_2d(n_channels, n_channels, stride=2))
            self.layer3.append(Res_2d(n_channels, n_channels * 2, stride=2))
            self.layer4.append(Res_2d(n_channels * 2, n_channels * 2, stride=2))
            self.layer5.append(Res_2d(n_channels * 2, n_channels * 2, stride=2))
            self.layer6.append(Res_2d(n_channels * 2, n_channels * 2, stride=2))
            self.layer7.append(Res_2d(n_channels * 2, n_channels * 4, stride=2))

        # Dense
        self.dense1 = nn.Linear(n_channels * 4 * n_stems, n_channels * 4)
        self.bn = nn.BatchNorm1d(n_channels * 4)
        self.dense2 = nn.Linear(n_channels * 4, n_class)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Spectrogram
        x = self.spec(x)
        # x is (N, stems, H, W)

        stem_features = []
        for stem in range(self.n_stems):
            # CNN
            x2 = x[:, stem, ...].unsqueeze(1)
            x2 = self.layer1[stem](x2)
            x2 = self.layer2[stem](x2)
            x2 = self.layer3[stem](x2)
            x2 = self.layer4[stem](x2)
            x2 = self.layer5[stem](x2)
            x2 = self.layer6[stem](x2)
            x2 = self.layer7[stem](x2)
            # x2 is (N, C, H, W)
            x2 = x2.squeeze(2)
            # x2 is (N, C, W)

            # Global Max Pooling
            if x2.size(-1) != 1:
                x2 = nn.MaxPool1d(x2.size(-1))(x2)
            x2 = x2.squeeze(2)
            # x2 is (N, C)
            stem_features.append(x2)

        x = torch.cat(stem_features, 1)
        # x is (N, n_stems * C)

        # Dense
        x = self.dense1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = nn.Sigmoid()(x)

        return x


def get_model(name: str, dataset: str, n_stems: int) -> tuple[nn.Module, int]:
    n_class = 50 if dataset != "gtzan" else 10
    match name:
        case "fcn":
            model = FCN(n_class=n_class, n_stems=n_stems)
            input_length = 29 * 16000
        case "musicnn":
            model = Musicnn(dataset=dataset, n_class=n_class, n_stems=n_stems)
            input_length = 3 * 16000
        case "crnn":
            model = CRNN(n_class=n_class, n_stems=n_stems)
            input_length = 29 * 16000
        case "sample":
            model = SampleCNN(n_class=n_class, n_stems=n_stems)
            input_length = 59049
        case "se":
            model = SampleCNNSE(n_class=n_class, n_stems=n_stems)
            input_length = 59049
        case "short":
            model = ShortChunkCNN(n_class=n_class, n_stems=n_stems)
            input_length = 59049
        case "short_multi_64":
            model = ShortChunkCNNMultiStem(
                n_class=n_class, n_stems=n_stems, n_channels=64
            )
            input_length = 59049
        case "short_multi_32":
            model = ShortChunkCNNMultiStem(
                n_class=n_class, n_stems=n_stems, n_channels=32
            )
            input_length = 59049
        case "short_res":
            model = ShortChunkCNN_Res(n_class=n_class, n_stems=n_stems)
            input_length = 59049
        case "short_res_multi_64":
            model = ShortChunkCNNMultiStem_Res(
                n_class=n_class, n_stems=n_stems, n_channels=64
            )
            input_length = 59049
        case "short_res_multi_32":
            model = ShortChunkCNNMultiStem_Res(
                n_class=n_class, n_stems=n_stems, n_channels=32
            )
            input_length = 59049
        case "attention":
            model = CNNSA(n_class=n_class, n_stems=n_stems)
            input_length = 15 * 16000
        case "hcnn":
            model = HarmonicCNN(n_class=n_class, n_stems=n_stems)
            input_length = 80000

    return model, input_length


MODEL_NAMES = [
    "fcn",
    "musicnn",
    "crnn",
    "sample",
    "se",
    "short",
    "short_multi_64",
    "short_multi_32",
    "short_res",
    "short_res_multi_64",
    "short_res_multi_32",
    "attention",
    "hcnn",
]
