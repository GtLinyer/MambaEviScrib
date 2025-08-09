import math

import torch
from torch import nn

from networks.mamba import PatchExpand, VSSLayer_up
from networks.unet import Encoder, Decoder


class MambaDecoder(nn.Module):
    def __init__(self, num_classes, depths=[2, 2, 9, 2], dims=[32, 64, 128, 256], d_state=16, drop_rate=0.,
                 attn_drop_rate=0., norm_layer=nn.LayerNorm, use_checkpoint = False):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depths))]

        self.num_layers = len(depths)
        self.embed_dim = dims[0]
        self.num_features = dims[-1]

        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()

        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2 * int(dims[0] * 2 ** (self.num_layers - 1 - i_layer)),
                                      int(dims[0] * 2 ** (
                                              self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = PatchExpand(dim=int(self.embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2,
                                       norm_layer=norm_layer)
            else:
                layer_up = VSSLayer_up(
                    dim=int(dims[0] * 2 ** (self.num_layers - 1 - i_layer)),
                    depth=depths[(self.num_layers - 1 - i_layer)],
                    d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,  # 20240109
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                        depths[:(self.num_layers - 1 - i_layer) + 1])],
                    norm_layer=norm_layer,
                    upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                    use_checkpoint=use_checkpoint,
                )
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)
        self.up = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=self.embed_dim, out_channels=num_classes, kernel_size=(1, 1), bias=False)
        )

    def forward_up(self, x, x_downsample):
        x = x.permute(0, 3, 2, 1)
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x_d = x_downsample[4 - inx].permute(0, 3, 2, 1)
                # print(x.shape, x_d.shape)
                x = torch.cat([x, x_d], -1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)

        x = self.norm_up(x)  # B H W C

        return x

    def forward(self, x, x_downsample):
        output = self.forward_up(x, x_downsample)
        output = output.permute(0, 3, 2, 1)
        output = self.up(output)
        return output


class MyModel(nn.Module):
    def __init__(self, in_chns, class_num):
        super(MyModel, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder_cnn = Decoder(params)
        self.encoder_mamba = MambaDecoder(class_num)

    def forward(self, x):
        feature = self.encoder(x)  # (b, 16, 256) (b, 32, 128) (b, 64, 64) (b, 128, 32) (b, 256, 16)
        output_cnn = self.decoder_cnn(feature)
        output_mamba = self.encoder_mamba(feature[-1], feature[:-1])
        return output_cnn, output_mamba


if __name__ == '__main__':
    model = MyModel(1, 2).cuda()
    in_data = torch.randn(8, 1, 256, 256).cuda()
    out1, out2 = model(in_data)
    print(out1.shape, out2.shape)
