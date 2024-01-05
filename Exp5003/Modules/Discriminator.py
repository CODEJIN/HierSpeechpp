import torch, torchaudio
from torch.nn import Conv1d, Conv2d
from typing import List, Tuple, Union

class Discriminator(torch.nn.Module):
    def __init__(
        self,
        period_list: List[int]= [2, 3, 5, 7, 11],
        period_channels_list: List[int]= [32, 128, 512, 1024, 1024],
        period_kernel_size: int= 5,
        period_stride: int= 3,
        stft_n_fft_list: List[int] = [2048, 1024, 512, 256, 128],
        stft_channels_list: List[int]= [32, 32, 32, 32, 32],
        stft_kernel_size_list: List[Union[int, Tuple[int, int]]]= [(3, 9), (3, 9), (3, 9), (3, 9), (3, 3)],
        stft_stride_list: List[Union[int, Tuple[int, int]]]= [(1, 1), (1, 2), (1, 2), (1, 2), (1, 1)],
        stft_dilation_list: List[Union[int, Tuple[int, int]]]= [(1, 1), (1, 1), (2, 1), (4, 1), (1, 1)],
        scale_pool_kernel_size_list: List[int]= [1, 4, 8, 16, 32],
        scale_channels_list: List[int]= [16, 64, 256, 1024, 1024, 1024],
        scale_kernel_size_list: List[int]= [15, 41, 41, 41, 41, 5],
        scale_stride_list: List[int]= [1, 4, 4, 4, 4, 1],
        scale_groups_list: List[int]= [1, 4, 16, 64, 256, 1],        
        use_stft_discriminator: bool= True,
        leaky_relu_negative_slope: float= 0.1
        ):
        super().__init__()
        self.use_stft_discriminator = use_stft_discriminator

        self.discriminators = torch.nn.ModuleList()
        
        self.discriminators.extend([
            Period_Discriminator(
                period= period,
                channels_list= period_channels_list,
                kernel_size= period_kernel_size,
                stride= period_stride,
                leaky_relu_negative_slope= leaky_relu_negative_slope
                )
            for period in period_list
            ])

        if use_stft_discriminator:
            self.discriminators.extend([
                STFT_Discriminator(
                    n_fft= n_fft,
                    channels_list= stft_channels_list,
                    kernel_size_list= stft_kernel_size_list,
                    stride_list= stft_stride_list,
                    dilation_list= stft_dilation_list,
                    leaky_relu_negative_slope= leaky_relu_negative_slope
                    )
                for n_fft in stft_n_fft_list
                ])
        else:
            self.discriminators.extend([
                Scale_Discriminator(
                    pool_kernel_size= pool_kernel_size,
                    channels_list = scale_channels_list,
                    kernel_size_list= scale_kernel_size_list,
                    stride_list= scale_stride_list,
                    groups_list= scale_groups_list,
                    leaky_relu_negative_slope= leaky_relu_negative_slope
                    )
                for pool_kernel_size in scale_pool_kernel_size_list
                ])
            
        self.r1_regulator = R1_Regulator()

    def forward(
        self,
        audios: torch.FloatTensor,
        prediction_audios: torch.FloatTensor
        ):
        discriminations_list_for_real, feature_maps_list_for_real = [], []        
        for discriminator in self.discriminators:
            discriminations_for_real, feature_maps_for_real = discriminator(audios)
            discriminations_list_for_real.append(discriminations_for_real)
            feature_maps_list_for_real.extend(feature_maps_for_real)

        discriminations_list_for_fake, feature_maps_list_for_fake = [], []        
        for discriminator in self.discriminators:
            discriminations_for_fake, feature_maps_for_fake = discriminator(prediction_audios)
            discriminations_list_for_fake.append(discriminations_for_fake)
            feature_maps_list_for_fake.extend(feature_maps_for_fake)

        with torch.cuda.amp.autocast(enabled= False):
            discrimination_loss = Discriminator_Loss(discriminations_list_for_real, discriminations_list_for_fake)
            r1_loss = self.r1_regulator(discriminations_list_for_real, audios)
            generator_loss = Generator_Loss(discriminations_list_for_fake)
            feature_map_loss = Feature_Map_Loss(feature_maps_list_for_real, feature_maps_list_for_fake)

        return discrimination_loss, r1_loss, generator_loss, feature_map_loss


class Period_Discriminator(torch.nn.Module):
    def __init__(
        self,
        period,
        channels_list: List[int]= [32, 128, 512, 1024, 1024],
        kernel_size: int= 5,
        stride: int= 3,
        leaky_relu_negative_slope: float= 0.1
        ):
        super().__init__()
        self.period = period

        previous_channels = 1
        self.blocks = torch.nn.ModuleList()
        for channels in channels_list:
            block = torch.nn.Sequential(
                torch.nn.utils.weight_norm(Conv2d(
                    in_channels= previous_channels,
                    out_channels= channels,
                    kernel_size= (kernel_size, 1),
                    stride= (stride, 1),
                    padding= ((kernel_size - 1) // 2, 0)
                    )),
                torch.nn.LeakyReLU(negative_slope= leaky_relu_negative_slope)
                )
            self.blocks.append(block)
            previous_channels = channels
        
        # Postnet
        self.blocks.append(torch.nn.utils.weight_norm(Conv2d(
            in_channels= previous_channels,
            out_channels= 1,
            kernel_size= (3, 1),
            padding= (1, 0)
            )))

    def forward(self, audios: torch.Tensor):
        x = audios.unsqueeze(1)

        # dividable by period
        if x.size(2) % self.period != 0: 
            n_pad = self.period - (x.size(2) % self.period)
            x = torch.nn.functional.pad(x, (0, n_pad), "reflect")
        x = x.view(x.size(0), x.size(1), x.size(2) // self.period, self.period) # [Batch, 1, Audio_d // Period, Period]

        feature_maps = []
        for block in self.blocks:
            x = block(x)
            feature_maps.append(x)

        x = x.flatten(start_dim= 1)
        
        return x, feature_maps

class Scale_Discriminator(torch.nn.Module):
    def __init__(
        self,
        pool_kernel_size: int,
        channels_list: List[int]= [16, 64, 256, 1024, 1024, 1024],
        kernel_size_list: List[int]= [15, 41, 41, 41, 41, 5],
        stride_list: List[int]= [1, 4, 4, 4, 4, 1],
        groups_list: List[int]= [1, 4, 16, 64, 256, 1],
        leaky_relu_negative_slope: float= 0.1
        ):
        super().__init__()
        self.pool = torch.nn.AvgPool1d(
            kernel_size= pool_kernel_size,
            stride= max(pool_kernel_size // 2, 1),
            padding= pool_kernel_size // 2
            )
        previous_channels = 1
        self.blocks = torch.nn.ModuleList()
        for channels, kernel_size, stride, groups in zip(
            channels_list,
            kernel_size_list,
            stride_list,
            groups_list,
            ):
            block = torch.nn.Sequential(
                torch.nn.utils.weight_norm(Conv1d(
                    in_channels= previous_channels,
                    out_channels= channels,
                    kernel_size= kernel_size,
                    stride= stride,
                    groups= groups,
                    padding= (kernel_size - 1) // 2
                    )),
                torch.nn.LeakyReLU(negative_slope= leaky_relu_negative_slope)
                )
            self.blocks.append(block)
            previous_channels = channels
        
        # Postnet
        self.blocks.append(torch.nn.utils.weight_norm(Conv1d(
            in_channels= previous_channels,
            out_channels= 1,
            kernel_size= 3,
            padding= 1
            )))

    def forward(self, audios: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = audios.unsqueeze(1) # [Batch, 1, Audio_t]
        x = self.pool(x)

        feature_maps = []
        for block in self.blocks:
            x = block(x)
            feature_maps.append(x)

        x = x.flatten(start_dim= 1)
        
        return x, feature_maps

class STFT_Discriminator(torch.nn.Module):
    def __init__(
        self,
        n_fft: int,
        channels_list: List[int]= [32, 32, 32, 32, 32],
        kernel_size_list: List[Union[int, Tuple[int, int]]]= [(3, 9), (3, 9), (3, 9), (3, 9), (3, 3)],
        stride_list: List[Union[int, Tuple[int, int]]]= [(1, 1), (1, 2), (1, 2), (1, 2), (1, 1)],
        dilation_list: List[Union[int, Tuple[int, int]]]= [(1, 1), (1, 1), (2, 1), (4, 1), (1, 1)],
        leaky_relu_negative_slope: float= 0.1
        ):
        super().__init__()
        self.prenet = torchaudio.transforms.Spectrogram(
            n_fft= n_fft,
            hop_length= n_fft // 4,
            win_length= n_fft,
            window_fn=torch.hann_window,
            normalized= True,
            center= False,
            pad_mode= None,
            power= None,
            return_complex= True
            )

        previous_channels = 2
        self.blocks = torch.nn.ModuleList()
        for channels, kernel_size, stride, dilation  in zip(channels_list, kernel_size_list, stride_list, dilation_list):
            block = torch.nn.Sequential(
                torch.nn.utils.weight_norm(Conv2d(
                    in_channels= previous_channels,
                    out_channels= channels,
                    kernel_size= kernel_size, # (kernel_size, 1),
                    stride= stride, # (stride, 1),
                    dilation= dilation,
                    padding= (
                        (kernel_size[0] * dilation[0] - dilation[0]) // 2,
                        (kernel_size[1] * dilation[1] - dilation[1]) // 2
                        )
                    )),
                torch.nn.LeakyReLU(negative_slope= leaky_relu_negative_slope)
                )
            self.blocks.append(block)
            previous_channels = channels
        
        # Postnet
        self.blocks.append(torch.nn.utils.weight_norm(Conv2d(
            in_channels= previous_channels,
            out_channels= 1,
            kernel_size= 3,
            padding= 1
            )))

    def forward(self, audios: torch.Tensor):
        x = self.prenet(audios).permute(0, 2, 1)    # [Batch, Feature_t, Feature_d]
        x = torch.stack([x.real, x.imag], dim= 1)   # [Batch, 2, Feature_t, Feature_d]

        feature_maps = []
        for block in self.blocks:
            x = block(x)
            feature_maps.append(x)

        x = x.flatten(start_dim= 1)
        
        return x, feature_maps


def Feature_Map_Loss(feature_maps_list_for_real, feature_maps_list_for_fake):
    return torch.stack([
        torch.mean(torch.abs(feature_maps_for_real.detach() - feature_maps_for_fake))
        for feature_maps_for_real, feature_maps_for_fake in zip(
            feature_maps_list_for_real,
            feature_maps_list_for_fake
            )
        ]).sum()

def Discriminator_Loss(discriminations_list_for_real, discriminations_list_for_fake):
    return torch.stack([
        (1 - discriminations_for_real).pow(2.0).mean() + discriminations_for_fake.pow(2.0).mean()
        for discriminations_for_real, discriminations_for_fake in zip(
            discriminations_list_for_real,
            discriminations_list_for_fake
            )
        ]).sum()

def Generator_Loss(discriminations_list_for_fake):
    return torch.stack([
        (1 - discriminations_for_fake).pow(2.0).mean()
        for discriminations_for_fake in discriminations_list_for_fake
        ]).sum()

class R1_Regulator(torch.nn.Module):
    def forward(
        self,
        discriminations_list: List[torch.Tensor],
        audios: torch.Tensor
        ):
        x = torch.autograd.grad(
            outputs= [
                discriminations.sum()
                for discriminations in discriminations_list
                ],
            inputs= audios,
            create_graph= True,
            retain_graph= True,
            only_inputs= True
            )[0].pow(2)
        x = (x.view(audios.size(0), -1).norm(2, dim=1) ** 2).mean()

        return x