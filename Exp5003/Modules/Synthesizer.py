from argparse import Namespace
import torch
import math
from functools import partial
from typing import Optional, List, Dict, Tuple, Union

from .Wav2Vec2 import Wav2vec2
from .Style_Encoder import Style_Encoder
from .Resample import Aliasing_Free_Activation
from .Common import FFT_Block, WaveNet as Source_Filter_WaveNet
from .Flow import FlowBlock_Transformer, WaveNet, Flow_KL_Loss
from .Layer import Conv_Init, Lambda
from meldataset import spectrogram, spectrogram_to_mel, mel_spectrogram
from yin import estimate as yin

class Synthesizer(torch.nn.Module):
    def __init__(self, hyper_parameters: Namespace):
        super().__init__()
        self.hp = hyper_parameters

        self.wav2vec2 = Wav2vec2()

        self.style_encoder = Style_Encoder(
            in_channels= self.hp.Sound.N_Mel,
            out_channels= self.hp.Style.Size,
            temporal_kernel_size= self.hp.Style.Temporal_kernel_Size,
            num_heads= self.hp.Style.Head,
            dropout_rate= self.hp.Style.Dropout_Rate,
            )

        self.speaker_agnostic_encoder = Source_Filter_Encoder(self.hp)
        self.speaker_related_encoder = Source_Filter_Encoder(self.hp)        
        self.encoding_flow = FlowBlock_Transformer(
            channels= self.hp.Encoder.Size,
            calc_channels= self.hp.Encoder.Size,
            flow_stack= self.hp.Encoding_Flow.Stack,            
            condition_channels= self.hp.Style.Size,
            flow_ditblock_stack= self.hp.Encoding_Flow.DiT_Stack,
            flow_ditblock_num_heads= self.hp.Encoding_Flow.Head,
            flow_ditblock_ffn_kernel_size= self.hp.Encoding_Flow.Kernel_Size,
            flow_ditblock_dropout_rate= self.hp.Encoding_Flow.Dropout_Rate,
            )
        
        
        self.decoder = Decoder(self.hp)        
        self.acoustic_encoder = Acoustic_Encoder(self.hp)
        self.acoustic_flow = FlowBlock_Transformer(
            channels= self.hp.Encoder.Size,
            calc_channels= self.hp.Encoder.Size,
            flow_stack= self.hp.Acoustic_Flow.Stack,
            condition_channels= self.hp.Style.Size,
            flow_ditblock_stack= self.hp.Acoustic_Flow.DiT_Stack,
            flow_ditblock_num_heads= self.hp.Acoustic_Flow.Head,
            flow_ditblock_ffn_kernel_size= self.hp.Acoustic_Flow.Kernel_Size,
            flow_ditblock_dropout_rate= self.hp.Acoustic_Flow.Dropout_Rate,
            )
        
        self.semantic_f0_predictor = Semantic_F0_Predictor(self.hp)

        self.prosody_encoder = Prosody_Encoder(self.hp)

        self.segment = Segment()

        self.linear_spectrogram_func = partial(
            spectrogram,
            n_fft= self.hp.Sound.N_FFT,
            hop_size= self.hp.Sound.Hop_Size,
            win_size= self.hp.Sound.Window_Size,
            use_normalize= False
            )
        self.linear_to_mel_spectrogram_func = partial(
            spectrogram_to_mel,
            n_fft= self.hp.Sound.N_FFT,
            num_mels= self.hp.Sound.N_Mel,
            sampling_rate= self.hp.Sound.Sample_Rate,
            win_size= self.hp.Sound.Window_Size,
            fmin= 0,
            fmax= None,
            use_denorm= False
            )
        self.mel_spectrogram_func = partial(
            mel_spectrogram,
            n_fft= self.hp.Sound.N_FFT,
            num_mels= self.hp.Sound.N_Mel,
            sampling_rate= self.hp.Sound.Sample_Rate,
            hop_size= self.hp.Sound.Hop_Size,
            win_size= self.hp.Sound.Window_Size,
            fmin= 0,
            fmax= None
            )
        self.prosody_func = partial(
            mel_spectrogram,
            n_fft= self.hp.Sound.N_FFT,
            num_mels= self.hp.Sound.N_Mel // 4,
            sampling_rate= self.hp.Sound.Sample_Rate,
            hop_size= self.hp.Sound.Hop_Size,
            win_size= self.hp.Sound.Window_Size,
            fmin= 0,
            fmax= None
            )
        self.f0_func = partial(
            yin,
            sample_rate= self.hp.Sound.Sample_Rate,
            pitch_min= self.hp.Sound.F0_Min,
            pitch_max= self.hp.Sound.F0_Max,
            hop_size= self.hp.Sound.F0_Hop_Size,
            window_size= self.hp.Sound.F0_Hop_Size * 4,
            threshold= 0.1
            )

    def forward(
        self,
        audios: torch.FloatTensor,
        audio_lengths: torch.IntTensor,
        ):
        audios_perturbed = Perturbing(
            audios= audios,
            sample_rate= self.hp.Sound.Sample_Rate
            )
        contents = self.wav2vec2(audios)
        contents_perturbed = self.wav2vec2(audios_perturbed)
        linear_spectrograms = self.linear_spectrogram_func(audios)
        mel_spectrograms = styles = self.linear_to_mel_spectrogram_func(linear_spectrograms)
        prosodies = self.prosody_func(audios)   # [Batch, Prosody_d, Content_t]
        f0s = self.f0_func(audios) / 100.0
        f0s_perturbed = self.f0_func(audios_perturbed) / 100.0
        content_lengths = spectrogram_lengths = style_lengths = audio_lengths // self.hp.Sound.Hop_Size

        styles = self.style_encoder(
            styles= styles,
            lengths= style_lengths
            )[:, :, None]   # [Batch, Style_d, 1]

        # Segment
        contents, offsets = self.segment(
            patterns= contents.permute(0, 2, 1),
            segment_size= self.hp.Train.Segment_Size,
            lengths= content_lengths
            )
        contents = contents.permute(0, 2, 1)    # [Batch, Content_d, Segment_t]

        contents_perturbed, _ = self.segment(
            patterns= contents_perturbed.permute(0, 2, 1),
            segment_size= self.hp.Train.Segment_Size,
            offsets= offsets
            )
        contents_perturbed = contents_perturbed.permute(0, 2, 1)    # [Batch, Linear_Spectrogram_d, Segment_t]

        audios, _ = self.segment(
            patterns= audios,
            segment_size= self.hp.Train.Segment_Size * self.hp.Sound.Hop_Size,
            offsets= offsets * self.hp.Sound.Hop_Size
            )   # [Batch, Audio_t(Segment_t * Hop_Size)]

        linear_spectrograms, _ = self.segment(
            patterns= linear_spectrograms.permute(0, 2, 1),
            segment_size= self.hp.Train.Segment_Size,
            offsets= offsets
            )
        linear_spectrograms = linear_spectrograms.permute(0, 2, 1)    # [Batch, Linear_Spectrogram_d, Segment_t]

        mel_spectrograms, _ = self.segment(
            patterns= mel_spectrograms.permute(0, 2, 1),
            segment_size= self.hp.Train.Segment_Size,
            offsets= offsets
            )
        mel_spectrograms = mel_spectrograms.permute(0, 2, 1)    # [Batch, Linear_Spectrogram_d, Segment_t]

        prosodies, _ = self.segment(
            patterns= prosodies.permute(0, 2, 1),
            segment_size= self.hp.Train.Segment_Size,
            offsets= offsets
            )
        prosodies = prosodies.permute(0, 2, 1)    # [Batch, Content_d, Segment_t]

        f0s, _ = self.segment(
            patterns= f0s,
            segment_size= self.hp.Train.Segment_Size * (self.hp.Sound.Hop_Size // self.hp.Sound.F0_Hop_Size),
            offsets= offsets * (self.hp.Sound.Hop_Size // self.hp.Sound.F0_Hop_Size)
            )   # [Batch, F0_t(Segment_t * F0_Ratio)]
        
        f0s_perturbed, _ = self.segment(
            patterns= f0s_perturbed,
            segment_size= self.hp.Train.Segment_Size * (self.hp.Sound.Hop_Size // self.hp.Sound.F0_Hop_Size),
            offsets= offsets * (self.hp.Sound.Hop_Size // self.hp.Sound.F0_Hop_Size)
            )   # [Batch, F0_t(Segment_t * F0_Ratio)]

        # Calc
        encoding_means_perturbed, encoding_log_stds_perturbed = self.speaker_agnostic_encoder(
            contents= contents_perturbed,
            f0s= f0s_perturbed,
            styles= styles
            )
        encoding_samples_perturbed = encoding_means_perturbed + encoding_log_stds_perturbed.exp() * torch.randn_like(encoding_log_stds_perturbed)
        flows_encoding_perturbed_to_clean = self.encoding_flow(
            x= encoding_samples_perturbed,
            styles= styles,
            reverse= True
            )
        
        encoding_means_clean, encoding_log_stds_clean = self.speaker_related_encoder(
            contents= contents,
            f0s= f0s,
            styles= styles
            )
        encoding_samples_clean = encoding_means_clean + encoding_log_stds_clean.exp() * torch.randn_like(encoding_log_stds_clean)
        flows_encoding_clean_to_perturbed = self.encoding_flow(
            x= encoding_samples_clean,
            styles= styles,
            reverse= False
            )
        flows_encoding_clean_to_acoustic = self.acoustic_flow(
            x= encoding_samples_clean,
            styles= styles,
            reverse= True
            )
        
        acoustic_means, acoustic_log_stds = self.acoustic_encoder(
            audios= audios,
            linear_spectrograms= linear_spectrograms,
            styles= styles
            )
        acoustic_samples = acoustic_means + acoustic_log_stds.exp() * torch.randn_like(acoustic_log_stds)
        flows_acoustic_to_encoding_clean = self.acoustic_flow(
            x= acoustic_samples,
            styles= styles,
            reverse= False
            )   # [Batch, Enc_d, Feature_t]

        prediction_prosodies = self.prosody_encoder(
            encodings= encoding_samples_clean,
            styles= styles
            )

        semantic_f0_encodings, prediction_semantic_f0s = self.semantic_f0_predictor(
            encodings= acoustic_samples,
            styles= styles
            )

        prediction_audios = self.decoder(
            encodings= acoustic_samples,
            semantic_f0_encodings= semantic_f0_encodings,
            styles= styles * (
                torch.rand_like(styles[:, :1, :1]) > 
                self.hp.Decoder.Style_Null_Probability
                ).float()
            )
        prediction_mels = self.mel_spectrogram_func(prediction_audios)

        stft_loss = torch.nn.functional.l1_loss(prediction_mels, mel_spectrograms)
        semantic_f0_loss = torch.nn.functional.l1_loss(prediction_semantic_f0s, f0s)
        prosody_loss = torch.nn.functional.l1_loss(prediction_prosodies, prosodies)
        encoding_perturbed_to_clean_kld_loss = Flow_KL_Loss(
            encoding_means= encoding_means_perturbed,
            encoding_log_stds= encoding_log_stds_perturbed,
            flows= flows_encoding_clean_to_perturbed,
            flow_log_stds= encoding_log_stds_clean,
            )
        encoding_clean_to_perturbed_kld_loss = Flow_KL_Loss(
            encoding_means= encoding_means_clean,
            encoding_log_stds= encoding_log_stds_clean,
            flows= flows_encoding_perturbed_to_clean,
            flow_log_stds= encoding_log_stds_perturbed,
            )
        encoding_clean_to_acoustic_kld_loss = Flow_KL_Loss(
            encoding_means= encoding_means_clean,
            encoding_log_stds= encoding_log_stds_clean,
            flows= flows_acoustic_to_encoding_clean,
            flow_log_stds= acoustic_log_stds,
            )
        acoustic_to_encoding_clean_kld_loss = Flow_KL_Loss(
            encoding_means= acoustic_means,
            encoding_log_stds= acoustic_log_stds,
            flows= flows_encoding_clean_to_acoustic,
            flow_log_stds= encoding_log_stds_clean,
            )

        return \
            prediction_audios, audios, \
            stft_loss, semantic_f0_loss, prosody_loss, \
            encoding_perturbed_to_clean_kld_loss, encoding_clean_to_perturbed_kld_loss, \
            encoding_clean_to_acoustic_kld_loss, acoustic_to_encoding_clean_kld_loss

    def Inference(
        self,
        source_audios: torch.FloatTensor,
        reference_audios: torch.FloatTensor,
        source_audio_lengths: Optional[torch.IntTensor]= None,
        reference_audio_lengths: Optional[torch.IntTensor]= None,
        ):
        contents = self.wav2vec2(source_audios)
        styles = self.mel_spectrogram_func(reference_audios)   # will be reference_audios
        f0s = self.f0_func(source_audios) / 100.0   # this is temporal.
        
        content_lengths = None
        if not source_audio_lengths is None:
            content_lengths = source_audio_lengths // self.hp.Sound.Hop_Size
        style_lengths = None
        if not reference_audio_lengths is None:
            style_lengths = reference_audio_lengths // self.hp.Sound.Hop_Size

        styles = self.style_encoder(
            styles= styles,
            lengths= style_lengths
            )[:, :, None]   # [Batch, Style_d, 1]

        encoding_means, encoding_log_stds = self.speaker_agnostic_encoder(
            contents= contents,
            f0s= f0s,
            styles= styles,
            lengths= content_lengths
            )        
        encoding_samples = encoding_means + encoding_log_stds.exp() * torch.randn_like(encoding_log_stds)
        encoding_samples = self.encoding_flow(
            x= encoding_samples,
            styles= styles,
            reverse= True
            )
        acoustic_samples = self.acoustic_flow(
            x= encoding_samples,
            lengths= content_lengths,
            styles= styles,
            reverse= True
            )   # [Batch, Enc_d, Feature_t]

        semantic_f0_encodings, _ = self.semantic_f0_predictor(
            encodings= acoustic_samples,
            styles= styles
            )

        audio_predictions = self.decoder(
            encodings= acoustic_samples,
            semantic_f0_encodings= semantic_f0_encodings,
            styles= styles,
            lengths= content_lengths
            )

        return audio_predictions

    def Scale_to_Tensor(
        self,
        tokens: torch.Tensor,
        scale: Union[float, List[float], torch.Tensor]
        ):
        if isinstance(scale, float):
            scale = torch.FloatTensor([scale,]).unsqueeze(0).expand(tokens.size(0), tokens.size(1))
        elif isinstance(scale, list):
            if len(scale) != tokens.size(0):
                raise ValueError(f'When scale is a list, the length must be same to the batch size: {len(scale)} != {tokens.size(0)}')
            scale = torch.FloatTensor(scale).unsqueeze(1).expand(tokens.size(0), tokens.size(1))
        elif isinstance(scale, torch.Tensor):
            if scale.ndim != 2:
                raise ValueError('When scale is a tensor, ndim must be 2.')
            elif scale.size(0) != tokens.size(0):
                raise ValueError(f'When scale is a tensor, the dimension 0 of tensor must be same to the batch size: {scale.size(0)} != {tokens.size(0)}')
            elif scale.size(1) != tokens.size(1):
                raise ValueError(f'When scale is a tensor, the dimension 1 of tensor must be same to the token length: {scale.size(1)} != {tokens.size(1)}')

        return scale.to(tokens.device)

class Source_Filter_Encoder(torch.nn.Module): 
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.content_prenet = Conv_Init(torch.nn.Conv1d(
            in_channels= 1024,
            out_channels= self.hp.Encoder.Size,
            kernel_size= 1,
            ), w_init_gain= 'linear')
        self.f0_prenet = Conv_Init(torch.nn.Conv1d(
            in_channels= 1,
            out_channels= self.hp.Encoder.Size,
            kernel_size= self.hp.Encoder.F0.Kernel_Size,
            stride= 4,
            padding= (self.hp.Encoder.F0.Kernel_Size - 1) // 2
            ), w_init_gain= 'linear')

        self.content_wavenet = Source_Filter_WaveNet(
            calc_channels= self.hp.Encoder.Size,
            style_channels= self.hp.Style.Size,
            conv_stack= self.hp.Encoder.Pre_Stack,
            kernel_size= self.hp.Encoder.Kernel_Size,
            dilation_rate= self.hp.Encoder.Dilation_Rate,
            dropout_rate= self.hp.Encoder.Dropout_Rate
            )
        self.f0_wavenet = Source_Filter_WaveNet(
            calc_channels= self.hp.Encoder.Size,
            style_channels= self.hp.Style.Size,
            conv_stack= self.hp.Encoder.Pre_Stack,
            kernel_size= self.hp.Encoder.Kernel_Size,
            dilation_rate= self.hp.Encoder.Dilation_Rate,
            dropout_rate= self.hp.Encoder.Dropout_Rate
            )
        self.post_wavenet = Source_Filter_WaveNet(
            calc_channels= self.hp.Encoder.Size,
            style_channels= self.hp.Style.Size,
            conv_stack= self.hp.Encoder.Post_Stack,
            kernel_size= self.hp.Encoder.Kernel_Size,
            dilation_rate= self.hp.Encoder.Dilation_Rate,
            dropout_rate= self.hp.Encoder.Dropout_Rate
            )

        self.projection = Conv_Init(torch.nn.Conv1d(
            in_channels= self.hp.Encoder.Size,
            out_channels= self.hp.Encoder.Size * 2,
            kernel_size= 1
            ), w_init_gain= 'linear')

    def forward(
        self,
        contents: torch.FloatTensor,
        f0s: torch.FloatTensor,
        styles: torch.FloatTensor,
        lengths: Optional[torch.IntTensor]= None,
        ) -> torch.FloatTensor:
        '''
        contents: [Batch, Content_d, Content_t], maybe wav2vec2 feature
        styles: [Batch, Style_d]
        '''
        bool_masks= None
        float_masks = 1.0
        if not lengths is None:
            bool_masks = Mask_Generate(
                lengths= lengths,
                max_length= contents.size(2)
                )[:, None]
            float_masks = (~bool_masks).float()

        contents = self.content_prenet(contents) * float_masks
        f0s = self.f0_prenet(f0s[:, None]) * float_masks    # the length of f0s is same to the length of contents by f0_prenet.
        
        contents = self.content_wavenet(
            x= contents,
            conditions= styles,
            float_masks= float_masks
            )
        f0s = self.f0_wavenet(
            x= f0s,
            conditions= styles,
            float_masks= float_masks
            )

        encodings = self.post_wavenet(
            x= contents + f0s,
            conditions= styles,
            float_masks= float_masks
            )
        means, log_stds = self.projection(encodings).chunk(chunks= 2, dim= 1)    # [Batch, Enc_d, Content_t] * 2
        
        return means, log_stds

@torch.jit.script
def Fused_Gate(x):
    x_tanh, x_sigmoid = x.chunk(chunks= 2, dim= 1)
    x = x_tanh.tanh() * x_sigmoid.sigmoid()

    return x

class Semantic_F0_Predictor(torch.nn.Module): 
    def __init__(
        self,
        hyper_parameters: Namespace,
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.prenet = Conv_Init(torch.nn.Conv1d(
            in_channels= self.hp.Encoder.Size,
            out_channels= self.hp.Semantic_F0_Predictor.Upsample.Base_Size,
            kernel_size= self.hp.Semantic_F0_Predictor.Prenet.Kernel_Size,
            padding= (self.hp.Semantic_F0_Predictor.Prenet.Kernel_Size - 1) // 2
            ), w_init_gain= 'leaky_relu'
            )
        self.style = Conv_Init(torch.nn.Conv1d(
            in_channels= self.hp.Style.Size,
            out_channels= self.hp.Semantic_F0_Predictor.Upsample.Base_Size,
            kernel_size= 1,
            ), w_init_gain= 'leaky_relu'
            )

        self.upsample_blocks = torch.nn.ModuleList()        
        self.residual_blocks = torch.nn.ModuleList()
        previous_channels= self.hp.Semantic_F0_Predictor.Upsample.Base_Size
        for index, (upsample_rate, kernel_size) in enumerate(zip(
            self.hp.Semantic_F0_Predictor.Upsample.Rate,
            self.hp.Semantic_F0_Predictor.Upsample.Kernel_Size
            )):
            current_channels = self.hp.Semantic_F0_Predictor.Upsample.Base_Size // (2 ** (index + 1))
            upsample_block = torch.nn.utils.weight_norm(torch.nn.ConvTranspose1d(
                in_channels= previous_channels,
                out_channels= current_channels,
                kernel_size= kernel_size,
                stride= upsample_rate,
                padding= (kernel_size - upsample_rate) // 2
                ))
            self.upsample_blocks.append(upsample_block)
            residual_blocks = torch.nn.ModuleList()
            for residual_kernel_size, residual_dilation_size in zip(
                self.hp.Semantic_F0_Predictor.Residual_Block.Kernel_Size,
                self.hp.Semantic_F0_Predictor.Residual_Block.Dilation_Size
                ):
                residual_blocks.append(Decoder_Residual_Block(
                    channels= current_channels,
                    kernel_size= residual_kernel_size,
                    dilations= residual_dilation_size,
                    negative_slope= self.hp.Semantic_F0_Predictor.LeakyRelu_Negative_Slope
                    ))
            self.residual_blocks.append(residual_blocks)
            previous_channels = current_channels

        # self.leaky_relu = torch.nn.LeakyReLU(
        #     negative_slope= self.hp.Semantic_F0_Predictor.LeakyRelu_Negative_Slope
        #     )
        self.aliasing_free_activation = Aliasing_Free_Activation(
            channels= previous_channels,
            snake_use_log_scale= True
            )
        self.postnet = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels= previous_channels,
                out_channels= 1,
                kernel_size= self.hp.Semantic_F0_Predictor.Postnet.Kernel_Size,
                padding= (self.hp.Semantic_F0_Predictor.Postnet.Kernel_Size - 1) // 2,
                bias= False,
                ),
            Lambda(lambda x: x[:, 0])
            )

        # This is critical when using weight normalization.
        def weight_norm_initialize_weight(module):
            if 'Conv' in module.__class__.__name__:
                module.weight.data.normal_(0.0, 0.01)
        self.upsample_blocks.apply(weight_norm_initialize_weight)
        self.residual_blocks.apply(weight_norm_initialize_weight)

    def Remove_Weight_Norm(self):
        for block in self.upsample_blocks:
            torch.nn.utils.remove_weight_norm(block[1])

        for blocks in self.residual_blocks:
            for block in blocks:
                for conv in block.in_convs + block.out_convs:
                    torch.nn.utils.remove_weight_norm(conv)

    def forward(
        self,
        encodings: torch.FloatTensor,
        styles: torch.FloatTensor,
        lengths: Optional[torch.Tensor]= None,
        ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        bool_masks= None
        float_masks = 1.0
        if not lengths is None:
            bool_masks = Mask_Generate(
                lengths= lengths,
                max_length= encodings.size(2)
                )
            float_masks = (~bool_masks)[:, None].float()

        x = (self.prenet(encodings) + self.style(styles)) * float_masks
        for upsample_block, residual_blocks, upsample_rate in zip(
            self.upsample_blocks,
            self.residual_blocks,
            self.hp.Semantic_F0_Predictor.Upsample.Rate
            ):
            x = upsample_block(x)
            if not lengths is None:
                lengths = lengths * upsample_rate
                float_masks = (~Mask_Generate(
                    lengths= lengths,
                    max_length= x.size(2)
                    ))[:, None].float()

            x = torch.stack(
                [block(x, float_masks) for block in residual_blocks],
                dim= 1
                ).mean(dim= 1)
        
        # semantic_f0_encodings = self.leaky_relu(x)
        semantic_f0_encodings = self.aliasing_free_activation(x)
        prediction_semantic_f0s = self.postnet(semantic_f0_encodings)

        return semantic_f0_encodings, prediction_semantic_f0s


class Decoder(torch.nn.Module): 
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.encoding_prenet = torch.nn.Conv1d(
            in_channels= self.hp.Encoder.Size,
            out_channels= self.hp.Decoder.Upsample.Base_Size,
            kernel_size= self.hp.Decoder.Prenet.Kernel_Size,
            padding= (self.hp.Decoder.Prenet.Kernel_Size - 1) // 2
            )
        self.f0_prenet = F0_Prenet(self.hp)
        self.style = torch.nn.Conv1d(
            in_channels= self.hp.Style.Size,
            out_channels= self.hp.Decoder.Upsample.Base_Size,
            kernel_size= 1
            )

        self.f0_mid = torch.nn.Conv1d(
            in_channels= \
                self.hp.Semantic_F0_Predictor.Upsample.Base_Size // \
                2 ** len(self.hp.Semantic_F0_Predictor.Upsample.Rate),
            out_channels= self.hp.Decoder.Upsample.Base_Size // 2,
            kernel_size= 1
            )

        self.upsample_blocks = torch.nn.ModuleList()
        self.residual_blocks = torch.nn.ModuleList()
        previous_channels= self.hp.Decoder.Upsample.Base_Size
        for index, (upsample_rate, kernel_size) in enumerate(zip(
            self.hp.Decoder.Upsample.Rate,
            self.hp.Decoder.Upsample.Kernel_Size
            )):
            current_channels = self.hp.Decoder.Upsample.Base_Size // (2 ** (index + 1))
            upsample_block = torch.nn.Sequential(
                torch.nn.LeakyReLU(
                    negative_slope= self.hp.Decoder.LeakyRelu_Negative_Slope
                    ),
                torch.nn.utils.weight_norm(torch.nn.ConvTranspose1d(
                    in_channels= previous_channels,
                    out_channels= current_channels,
                    kernel_size= kernel_size,
                    stride= upsample_rate,
                    padding= (kernel_size - upsample_rate) // 2
                    ))
                )
            self.upsample_blocks.append(upsample_block)
            residual_blocks = torch.nn.ModuleList()
            for residual_kernel_size, residual_dilation_size in zip(
                self.hp.Decoder.Residual_Block.Kernel_Size,
                self.hp.Decoder.Residual_Block.Dilation_Size
                ):
                residual_blocks.append(Decoder_Residual_Block(
                    channels= current_channels,
                    kernel_size= residual_kernel_size,
                    dilations= residual_dilation_size,
                    negative_slope= self.hp.Decoder.LeakyRelu_Negative_Slope
                    ))
            self.residual_blocks.append(residual_blocks)
            previous_channels = current_channels

        self.postnet = torch.nn.Sequential(
            # torch.nn.LeakyReLU(),
            Aliasing_Free_Activation(
                channels= previous_channels,
                snake_use_log_scale= True
                ),
            torch.nn.Conv1d(
                in_channels= previous_channels,
                out_channels= 1,
                kernel_size= self.hp.Decoder.Postnet.Kernel_Size,
                padding= (self.hp.Decoder.Postnet.Kernel_Size - 1) // 2,
                bias= False,
                ),
            torch.nn.Tanh(),
            Lambda(lambda x: x.squeeze(1))
            )

        # This is critical when using weight normalization.
        def weight_norm_initialize_weight(module):
            if 'Conv' in module.__class__.__name__:
                module.weight.data.normal_(0.0, 0.01)
        self.upsample_blocks.apply(weight_norm_initialize_weight)
        self.residual_blocks.apply(weight_norm_initialize_weight)
            
    def forward(
        self,
        encodings: torch.FloatTensor,
        semantic_f0_encodings: torch.FloatTensor,
        styles: torch.FloatTensor,
        lengths: Optional[torch.IntTensor]= None,
        ) -> torch.Tensor:
        bool_masks= None
        float_masks = 1.0
        if not lengths is None:
            bool_masks = Mask_Generate(
                lengths= lengths,
                max_length= encodings.size(2)
                )
            float_masks = (~bool_masks)[:, None].float()

        decodings = (self.encoding_prenet(encodings) + self.f0_prenet(semantic_f0_encodings) + self.style(styles)) * float_masks
        for index, (upsample_block, residual_blocks, upsample_rate) in enumerate(zip(
            self.upsample_blocks,
            self.residual_blocks,
            self.hp.Decoder.Upsample.Rate
            )):
            decodings = upsample_block(decodings)
            if index == 0:
                semantic_f0_encodings
                decodings = decodings + self.f0_mid(semantic_f0_encodings)

            if not lengths is None:
                lengths = lengths * upsample_rate
                float_masks = (~Mask_Generate(
                    lengths= lengths,
                    max_length= decodings.size(2)
                    )).unsqueeze(1).float()
            decodings = torch.stack(
                [block(decodings, float_masks) for block in residual_blocks],
                dim= 1
                ).mean(dim= 1)
            
        predictions = self.postnet(decodings)

        return predictions

class Decoder_Residual_Block(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilations: Union[List, Tuple],
        negative_slope: float= 0.1
        ):
        super().__init__()

        self.in_aliasing_free_activations = torch.nn.ModuleList()
        self.in_convs = torch.nn.ModuleList()
        self.out_aliasing_free_activations = torch.nn.ModuleList()
        self.out_convs = torch.nn.ModuleList()
        for dilation in dilations:
            self.in_aliasing_free_activations.append(Aliasing_Free_Activation(
                channels= channels,
                snake_use_log_scale= True
                ))
            self.in_convs.append(torch.nn.utils.weight_norm(torch.nn.Conv1d(
                in_channels= channels,
                out_channels= channels,
                kernel_size= kernel_size,
                dilation= dilation,
                padding= (kernel_size * dilation - dilation) // 2
                )))
            self.out_aliasing_free_activations.append(Aliasing_Free_Activation(
                channels= channels,
                snake_use_log_scale= True
                ))
            self.out_convs.append(torch.nn.utils.weight_norm(torch.nn.Conv1d(
                in_channels= channels,
                out_channels= channels,
                kernel_size= kernel_size,
                dilation= 1,
                padding= (kernel_size - 1) // 2
                )))

        self.leaky_relu = torch.nn.LeakyReLU(negative_slope= negative_slope)

    def forward(
        self,
        x: torch.Tensor,
        masks: torch.Tensor
        ):
        for in_aliasing_free_activation, in_conv, out_aliasing_free_activation, out_conv in zip(
            self.in_aliasing_free_activations,
            self.in_convs,
            self.out_aliasing_free_activations,
            self.out_convs
            ):
            residuals = x
            # x = self.leaky_relu(x) * masks
            x = in_aliasing_free_activation(x) * masks
            x = in_conv(x) * masks
            # x = self.leaky_relu(x) * masks
            x = out_aliasing_free_activation(x) * masks
            x = out_conv(x) * masks
            x = x + residuals
        
        return x * masks

class F0_Prenet(torch.nn.Module):
    def __init__(
        self,
        hyper_parameters: Namespace,
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.convs = torch.nn.ModuleList()

        previous_channels = \
            self.hp.Semantic_F0_Predictor.Upsample.Base_Size // \
            2 ** len(self.hp.Semantic_F0_Predictor.Upsample.Rate)
        
        self.residual = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            in_channels= previous_channels,
            out_channels= self.hp.Decoder.Upsample.Base_Size,
            kernel_size= 1
            ))

        for kernel_size, dilation in zip(
            self.hp.Decoder.F0_Prenet.Kernel_Size,
            self.hp.Decoder.F0_Prenet.Dilation_Size
            ):
            current_channels = self.hp.Decoder.Upsample.Base_Size
            self.convs.append(torch.nn.Sequential(
                torch.nn.LeakyReLU(
                    negative_slope= self.hp.Decoder.LeakyRelu_Negative_Slope
                    ),
                torch.nn.utils.weight_norm(torch.nn.Conv1d(
                    in_channels= previous_channels,
                    out_channels= current_channels,
                    kernel_size= kernel_size,
                    dilation= dilation,
                    padding= (kernel_size * dilation - dilation) // 2
                    ))
                ))
            previous_channels = current_channels

    def forward(
        self,
        semantic_f0_encodings: torch.FloatTensor
        ):
        scale_factor = self.hp.Sound.F0_Hop_Size / self.hp.Sound.Hop_Size

        residuals = self.residual(semantic_f0_encodings)
        residuals = torch.nn.functional.interpolate(residuals, scale_factor= scale_factor)
        
        x = torch.nn.functional.interpolate(semantic_f0_encodings, scale_factor= scale_factor)
        for conv in self.convs:
            x = conv(x)

        return x + residuals

class Prosody_Encoder(torch.nn.Module):
    def __init__(
        self,
        hyper_parameters: Namespace
        ) -> None:
        super().__init__()
        self.hp = hyper_parameters

        self.prenet = Conv_Init(torch.nn.Conv1d(
            in_channels= self.hp.Encoder.Size,
            out_channels= self.hp.Encoder.Size,
            kernel_size= self.hp.Prosody_Encoder.Prenet.Kernel_Size,
            padding= (self.hp.Prosody_Encoder.Prenet.Kernel_Size - 1) // 2
            ), w_init_gain= 'linear')
        self.style = Conv_Init(torch.nn.Conv1d(
            in_channels= self.hp.Style.Size,
            out_channels= self.hp.Encoder.Size,
            kernel_size= 1
            ), w_init_gain= 'linear')
        
        self.blocks = torch.nn.ModuleList([
            FFT_Block(
                channels= self.hp.Prosody_Encoder.Size,
                num_head= self.hp.Prosody_Encoder.Residual.Head,
                ffn_kernel_size= self.hp.Prosody_Encoder.Residual.Kernel_Size,
                dropout_rate= self.hp.Prosody_Encoder.Residual.Dropout_Rate,
                )
            for index in range(self.hp.Prosody_Encoder.Residual.Stack)
            ])
        
        self.projection = Conv_Init(torch.nn.Conv1d(
            in_channels= self.hp.Encoder.Size,
            out_channels= self.hp.Sound.N_Mel // 4,
            kernel_size= 1,
            bias= False
            ), w_init_gain= 'linear')

    def forward(
        self,
        encodings: torch.FloatTensor,
        styles: torch.FloatTensor,
        lengths: Optional[torch.IntTensor]= None
        ):
        bool_masks= None
        float_masks = 1.0
        if not lengths is None:
            bool_masks = Mask_Generate(
                lengths= lengths,
                max_length= encodings.size(2)
                )
            float_masks = (~bool_masks)[:, None].float()

        x = (self.prenet(encodings) + self.style(styles)) * float_masks
        for block in self.blocks:
            x = block(x, lengths)
        x = self.projection(x) * float_masks

        return x


class Acoustic_Encoder(torch.nn.Module): 
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.audio_prenet = Conv_Init(torch.nn.Conv1d(
            in_channels= 1,
            out_channels= self.hp.Acoustic_Encoder.Audio.Downsample.Base_Size,
            kernel_size= self.hp.Acoustic_Encoder.Audio.Prenet.Kernel_Size,
            padding= (self.hp.Acoustic_Encoder.Audio.Prenet.Kernel_Size - 1) // 2
            ), w_init_gain= 'linear')

        self.audio_downsample_blocks = torch.nn.ModuleList()
        self.audio_residual_blocks = torch.nn.ModuleList()
        previous_channels= self.hp.Acoustic_Encoder.Audio.Downsample.Base_Size
        for index, (downsample_rate, kernel_size) in enumerate(zip(
            self.hp.Acoustic_Encoder.Audio.Downsample.Rate,
            self.hp.Acoustic_Encoder.Audio.Downsample.Kernel_Size
            )):
            current_channels = self.hp.Acoustic_Encoder.Audio.Downsample.Base_Size * (2 ** (index + 1))
            downsample_block = torch.nn.utils.weight_norm(torch.nn.Conv1d(
                in_channels= previous_channels,
                out_channels= current_channels,
                kernel_size= kernel_size,
                stride= downsample_rate,
                padding= (kernel_size - 1) // 2
                ))
            self.audio_downsample_blocks.append(downsample_block)
            residual_blocks = torch.nn.ModuleList()
            for residual_kernel_size, residual_dilation_size in zip(
                self.hp.Acoustic_Encoder.Audio.Residual_Block.Kernel_Size,
                self.hp.Acoustic_Encoder.Audio.Residual_Block.Dilation_Size
                ):
                residual_blocks.append(Decoder_Residual_Block(
                    channels= current_channels,
                    kernel_size= residual_kernel_size,
                    dilations= residual_dilation_size,
                    negative_slope= self.hp.Acoustic_Encoder.Audio.LeakyRelu_Negative_Slope
                    ))
            self.audio_residual_blocks.append(residual_blocks)
            previous_channels = current_channels

        self.audio_postnet = torch.nn.Sequential(
            # torch.nn.LeakyReLU(
            #     negative_slope= self.hp.Acoustic_Encoder.Audio.LeakyRelu_Negative_Slope
            #     ),
            Aliasing_Free_Activation(
                channels= previous_channels,
                snake_use_log_scale= True
                ),
            torch.nn.Conv1d(
                in_channels= previous_channels,
                out_channels= self.hp.Encoder.Size,
                kernel_size= self.hp.Acoustic_Encoder.Audio.Postnet.Kernel_Size,
                padding= (self.hp.Acoustic_Encoder.Audio.Postnet.Kernel_Size - 1) // 2,
                bias= False,
                ),
            )

        self.linear_spectrogram_prenet = Conv_Init(torch.nn.Conv1d(
            in_channels= self.hp.Sound.N_FFT // 2 + 1,
            out_channels= self.hp.Encoder.Size,
            kernel_size= 1,
            ), w_init_gain= 'linear')
        self.linear_spectrogram_wavenet = WaveNet(
            calc_channels= self.hp.Encoder.Size,
            conv_stack= self.hp.Acoustic_Encoder.Linear_Spectrogram.Conv_Stack,
            kernel_size= self.hp.Acoustic_Encoder.Linear_Spectrogram.Kernel_Size,
            dilation_rate= self.hp.Acoustic_Encoder.Linear_Spectrogram.Dilation_Rate,
            dropout_rate= self.hp.Acoustic_Encoder.Linear_Spectrogram.Dropout_Rate,
            condition_channels= self.hp.Style.Size
            )        
        self.projection = Conv_Init(torch.nn.Conv1d(
            in_channels= self.hp.Encoder.Size * 2,
            out_channels= self.hp.Encoder.Size * 2,
            kernel_size= 1,
            ), w_init_gain= 'linear')
            
    def forward(
        self,
        audios: torch.FloatTensor,
        linear_spectrograms: torch.FloatTensor,
        styles: torch.FloatTensor,
        audio_lengths: Optional[torch.IntTensor]= None,
        linear_spectrogram_lengths: Optional[torch.IntTensor]= None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        features: [Batch, Feature_d, Feature_t], Spectrogram
        lengths: [Batch]
        '''
        audio_bool_masks= None
        audio_float_masks = 1.0
        if not audio_lengths is None:
            audio_bool_masks = Mask_Generate(
                lengths= audio_lengths,
                max_length= audios.size(1)
                )[:, None]
            audio_float_masks = (~audio_bool_masks).float()

        linear_spectrogram_bool_masks= None
        linear_spectrogram_float_masks = 1.0
        if not linear_spectrogram_lengths is None:
            linear_spectrogram_bool_masks = Mask_Generate(
                lengths= linear_spectrogram_lengths,
                max_length= linear_spectrograms.size(2)
                )[:, None]
            linear_spectrogram_float_masks = (~linear_spectrogram_bool_masks).float()

        audio_encodings = self.audio_prenet(audios[:, None])
        for downsample_block, residual_blocks, downsample_rate in zip(
            self.audio_downsample_blocks,
            self.audio_residual_blocks,
            self.hp.Acoustic_Encoder.Audio.Downsample.Rate
            ):
            audio_encodings = downsample_block(audio_encodings)

            if not audio_lengths is None:
                audio_lengths = audio_lengths // downsample_rate            
                audio_float_masks = (~Mask_Generate(
                    lengths= audio_lengths,
                    max_length= audio_encodings.size(2)
                    ))[:, None].float()

            audio_encodings = torch.stack(
                [block(audio_encodings, audio_float_masks) for block in residual_blocks],
                dim= 1
                ).mean(dim= 1)
            
        audio_encodings = self.audio_postnet(audio_encodings) * audio_float_masks

        linear_spectrograms_encodings = self.linear_spectrogram_prenet(linear_spectrograms) * linear_spectrogram_float_masks   # [Batch, Acoustic_d, Feature_t]
        linear_spectrograms_encodings = self.linear_spectrogram_wavenet(
            linear_spectrograms_encodings,
            linear_spectrogram_float_masks,
            conditions= styles
            )
        means, stds = self.projection(torch.cat([audio_encodings, linear_spectrograms_encodings], dim= 1)).chunk(chunks= 2, dim= 1)   # [Batch, Acoustic_d, Feature_t] * 2
        log_stds = torch.nn.functional.softplus(stds).log()
        
        return means, log_stds

class Segment(torch.nn.Module):
    def forward(
        self,
        patterns: torch.Tensor,
        segment_size: int,
        lengths: torch.Tensor= None,
        offsets: torch.Tensor= None
        ):
        '''
        patterns: [Batch, Time, ...]
        lengths: [Batch]
        segment_size: an integer scalar
        '''
        if offsets is None:
            offsets = (torch.rand_like(patterns[:, 0, 0]) * (lengths - segment_size)).long()
        segments = torch.stack([
            pattern[offset:offset + segment_size]
            for pattern, offset in zip(patterns, offsets)
            ], dim= 0)
        
        return segments, offsets

def Mask_Generate(lengths: torch.Tensor, max_length: Optional[Union[int, torch.Tensor]]= None):
    '''
    lengths: [Batch]
    max_lengths: an int value. If None, max_lengths == max(lengths)
    '''
    max_length = max_length or torch.max(lengths)
    sequence = torch.arange(max_length)[None, :].to(lengths.device)
    return sequence >= lengths[:, None]    # [Batch, Time]

import torchaudio
from random import choice
def Perturbing(
    audios: torch.FloatTensor,
    sample_rate: int= 16000,
    f0_perturbation_factor_min: int= -12,
    f0_perturbation_factor_max: int= 12,
    pes_low_cutoff: int= 60,
    pes_high_cutoff: int= 10000,
    pes_q_min: float= 2.0,
    pes_q_max: float= 5.0,
    gain_min: float= -12.0,
    gain_max: float= 12.0,
    ):
    step = choice(list(range(f0_perturbation_factor_min, f0_perturbation_factor_max + 1, 1)))
    audios = torchaudio.functional.pitch_shift(
        audios,
        sample_rate= sample_rate,
        n_steps= step,
        bins_per_octave=12
        )
    if torch.rand(1) > 0.5:
        audios = torchaudio.functional.pitch_shift(
            audios,
            sample_rate= sample_rate,
            n_steps= -step,
            bins_per_octave= 12
            )
    with torch.cuda.amp.autocast(enabled= False):   # equalizer_biquad does not support fp16
        audios = audios.float()
        cutoff_frequency_hls = torch.tensor(min(pes_low_cutoff // 2, sample_rate // 2)).to(audios.device)
        cutoff_frequency_hhs = torch.tensor(min(pes_high_cutoff // 2, sample_rate // 2)).to(audios.device) # when over sample_rate // 2, audio is broken.
        num_peaking_filters = 8
        frequencies_peak = torch.logspace(cutoff_frequency_hls.log10(), cutoff_frequency_hhs.log10(), num_peaking_filters + 2)[1:-1]

        audios = torchaudio.functional.equalizer_biquad(
            waveform= audios,
            sample_rate= sample_rate,
            center_freq= cutoff_frequency_hhs,
            gain= 0.0
            )
        audios = torchaudio.functional.equalizer_biquad(
            waveform= audios,
            sample_rate= sample_rate,
            center_freq= cutoff_frequency_hls,
            gain= 0.0
            )
        for x in frequencies_peak:
            audios = torchaudio.functional.equalizer_biquad(
            waveform= audios,
            sample_rate= sample_rate,
            center_freq= x,
            gain= (gain_max - gain_min) * torch.rand(1) + gain_min,
            Q= pes_q_min * (pes_q_max - pes_q_min) ** torch.rand(1)
            )

    return audios