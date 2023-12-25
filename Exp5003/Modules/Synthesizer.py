from argparse import Namespace
import torch, torchaudio
import math
from typing import Optional, List, Dict, Tuple, Union
from functools import partial
import transformers

from .Common import WaveNet, FFT_Block, Mask_Generate
from .Style_Encoder import Style_Encoder
from .Flow import FlowBlock_Transformer, Flow_KL_Loss
from .Layer import Conv_Init, Lambda
from .Resample import UpSample1d, DownSample1d

from meldataset import mel_spectrogram
from yin import estimate as yin


class Synthesizer(torch.nn.Module):
    def __init__(
        self,
        hyper_parameters: Namespace,
        f0_mean: float,
        f0_std: float
        ):
        super().__init__()
        self.hp = hyper_parameters
        self.f0_mean = f0_mean
        self.f0_std = f0_std

        # audio preprocess
        self.wav2vec2 = Wav2vec2()
        self.f0_func = partial(
            yin,
            sample_rate= self.hp.Sound.Sample_Rate,
            pitch_min= self.hp.Sound.F0_Min,
            pitch_max= self.hp.Sound.F0_Max,
            hop_size= self.hp.Sound.F0_Hop_Size,
            window_size= self.hp.Sound.F0_Hop_Size * 4,
            threshold= 0.1
            )
        self.mel_func = partial(
            mel_spectrogram,
            n_fft= self.hp.Sound.N_FFT,
            num_mels= self.hp.Sound.N_Mel,
            sampling_rate= self.hp.Sound.Sample_Rate,
            hop_size= self.hp.Sound.Hop_Size,
            win_size= self.hp.Sound.Window_Size,
            fmin= 0,
            fmax= None,
            use_normalize= False
            )
        self.prosody_func = partial(
            mel_spectrogram,
            n_fft= self.hp.Sound.N_FFT,
            num_mels= self.hp.Sound.N_Mel // 4,
            sampling_rate= self.hp.Sound.Sample_Rate,
            hop_size= self.hp.Sound.Hop_Size,
            win_size= self.hp.Sound.Window_Size,
            fmin= 0,
            fmax= None,
            use_normalize= False
            )

        self.style_encoder = Style_Encoder(
            in_channels= self.hp.Sound.N_Mel,
            out_channels= self.hp.Text2Wav2Vec.Style.Size,
            temporal_kernel_size= self.hp.Text2Wav2Vec.Style.Temporal_kernel_Size,
            num_heads= self.hp.Text2Wav2Vec.Style.Head,
            dropout_rate= self.hp.Text2Wav2Vec.Style.Dropout_Rate,
            )
        
        self.speaker_agnostic_encoder = Source_Filter_Encoder(self.hp)
        self.speaker_related_encoder = Source_Filter_Encoder(self.hp)
        
        self.acoustic_encoder = Acoustic_Encoder(self.hp)
        
        self.source_filter_flow = FlowBlock_Transformer(
            channels= self.hp.Synthesizer.Source_Filter_Encoder.Size,
            calc_channels= self.hp.Synthesizer.Source_Filter_Encoder.Size,
            condition_channels= self.hp.Synthesizer.Style.Size,
            flow_stack= self.hp.Synthesizer.Source_Filter_Flow.Stack,
            flow_ditblock_stack= self.hp.Synthesizer.Source_Filter_Flow.DiT_Stack,
            flow_ditblock_num_heads= self.hp.Synthesizer.Source_Filter_Flow.Head,
            flow_ditblock_ffn_kernel_size= self.hp.Synthesizer.Source_Filter_Flow.Kernel_Size,
            flow_ditblock_dropout_rate= self.hp.Synthesizer.Source_Filter_Flow.Dropout_Rate,
            )
        self.acoustic_flow = FlowBlock_Transformer(
            channels= self.hp.Synthesizer.Source_Filter_Encoder.Size,
            calc_channels= self.hp.Synthesizer.Source_Filter_Encoder.Size,
            condition_channels= self.hp.Synthesizer.Style.Size,
            flow_stack= self.hp.Synthesizer.Acoustic_Flow.Stack,
            flow_ditblock_stack= self.hp.Synthesizer.Acoustic_Flow.DiT_Stack,
            flow_ditblock_num_heads= self.hp.Synthesizer.Acoustic_Flow.Head,
            flow_ditblock_ffn_kernel_size= self.hp.Synthesizer.Acoustic_Flow.Kernel_Size,
            flow_ditblock_dropout_rate= self.hp.Synthesizer.Acoustic_Flow.Dropout_Rate,
            )        
        
        self.semantic_f0_predictor = Semantic_F0_Predictor(self.hp)
        
        self.decoder = Decoder(self.hp)
        
        self.prosody_encoder = Prosody_Encoder(self.hp)
        
        self.segment = Segment()

    def forward(
        self,
        audios: torch.FloatTensor,
        audio_lengths: torch.IntTensor,
        linear_spectrograms: torch.FloatTensor,
        mel_spectrograms: torch.FloatTensor,
        spectrogram_lengths: torch.IntTensor
        ):
        # preprocess
        f0s = self.f0_func(audios) / 100.0
        audios_perturbed = Perturbing(audios)
        f0s_perturbed = self.f0_func(audios_perturbed) / 100.0
        contents = self.wav2vec2(audios)    # [Batch, Content_d, Content_t]
        contents_perturbed = self.wav2vec2(audios_perturbed)    # [Batch, Content_d, Content_t]
        prosodies = self.prosody_func(audios)   # [Batch, Prosody_d, Content_t]
        content_lengths = style_lengths = spectrogram_lengths   # [Batch]
        styles = mel_spectrograms   # [Batch, Mel_d, Content_t]

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
        contents_perturbed = contents_perturbed.permute(0, 2, 1)    # [Batch, Content_d, Segment_t]

        prosodies, _ = self.segment(
            patterns= prosodies.permute(0, 2, 1),
            segment_size= self.hp.Train.Segment_Size,
            offsets= offsets
            )
        prosodies = prosodies.permute(0, 2, 1)    # [Batch, Content_d, Segment_t]

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
        mel_spectrograms = mel_spectrograms.permute(0, 2, 1)    # [Batch, Mel_d, Segment_t]

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
        encoding_flows_clean_from_perturbed = self.source_filter_flow(
            x= encoding_samples_perturbed,
            styles= styles,
            reverse= True
            )

        encoding_means_clean, encoding_log_stds_clean = self.speaker_related_encoder(
            contents= contents,
            f0s= f0s,
            styles= styles,
            )
        encoding_samples_clean = encoding_means_clean + encoding_log_stds_clean.exp() * torch.randn_like(encoding_log_stds_clean)
        encoding_flows_perturbed_from_clean = self.source_filter_flow(
            x= encoding_samples_clean,
            styles= styles,
            reverse= False
            )
        acoustic_flows = self.acoustic_flow(
            x= encoding_samples_clean,
            styles= styles,
            reverse= True
            )

        acoustic_means, acoustic_log_stds = self.acoustic_encoder(
            linear_spectrograms= linear_spectrograms,
            audios= audios,
            styles= styles
            )
        acoustic_samples = acoustic_means + acoustic_log_stds.exp() * torch.randn_like(acoustic_log_stds)
        encoding_flows_clean_from_acoustic = self.acoustic_flow(
            x= acoustic_samples,
            styles= styles,
            reverse= False
            )

        prediction_prosodies = self.prosody_encoder(
            encodings= encoding_samples_clean,
            styles= styles
            )

        semantic_f0_encodings, prediction_semantic_f0s = self.semantic_f0_predictor(
            encodings= acoustic_samples,
            styles= styles
            )
        
        prediction_audios = self.decoder(
            acoustic_samples= acoustic_samples,
            semantic_f0_encodings= semantic_f0_encodings,
            styles= styles * (
                torch.rand_like(styles[:, :1, :1]) > 
                self.hp.Synthesizer.Decoder.Style_Null_Probability
                ).float()
            )
        prediction_mels = self.mel_func(prediction_audios)

        with torch.cuda.amp.autocast(enabled= False):
            stft_loss = torch.nn.functional.l1_loss(prediction_mels, mel_spectrograms)
            semantic_f0_loss = torch.nn.functional.l1_loss(
                prediction_semantic_f0s,
                f0s,
                )
            prosody_loss = torch.nn.functional.l1_loss(
                prediction_prosodies,
                prosodies,
                )
            encoding_perturbed_to_clean_kld_loss = Flow_KL_Loss(
                encoding_means= encoding_means_clean,
                encoding_log_stds= encoding_log_stds_clean,
                flows= encoding_flows_clean_from_perturbed,
                flow_log_stds= encoding_log_stds_perturbed,
                )
            encoding_clean_to_perturbed_kld_loss = Flow_KL_Loss(
                encoding_means= encoding_means_perturbed,
                encoding_log_stds= encoding_log_stds_perturbed,
                flows= encoding_flows_perturbed_from_clean,
                flow_log_stds= encoding_log_stds_clean,
                )
            acoustic_to_encoding_clean_kld_loss = Flow_KL_Loss(
                encoding_means= encoding_means_clean,
                encoding_log_stds= encoding_log_stds_clean,
                flows= encoding_flows_clean_from_acoustic,
                flow_log_stds= acoustic_log_stds,
                )
            encoding_clean_to_acoustic_kld_loss = Flow_KL_Loss(
                encoding_means= acoustic_means,
                encoding_log_stds= acoustic_log_stds,
                flows= acoustic_flows,
                flow_log_stds= encoding_log_stds_clean,
                )

        return \
            prediction_audios, audios, \
            encoding_perturbed_to_clean_kld_loss, encoding_clean_to_perturbed_kld_loss, \
            acoustic_to_encoding_clean_kld_loss, encoding_clean_to_acoustic_kld_loss, \
            stft_loss, semantic_f0_loss, prosody_loss

    def Inference_VC(
        self,
        source_audios: torch.FloatTensor,
        source_f0s: torch.FloatTensor,
        styles: torch.FloatTensor,
        source_audio_lengths: Optional[torch.IntTensor]= None,
        style_lengths: Optional[torch.IntTensor]= None,
        ):
        contents = self.wav2vec2(source_audios)
        content_lengths = source_audio_lengths // self.hp.Sound.Hop_Size

        if not style_lengths is None:
            style_bool_masks = Mask_Generate(
                lengths= style_lengths,
                max_length= styles.size(2)
                )[:, None, :]
            styles.masked_fill_(mask= style_bool_masks, value= 0.0)
            styles.masked_fill_(mask= style_bool_masks, value= styles.min())

        styles = self.style_encoder(
            styles= styles,
            lengths= style_lengths
            )[:, :, None]   # [Batch, Style_d, 1]
        
        encoding_means, encoding_log_stds = self.speaker_agnostic_encoder(
            contents= contents,
            f0s= source_f0s,
            styles= styles,
            lengths= content_lengths,
            )
        
        encoding_samples = self.source_filter_flow(
            x= encoding_means + encoding_log_stds.exp() * torch.randn_like(encoding_log_stds),
            styles= styles,
            lengths= content_lengths,
            reverse= True
            )

        acoustic_samples = self.acoustic_flow(
            x= encoding_samples,
            styles= styles,
            lengths= content_lengths,
            reverse= True
            )
        
        semantic_f0_encodings, prediction_semantic_f0s = self.semantic_f0_predictor(
            encodings= acoustic_samples,
            styles= styles,
            lengths= content_lengths
            )

        prediction_audios = self.decoder(
            acoustic_samples= acoustic_samples,
            semantic_f0_encodings= semantic_f0_encodings,
            styles= styles,
            lengths= content_lengths
            )
        
        return prediction_audios, prediction_semantic_f0s
    
    def Inference_TTS(
        self,
        contents: torch.FloatTensor,
        f0s: torch.FloatTensor,
        styles: torch.FloatTensor,
        content_lengths: Optional[torch.IntTensor]= None,
        style_lengths: Optional[torch.IntTensor]= None,
        ):        
        styles = self.style_encoder(
            styles= styles,
            lengths= style_lengths
            )[:, :, None]   # [Batch, Style_d, 1]        
        
        encoding_means, encoding_log_stds = self.speaker_agnostic_encoder(
            contents= contents,
            f0s= f0s,
            styles= styles,
            lengths= content_lengths,
            )
        
        encoding_samples = self.source_filter_flow(
            x= encoding_means + encoding_log_stds.exp() * torch.randn_like(encoding_log_stds),
            styles= styles,
            lengths= content_lengths,
            reverse= True
            )

        acoustic_samples = self.acoustic_flow(
            x= encoding_samples,
            styles= styles,
            lengths= content_lengths,
            reverse= True
            )
        
        semantic_f0_encodings, _ = self.semantic_f0_predictor(
            encodings= acoustic_samples,
            styles= styles,
            lengths= content_lengths
            )
        
        prediction_audios = self.decoder(
            acoustic_samples= acoustic_samples,
            semantic_f0_encodings= semantic_f0_encodings,
            styles= styles,
            lengths= content_lengths
            )
        
        return prediction_audios


class Source_Filter_Encoder(torch.nn.Module): 
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.content_prenet = Conv_Init(torch.nn.Conv1d(
            in_channels= self.hp.Text2Wav2Vec.Size,
            out_channels= self.hp.Synthesizer.Source_Filter_Encoder.Size,
            kernel_size= 1,
            ), w_init_gain= 'linear')
        self.f0_prenet = Conv_Init(torch.nn.Conv1d(
            in_channels= 1,
            out_channels= self.hp.Synthesizer.Source_Filter_Encoder.Size,
            kernel_size= self.hp.Synthesizer.Source_Filter_Encoder.F0.Kernel_Size,
            stride= math.prod(self.hp.Text2Wav2Vec.F0_Predictor.Upsample.Rate),
            padding= (self.hp.Synthesizer.Source_Filter_Encoder.F0.Kernel_Size - 1) // 2
            ), w_init_gain= 'linear')
        
        self.content_wavenet = WaveNet(
            calc_channels= self.hp.Synthesizer.Source_Filter_Encoder.Size,
            style_channels= self.hp.Synthesizer.Style.Size,
            conv_stack= self.hp.Synthesizer.Source_Filter_Encoder.Pre_Stack,
            kernel_size= self.hp.Synthesizer.Source_Filter_Encoder.Kernel_Size,
            dilation_rate= self.hp.Synthesizer.Source_Filter_Encoder.Dilation_Rate,
            dropout_rate= self.hp.Synthesizer.Source_Filter_Encoder.Dropout_Rate
            )
        self.f0_wavenet = WaveNet(
            calc_channels= self.hp.Synthesizer.Source_Filter_Encoder.Size,
            style_channels= self.hp.Synthesizer.Style.Size,
            conv_stack= self.hp.Synthesizer.Source_Filter_Encoder.Pre_Stack,
            kernel_size= self.hp.Synthesizer.Source_Filter_Encoder.Kernel_Size,
            dilation_rate= self.hp.Synthesizer.Source_Filter_Encoder.Dilation_Rate,
            dropout_rate= self.hp.Synthesizer.Source_Filter_Encoder.Dropout_Rate
            )
        self.post_wavenet = WaveNet(
            calc_channels= self.hp.Synthesizer.Source_Filter_Encoder.Size,
            style_channels= self.hp.Synthesizer.Style.Size,
            conv_stack= self.hp.Synthesizer.Source_Filter_Encoder.Post_Stack,
            kernel_size= self.hp.Synthesizer.Source_Filter_Encoder.Kernel_Size,
            dilation_rate= self.hp.Synthesizer.Source_Filter_Encoder.Dilation_Rate,
            dropout_rate= self.hp.Synthesizer.Source_Filter_Encoder.Dropout_Rate
            )

        self.projection = Conv_Init(torch.nn.Conv1d(
            in_channels= self.hp.Synthesizer.Source_Filter_Encoder.Size,
            out_channels= self.hp.Synthesizer.Source_Filter_Encoder.Size * 2,
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

class Acoustic_Encoder(torch.nn.Module): 
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.audio_prenet = Conv_Init(torch.nn.Conv1d(
            in_channels= 1,
            out_channels= self.hp.Synthesizer.Acoustic_Encoder.Audio.Downsample.Base_Size,
            kernel_size= self.hp.Synthesizer.Acoustic_Encoder.Audio.Prenet.Kernel_Size,
            padding= (self.hp.Synthesizer.Acoustic_Encoder.Audio.Prenet.Kernel_Size - 1) // 2
            ), w_init_gain= 'linear')
        
        self.downsample_blocks = torch.nn.ModuleList()
        self.residual_blocks = torch.nn.ModuleList()
        previous_channels= self.hp.Synthesizer.Acoustic_Encoder.Audio.Downsample.Base_Size
        for index, (downsample_rate, kernel_size) in enumerate(zip(
            self.hp.Synthesizer.Acoustic_Encoder.Audio.Downsample.Rate,
            self.hp.Synthesizer.Acoustic_Encoder.Audio.Downsample.Kernel_Size
            )):
            current_channels = self.hp.Synthesizer.Acoustic_Encoder.Audio.Downsample.Base_Size * (2 ** (index + 1))
            downsample_block = torch.nn.utils.weight_norm(torch.nn.Conv1d(
                in_channels= previous_channels,
                out_channels= current_channels,
                kernel_size= kernel_size,
                stride= downsample_rate,
                padding= (kernel_size - 1) // 2
                ))
            self.downsample_blocks.append(downsample_block)
            residual_blocks = torch.nn.ModuleList()
            for residual_kernel_size, residual_dilation_size in zip(
                self.hp.Synthesizer.Acoustic_Encoder.Audio.Residual_Block.Kernel_Size,
                self.hp.Synthesizer.Acoustic_Encoder.Audio.Residual_Block.Dilation_Size
                ):
                residual_blocks.append(Acoustic_Encoder_Residual_Block(
                    channels= current_channels,
                    kernel_size= residual_kernel_size,
                    dilations= residual_dilation_size
                    ))
            self.residual_blocks.append(residual_blocks)
            previous_channels = current_channels

        self.audio_postnet = torch.nn.Sequential(
            Aliasing_Activation(
                channels= previous_channels,
                snake_use_log_scale= True
                ),
            torch.nn.Conv1d(
                in_channels= previous_channels,
                out_channels= self.hp.Synthesizer.Source_Filter_Encoder.Size,
                kernel_size= self.hp.Synthesizer.Acoustic_Encoder.Audio.Postnet.Kernel_Size,
                padding= (self.hp.Synthesizer.Acoustic_Encoder.Audio.Postnet.Kernel_Size - 1) // 2,
                bias= False,                
                ),
            )


        self.linear_spectrogram_prenet = Conv_Init(torch.nn.Conv1d(
            in_channels= self.hp.Sound.N_FFT // 2 + 1,
            out_channels= self.hp.Synthesizer.Source_Filter_Encoder.Size,
            kernel_size= 1,
            ), w_init_gain= 'linear')
        
        self.linear_spectrogram_wavenet = WaveNet(
            calc_channels= self.hp.Synthesizer.Source_Filter_Encoder.Size,
            style_channels= self.hp.Synthesizer.Style.Size,
            conv_stack= self.hp.Synthesizer.Acoustic_Encoder.Linear_Spectrogram.Wavenet.Stack,
            kernel_size= self.hp.Synthesizer.Acoustic_Encoder.Linear_Spectrogram.Wavenet.Kernel_Size,
            dilation_rate= self.hp.Synthesizer.Acoustic_Encoder.Linear_Spectrogram.Wavenet.Dilation_Rate,
            )
        
        self.projection = Conv_Init(torch.nn.Conv1d(
            in_channels= self.hp.Synthesizer.Source_Filter_Encoder.Size * 2,
            out_channels= self.hp.Synthesizer.Source_Filter_Encoder.Size * 2,
            kernel_size= 1
            ), w_init_gain= 'linear')

    def forward(
        self,
        linear_spectrograms: torch.FloatTensor,
        audios: torch.FloatTensor,
        styles: torch.FloatTensor,
        linear_spectrogram_lengths: Optional[torch.IntTensor]= None,
        audio_lengths: Optional[torch.IntTensor]= None, # linear_spectrogram_lengths // hop_size
        ):
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

        audios = self.audio_prenet(audios[:, None])
        for downsample_block, residual_blocks, downsample_rate in zip(
            self.downsample_blocks,
            self.residual_blocks,
            self.hp.Synthesizer.Acoustic_Encoder.Audio.Downsample.Rate
            ):
            audios = downsample_block(audios)

            if not audio_lengths is None:
                audio_lengths = audio_lengths // downsample_rate            
                audio_float_masks = (~Mask_Generate(
                    lengths= audio_lengths,
                    max_length= audios.size(2)
                    ))[:, None].float()

            audios = torch.stack(
                [block(audios, audio_float_masks) for block in residual_blocks],
                dim= 1
                ).mean(dim= 1)
            
        audios = self.audio_postnet(audios) * audio_float_masks


        linear_spectrograms = self.linear_spectrogram_prenet(linear_spectrograms) * linear_spectrogram_float_masks
        linear_spectrograms = self.linear_spectrogram_wavenet(
            x= linear_spectrograms,
            conditions= styles,
            float_masks= linear_spectrogram_float_masks
            )
        
        means, log_stds = self.projection(torch.cat([audios, linear_spectrograms], dim= 1)).chunk(chunks= 2, dim= 1)
        
        return means, log_stds

class Acoustic_Encoder_Residual_Block(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilations: Union[List, Tuple]
        ):
        super().__init__()

        self.in_snakes = torch.nn.ModuleList()
        self.in_convs = torch.nn.ModuleList()
        self.out_snakes = torch.nn.ModuleList()
        self.out_convs = torch.nn.ModuleList()
        for dilation in dilations:
            self.in_snakes.append(Aliasing_Activation(
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
            self.out_snakes.append(Aliasing_Activation(
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

    def forward(
        self,
        x: torch.Tensor,
        float_masks: Union[torch.Tensor, float]= 1.0
        ):
        for in_snake, in_conv, out_snake, out_conv in zip(self.in_snakes, self.in_convs, self.out_snakes, self.out_convs):
            residuals = x
            x = in_snake(x) * float_masks
            x = in_conv(x)
            x = out_snake(x) * float_masks
            x = out_conv(x)
            x = x + residuals
        
        return x * float_masks

class Aliasing_Activation(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        upsample_ratio: int= 2,
        downsample_ratio: int= 2,
        upsample_kernel_size: int= 12,
        downsample_kernel_size: int= 12,
        snake_alpha: float= 1.0,
        snake_use_log_scale: float= False
        ):
        super().__init__()

        self.upsample = UpSample1d(
            ratio= upsample_ratio,
            kernel_size= upsample_kernel_size,
            )
        self.snake = Snake(
            channels= channels,
            alpha= snake_alpha,
            use_log_scale= snake_use_log_scale
            )
        self.downsample = DownSample1d(
            ratio= downsample_ratio,
            kernel_size= downsample_kernel_size,
            )

    def forward(self, x: torch.FloatTensor):
        x = self.upsample(x)
        x = self.snake(x)
        x = self.downsample(x)

        return x

class Snake(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        alpha: float= 1.0,
        use_log_scale: bool= False
        ):
        super().__init__()
        self.use_log_scale = use_log_scale

        self.alpha = torch.nn.Parameter(torch.empty(channels))
        self.beta = torch.nn.Parameter(torch.empty(channels))

        torch.nn.init.constant_(self.alpha, val= 0.0 if use_log_scale else alpha)
        torch.nn.init.constant_(self.beta, val= 0.0 if use_log_scale else alpha)

    def forward(self, x: torch.FloatTensor):
        alpha = self.alpha[None, :, None]
        beta = self.beta[None, :, None]

        if self.use_log_scale:
            alpha = alpha.exp()
            beta = beta.exp()

        return x + (1.0 / (beta + 1e-5)) * (x * alpha).sin().pow(2.0)

class Semantic_F0_Predictor(torch.nn.Module): 
    def __init__(
        self,
        hyper_parameters: Namespace,
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.prenet = Conv_Init(torch.nn.Conv1d(
            in_channels= self.hp.Synthesizer.Source_Filter_Encoder.Size,
            out_channels= self.hp.Synthesizer.Semantic_F0_Predictor.Upsample.Base_Size,
            kernel_size= self.hp.Synthesizer.Semantic_F0_Predictor.Prenet.Kernel_Size,
            padding= (self.hp.Synthesizer.Semantic_F0_Predictor.Prenet.Kernel_Size - 1) // 2
            ), w_init_gain= 'leaky_relu'
            )
        self.style = Conv_Init(torch.nn.Conv1d(
            in_channels= self.hp.Synthesizer.Style.Size,
            out_channels= self.hp.Synthesizer.Semantic_F0_Predictor.Upsample.Base_Size,
            kernel_size= 1,
            ), w_init_gain= 'leaky_relu'
            )

        self.upsample_blocks = torch.nn.ModuleList()        
        self.residual_blocks = torch.nn.ModuleList()
        previous_channels= self.hp.Synthesizer.Semantic_F0_Predictor.Upsample.Base_Size
        for index, (upsample_rate, kernel_size) in enumerate(zip(
            self.hp.Synthesizer.Semantic_F0_Predictor.Upsample.Rate,
            self.hp.Synthesizer.Semantic_F0_Predictor.Upsample.Kernel_Size
            )):
            current_channels = self.hp.Synthesizer.Semantic_F0_Predictor.Upsample.Base_Size // (2 ** (index + 1))
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
                self.hp.Synthesizer.Semantic_F0_Predictor.Residual_Block.Kernel_Size,
                self.hp.Synthesizer.Semantic_F0_Predictor.Residual_Block.Dilation_Size
                ):
                residual_blocks.append(Acoustic_Encoder_Residual_Block(
                    channels= current_channels,
                    kernel_size= residual_kernel_size,
                    dilations= residual_dilation_size
                    ))
            self.residual_blocks.append(residual_blocks)
            previous_channels = current_channels

        self.post_activation_func = Aliasing_Activation(
            channels= previous_channels,
            snake_use_log_scale= True
            )
        self.postnet = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels= previous_channels,
                out_channels= 1,
                kernel_size= self.hp.Synthesizer.Semantic_F0_Predictor.Postnet.Kernel_Size,
                padding= (self.hp.Synthesizer.Semantic_F0_Predictor.Postnet.Kernel_Size - 1) // 2,
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
            self.hp.Synthesizer.Semantic_F0_Predictor.Upsample.Rate
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
        
        semantic_f0_encodings = self.post_activation_func(x)
        prediction_semantic_f0s = self.postnet(semantic_f0_encodings)

        return semantic_f0_encodings, prediction_semantic_f0s

class Decoder(torch.nn.Module):
    def __init__(
        self,
        hyper_parameters: Namespace,
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.encoding_prenet = torch.nn.Conv1d(
            in_channels= self.hp.Synthesizer.Source_Filter_Encoder.Size,
            out_channels= self.hp.Synthesizer.Decoder.Upsample.Base_Size,
            kernel_size= self.hp.Synthesizer.Decoder.Prenet.Kernel_Size,
            padding= (self.hp.Synthesizer.Decoder.Prenet.Kernel_Size - 1) // 2,
            )
        self.f0_prenet = F0_Prenet(self.hp)
        self.style = torch.nn.Conv1d(
            in_channels= self.hp.Synthesizer.Style.Size,
            out_channels= self.hp.Synthesizer.Decoder.Upsample.Base_Size,
            kernel_size= 1
            )
        
        self.f0_mid = torch.nn.Conv1d(
            in_channels= \
                self.hp.Synthesizer.Semantic_F0_Predictor.Upsample.Base_Size // \
                2 ** len(self.hp.Synthesizer.Semantic_F0_Predictor.Upsample.Rate),
            out_channels= self.hp.Synthesizer.Decoder.Upsample.Base_Size // 2,
            kernel_size= 1
            )


        self.upsample_blocks = torch.nn.ModuleList()
        self.residual_blocks = torch.nn.ModuleList()
        previous_channels= self.hp.Synthesizer.Decoder.Upsample.Base_Size
        for index, (upsample_rate, kernel_size) in enumerate(zip(
            self.hp.Synthesizer.Decoder.Upsample.Rate,
            self.hp.Synthesizer.Decoder.Upsample.Kernel_Size
            )):
            current_channels = self.hp.Synthesizer.Decoder.Upsample.Base_Size // (2 ** (index + 1))
            upsample_block = torch.nn.Sequential(
                torch.nn.LeakyReLU(
                    negative_slope= self.hp.Synthesizer.Decoder.LeakyRelu_Negative_Slope
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
                self.hp.Synthesizer.Decoder.Residual_Block.Kernel_Size,
                self.hp.Synthesizer.Decoder.Residual_Block.Dilation_Size
                ):
                residual_blocks.append(Acoustic_Encoder_Residual_Block(
                    channels= current_channels,
                    kernel_size= residual_kernel_size,
                    dilations= residual_dilation_size
                    ))
            self.residual_blocks.append(residual_blocks)
            previous_channels = current_channels

        self.postnet = torch.nn.Sequential(
            Aliasing_Activation(
                channels= previous_channels,
                snake_use_log_scale= True
                ),
            torch.nn.Conv1d(
                in_channels= previous_channels,
                out_channels= 1,
                kernel_size= self.hp.Synthesizer.Decoder.Postnet.Kernel_Size,
                padding= (self.hp.Synthesizer.Decoder.Postnet.Kernel_Size - 1) // 2,
                bias= False
                ),
            torch.nn.Tanh(),
            Lambda(lambda x: x.squeeze(1))
            )

        # This is critical when using weight normalization.
        def weight_norm_initialize_weight(module):
            if 'Conv' in module.__class__.__name__:
                module.weight.data.normal_(0.0, 0.01)
        self.f0_prenet.apply(weight_norm_initialize_weight)
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
        acoustic_samples: torch.FloatTensor,
        semantic_f0_encodings: torch.FloatTensor,
        styles: torch.FloatTensor,
        lengths: Optional[torch.IntTensor]= None,
        ) -> torch.Tensor:
        bool_masks= None
        float_masks = 1.0
        if not lengths is None:
            bool_masks = Mask_Generate(
                lengths= lengths,
                max_length= acoustic_samples.size(2)
                )
            float_masks = (~bool_masks)[:, None].float()

        decodings = self.encoding_prenet(acoustic_samples) + self.f0_prenet(semantic_f0_encodings) + self.style(styles)
        for index, (upsample_block, residual_blocks, upsample_rate) in enumerate(zip(
            self.upsample_blocks,
            self.residual_blocks,
            self.hp.Synthesizer.Decoder.Upsample.Rate
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

class F0_Prenet(torch.nn.Module):
    def __init__(
        self,
        hyper_parameters: Namespace,
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.convs = torch.nn.ModuleList()

        previous_channels = \
            self.hp.Synthesizer.Semantic_F0_Predictor.Upsample.Base_Size // \
            2 ** len(self.hp.Synthesizer.Semantic_F0_Predictor.Upsample.Rate)
        
        self.residual = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            in_channels= previous_channels,
            out_channels= self.hp.Synthesizer.Decoder.Upsample.Base_Size,
            kernel_size= 1
            ))

        for kernel_size, dilation in zip(
            self.hp.Synthesizer.Decoder.F0_Prenet.Kernel_Size,
            self.hp.Synthesizer.Decoder.F0_Prenet.Dilation_Size
            ):
            current_channels = self.hp.Synthesizer.Decoder.Upsample.Base_Size
            self.convs.append(torch.nn.Sequential(
                torch.nn.LeakyReLU(
                    negative_slope= self.hp.Synthesizer.Decoder.LeakyRelu_Negative_Slope
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
            in_channels= self.hp.Synthesizer.Source_Filter_Encoder.Size,
            out_channels= self.hp.Synthesizer.Source_Filter_Encoder.Size,
            kernel_size= self.hp.Synthesizer.Prosody_Encoder.Prenet.Kernel_Size,
            padding= (self.hp.Synthesizer.Prosody_Encoder.Prenet.Kernel_Size - 1) // 2
            ), w_init_gain= 'linear')
        self.style = Conv_Init(torch.nn.Conv1d(
            in_channels= self.hp.Synthesizer.Style.Size,
            out_channels= self.hp.Synthesizer.Source_Filter_Encoder.Size,
            kernel_size= 1
            ), w_init_gain= 'linear')
        
        self.blocks = torch.nn.ModuleList([
            FFT_Block(
                channels= self.hp.Synthesizer.Prosody_Encoder.Size,
                num_head= self.hp.Synthesizer.Prosody_Encoder.Residual.Head,
                ffn_kernel_size= self.hp.Synthesizer.Prosody_Encoder.Residual.Kernel_Size,
                dropout_rate= self.hp.Synthesizer.Prosody_Encoder.Residual.Dropout_Rate,
                )
            for index in range(self.hp.Synthesizer.Prosody_Encoder.Residual.Stack)
            ])
        
        self.projection = Conv_Init(torch.nn.Conv1d(
            in_channels= self.hp.Synthesizer.Source_Filter_Encoder.Size,
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

class Wav2vec2(torch.nn.Module):
    def __init__(
        self,
        layer: int= 7,
        hop_size: int= 320
        ):
        super().__init__()
        self.hop_size = hop_size

        self.wav2vec2 = transformers.Wav2Vec2ForPreTraining.from_pretrained("facebook/mms-300m")

        for param in self.wav2vec2.parameters():
            param.requires_grad = False
            param.grad = None

        self.wav2vec2.eval()
        self.feature_layer = layer

    @torch.no_grad()
    def forward(self, audios):
        padding = torch.zeros_like(audios[:, :self.hop_size // 2])
        audios = torch.concat([padding, audios, padding], dim= 1)
        outputs = self.wav2vec2(audios, output_hidden_states=True)
        contents = outputs.hidden_states[self.feature_layer].permute(0, 2, 1)

        return contents.detach()

    def train(self, mode: bool= True):
        super().train(mode= mode)
        self.wav2vec2.eval()

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