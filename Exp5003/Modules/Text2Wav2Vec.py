from argparse import Namespace
import torch
import math
from functools import partial
from typing import Optional, List, Dict, Tuple, Union

from .Wav2Vec2 import Wav2vec2
from .Common import WaveNet, FFT_Block, Mask_Generate
from .Style_Encoder import Style_Encoder
from .Monotonic_Alignment_Search import Calc_Duration
from .Stochastic_Duration_Predictor import Stochastic_Duration_Predictor
from .Flow import FlowBlock_Transformer, Flow_KL_Loss
from .Layer import Conv_Init, Lambda
from meldataset import mel_spectrogram
from yin import estimate as yin

class Text2Wav2Vec(torch.nn.Module):
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
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
        self.text_encoder = Token_Encoder(self.hp)
        self.content_encoder = Content_Encoder(self.hp)
        self.duration_predictor = Stochastic_Duration_Predictor(
            channels= self.hp.Token_Encoder.Size,
            calc_channels= self.hp.Token_Encoder.Size,
            kernel_size= self.hp.Duration_Predictor.Kernel_Size,
            conv_stack= self.hp.Duration_Predictor.Conv_Stack,
            flow_stack= self.hp.Duration_Predictor.Flow_Stack,
            dropout_rate= self.hp.Duration_Predictor.Dropout_Rate,
            condition_channels= self.hp.Style.Size,
            )
        self.f0_predictor = F0_Predictor(self.hp)

        self.flow = FlowBlock_Transformer(
            channels= self.hp.Token_Encoder.Size,
            calc_channels= self.hp.Token_Encoder.Size,
            condition_channels= self.hp.Style.Size,
            flow_stack= self.hp.Flow.Stack,
            flow_ditblock_stack= self.hp.Flow.DiT_Stack,
            flow_ditblock_num_heads= self.hp.Flow.Head,
            flow_ditblock_ffn_kernel_size= self.hp.Flow.Kernel_Size,
            flow_ditblock_dropout_rate= self.hp.Flow.Dropout_Rate,
            )

        self.decoder = Decoder(self.hp)

        self.token_predictor = Token_Predictor(self.hp)

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
        tokens_with_between_pad: torch.IntTensor,
        token_lengths_with_between_pad: torch.IntTensor,
        audios: torch.FloatTensor,
        audio_lengths: torch.IntTensor,
        tokens_without_between_pad: torch.IntTensor,
        token_lengths_without_between_pad: torch.IntTensor,
        ):
        contents = self.wav2vec2(audios)
        styles = self.mel_spectrogram_func(audios)   # will be reference_audios
        f0s = self.f0_func(audios) / 100.0
        content_lengths = style_lengths = audio_lengths // self.hp.Sound.Hop_Size

        styles = self.style_encoder(
            styles= styles,
            lengths= style_lengths
            )[:, :, None]   # [Batch, Style_d, 1]

        encoding_means, encoding_log_stds, encodings = self.text_encoder(
            tokens= tokens_with_between_pad,
            styles= styles,
            lengths= token_lengths_with_between_pad,
            )
        
        content_means, content_log_stds = self.content_encoder(
            contents= contents,
            styles= styles,
            lengths= content_lengths
            )
        content_samples = content_means + content_log_stds.exp() * torch.randn_like(content_log_stds)
        content_flows = self.flow(
            x= content_samples,
            styles= styles,
            lengths= content_lengths,
            reverse= False
            )   # [Batch, Enc_d, Feature_t]

        durations, alignments = Calc_Duration(
            encoding_means= encoding_means,
            encoding_log_stds= encoding_log_stds,
            encoding_lengths= token_lengths_with_between_pad,
            decodings= content_flows,
            decoding_lengths= content_lengths,
            )
        
        encoding_means = encoding_means @ alignments.permute(0, 2, 1)
        encoding_log_stds = encoding_log_stds @ alignments.permute(0, 2, 1)
        encoding_samples = encoding_means + encoding_log_stds.exp() * torch.randn_like(encoding_log_stds)
        encoding_flows = self.flow(
            x= encoding_samples,
            styles= styles,
            lengths= content_lengths,
            reverse= True
            )   # [Batch, Enc_d, Feature_t]

        prediction_contents = self.decoder(
            contents= content_samples,
            styles= styles,
            lengths= content_lengths
            )
        prediction_f0s = self.f0_predictor(
            contents= prediction_contents,
            styles= styles,
            lengths= content_lengths
            )
        
        prediction_tokens = self.token_predictor(
            encodings= content_samples
            )


        with torch.cuda.amp.autocast(enabled= False):
            content_to_encoding_kld_loss = Flow_KL_Loss(
                encoding_means= encoding_means,
                encoding_log_stds= encoding_log_stds,
                flows= content_flows,
                flow_log_stds= content_log_stds,
                float_masks= (~Mask_Generate(
                    lengths= content_lengths,
                    max_length= contents.size(2)
                    ))[:, None].float()
                )
            encoding_to_content_kld_loss = Flow_KL_Loss(
                encoding_means= content_means,
                encoding_log_stds= content_log_stds,
                flows= encoding_flows,
                flow_log_stds= encoding_log_stds,
                float_masks= (~Mask_Generate(
                    lengths= content_lengths,
                    max_length= contents.size(2)
                    ))[:, None].float()
                )

            content_loss = torch.nn.functional.l1_loss(prediction_contents, contents)
            _, duration_loss = self.duration_predictor(
                encodings= encodings,
                encoding_lengths= token_lengths_with_between_pad,
                durations= durations,
                conditions= styles,
                reverse= False
                )
            duration_loss = duration_loss.float().mean()
            f0_loss = torch.nn.functional.l1_loss(prediction_f0s, f0s)
            token_ctc_loss = torch.nn.functional.ctc_loss(
                log_probs= prediction_tokens.permute(2, 0, 1),
                targets= tokens_without_between_pad,
                input_lengths= content_lengths,
                target_lengths= token_lengths_without_between_pad,
                blank= self.hp.Tokens,
                zero_infinity= True
                )

        return \
            content_loss, content_to_encoding_kld_loss, encoding_to_content_kld_loss, \
            duration_loss, f0_loss, token_ctc_loss

    def Inference(
        self,
        tokens: torch.IntTensor,
        audios: torch.FloatTensor,
        token_lengths: Optional[torch.IntTensor]= None,
        audio_lengths: Optional[torch.IntTensor]= None
        ):
        styles = self.mel_spectrogram_func(audios)   # will be reference_audios
        style_lengths = audio_lengths // self.hp.Sound.Hop_Size

        styles = self.style_encoder(
            styles= styles,
            lengths= style_lengths
            )[:, :, None]   # [Batch, Style_d, 1]

        encoding_means, encoding_log_stds, encodings = self.text_encoder(
            tokens= tokens,
            styles= styles,
            lengths= token_lengths,            
            )
        
        durations, _ = self.duration_predictor(
            encodings= encodings,
            encoding_lengths= token_lengths,
            conditions= styles,
            reverse= True
            )
        alignments = self.Length_Regulate(durations)
        content_lengths = alignments.sum(dim= [1, 2])
        
        encoding_means = encoding_means @ alignments.permute(0, 2, 1)
        encoding_log_stds = encoding_log_stds @ alignments.permute(0, 2, 1)
        encoding_samples = encoding_means + encoding_log_stds.exp() * torch.randn_like(encoding_log_stds)        
        content_samples = self.flow(
            x= encoding_samples,
            styles= styles,
            lengths= content_lengths,
            reverse= True
            )   # [Batch, Enc_d, Feature_t]

        contents = self.decoder(
            contents= content_samples,
            styles= styles,
            lengths= content_lengths
            )
        f0s = self.f0_predictor(
            contents= contents,
            styles= styles,
            lengths= content_lengths
            )

        return contents, f0s, alignments

    def Length_Regulate(self, durations):
        """If target=None, then predicted durations are applied"""
        repeats = (durations.float() + 0.5).long()
        decoding_lengths = repeats.sum(dim=1)

        max_decoding_length = decoding_lengths.max()
        reps_cumsum = torch.cumsum(torch.nn.functional.pad(repeats, (1, 0, 0, 0), value=0.0), dim=1)[:, None, :]

        range_ = torch.arange(max_decoding_length)[None, :, None].to(durations.device)
        alignments = ((reps_cumsum[:, :, :-1] <= range_) &
                (reps_cumsum[:, :, 1:] > range_))
        
        return alignments.float()
    
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

class Content_Encoder(torch.nn.Module): 
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.prenet = Conv_Init(torch.nn.Conv1d(
            in_channels= self.hp.Wav2Vec2_Size,
            out_channels= self.hp.Token_Encoder.Size,
            kernel_size= 1,
            ), w_init_gain= 'linear')
        
        self.wavenet = WaveNet(
            calc_channels= self.hp.Token_Encoder.Size,
            style_channels= self.hp.Style.Size,
            conv_stack= self.hp.Content_Encoder.Wavenet.Stack,
            kernel_size= self.hp.Content_Encoder.Wavenet.Kernel_Size,
            dilation_rate= self.hp.Content_Encoder.Wavenet.Dilation_Rate
            )

        self.projection = Conv_Init(torch.nn.Conv1d(
            in_channels= self.hp.Token_Encoder.Size,
            out_channels= self.hp.Token_Encoder.Size * 2,
            kernel_size= 1
            ), w_init_gain= 'linear')

    def forward(
        self,
        contents: torch.FloatTensor,
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
                )
            float_masks = (~bool_masks)[:, None].float()

        contents = self.prenet(contents) * float_masks
        contents = self.wavenet(
            x= contents,
            conditions= styles,
            float_masks= float_masks
            )
        means, log_stds = self.projection(contents).chunk(chunks= 2, dim= 1)    # [Batch, Enc_d, Content_t] * 2
        
        return means, log_stds

class Token_Encoder(torch.nn.Module): 
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.token_embedding = torch.nn.Embedding(
            num_embeddings= self.hp.Tokens,
            embedding_dim= self.hp.Token_Encoder.Size,
            )
        embedding_variance = math.sqrt(3.0) * math.sqrt(2.0 / (self.hp.Tokens + self.hp.Token_Encoder.Size))
        self.token_embedding.weight.data.uniform_(-embedding_variance, embedding_variance)

        self.style = torch.nn.Conv1d(
            in_channels= self.hp.Style.Size,
            out_channels= self.hp.Token_Encoder.Size,
            kernel_size= 1
            )

        self.pre_encoder_residual_blocks = torch.nn.ModuleList([
            FFT_Block(
                channels= self.hp.Token_Encoder.Size,
                num_head= self.hp.Token_Encoder.Residual.Head,
                ffn_kernel_size= self.hp.Token_Encoder.Residual.Kernel_Size,
                dropout_rate= self.hp.Token_Encoder.Residual.Dropout_Rate,
                )
            for index in range(self.hp.Token_Encoder.Residual.Pre_Stack)
            ])
        
        self.post_encoder_residual_blocks = torch.nn.ModuleList([
            FFT_Block(
                channels= self.hp.Token_Encoder.Size,
                num_head= self.hp.Token_Encoder.Residual.Head,
                ffn_kernel_size= self.hp.Token_Encoder.Residual.Kernel_Size,
                dropout_rate= self.hp.Token_Encoder.Residual.Dropout_Rate,
                )
            for index in range(self.hp.Token_Encoder.Residual.Post_Stack)
            ])

        self.projection = Conv_Init(torch.nn.Conv1d(
            in_channels= self.hp.Token_Encoder.Size,
            out_channels= self.hp.Token_Encoder.Size * 2,
            kernel_size= 1,
            ), w_init_gain= 'linear')

    def forward(
        self,
        tokens: torch.IntTensor,
        styles: torch.FloatTensor,
        lengths: Optional[torch.IntTensor]= None,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        tokens: [Batch, Time]
        '''
        bool_masks= None
        float_masks = 1.0
        if not lengths is None:
            bool_masks = Mask_Generate(
                lengths= lengths,
                max_length= tokens.size(1)
                )
            float_masks = (~bool_masks)[:, None].float()

        encodings = self.token_embedding(tokens).permute(0, 2, 1) * math.sqrt(self.hp.Token_Encoder.Size)

        for block in self.pre_encoder_residual_blocks:
            encodings = block(
                x= encodings * float_masks,
                lengths= lengths
                )
            
        encodings = encodings + self.style(styles)

        for block in self.post_encoder_residual_blocks:
            encodings = block(
                x= encodings * float_masks,
                lengths= lengths
                )

        means, log_stds = self.projection(encodings).chunk(chunks= 2, dim= 1)   # [Batch, Acoustic_d, Feature_t] * 2

        return means, log_stds, encodings

class Decoder(torch.nn.Module): 
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.prenet = Conv_Init(torch.nn.Conv1d(
            in_channels= self.hp.Token_Encoder.Size,
            out_channels= self.hp.Decoder.Size,
            kernel_size= 1,
            ), w_init_gain= 'linear')
        
        self.wavenet = WaveNet(
            calc_channels= self.hp.Decoder.Size,
            style_channels= self.hp.Style.Size,
            conv_stack= self.hp.Decoder.Stack,
            kernel_size= self.hp.Decoder.Kernel_Size,
            dilation_rate= self.hp.Decoder.Dilation_Rate,
            dropout_rate= self.hp.Decoder.Dropout_Rate
            )

        self.projection = Conv_Init(torch.nn.Conv1d(
            in_channels= self.hp.Decoder.Size,
            out_channels= self.hp.Wav2Vec2_Size,
            kernel_size= 1
            ), w_init_gain= 'linear')
        
        self.leaky_relu = torch.nn.LeakyReLU(
            negative_slope= self.hp.Decoder.LeakyReLU_Negative_Slope
            )

    def forward(
        self,
        contents: torch.FloatTensor,
        styles: torch.FloatTensor,
        lengths: Optional[torch.IntTensor]= None,
        ) -> torch.FloatTensor:
        '''
        tokens: [Batch, Time]
        '''
        bool_masks= None
        float_masks = 1.0
        if not lengths is None:
            bool_masks = Mask_Generate(
                lengths= lengths,
                max_length= contents.size(2)
                )
            float_masks = (~bool_masks)[:, None].float()

        contents = self.prenet(contents) * float_masks
        contents = self.wavenet(
            x= contents,
            conditions= styles,
            float_masks= float_masks
            )
        contents = self.projection(contents) # [Batch, Content_d, Content_t]
        
        return contents

class F0_Predictor(torch.nn.Module): 
    def __init__(
        self,
        hyper_parameters: Namespace,
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.prenet = Conv_Init(torch.nn.Conv1d(
            in_channels= self.hp.Wav2Vec2_Size,
            out_channels= self.hp.F0_Predictor.Upsample.Base_Size,
            kernel_size= self.hp.F0_Predictor.Prenet.Kernel_Size,
            padding= (self.hp.F0_Predictor.Prenet.Kernel_Size - 1) // 2
            ), w_init_gain= 'leaky_relu'
            )
        self.style = Conv_Init(torch.nn.Conv1d(
            in_channels= self.hp.Style.Size,
            out_channels= self.hp.F0_Predictor.Upsample.Base_Size,
            kernel_size= 1,
            ), w_init_gain= 'leaky_relu'
            )

        self.upsample_blocks = torch.nn.ModuleList()        
        self.residual_blocks = torch.nn.ModuleList()
        previous_channels= self.hp.F0_Predictor.Upsample.Base_Size
        for index, (upsample_rate, kernel_size) in enumerate(zip(
            self.hp.F0_Predictor.Upsample.Rate,
            self.hp.F0_Predictor.Upsample.Kernel_Size
            )):
            current_channels = self.hp.F0_Predictor.Upsample.Base_Size // (2 ** (index + 1))
            upsample_block = torch.nn.Sequential(
                torch.nn.LeakyReLU(
                    negative_slope= self.hp.F0_Predictor.LeakyReLU_Negative_Slope
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
                self.hp.F0_Predictor.Residual_Block.Kernel_Size,
                self.hp.F0_Predictor.Residual_Block.Dilation_Size
                ):
                residual_blocks.append(F0_Predictor_Residual_Block(
                    channels= current_channels,
                    kernel_size= residual_kernel_size,
                    dilations= residual_dilation_size,
                    negative_slope= self.hp.F0_Predictor.LeakyReLU_Negative_Slope
                    ))
            self.residual_blocks.append(residual_blocks)
            previous_channels = current_channels

        self.postnet = torch.nn.Sequential(
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(
                in_channels= previous_channels,
                out_channels= 1,
                kernel_size= self.hp.F0_Predictor.Postnet.Kernel_Size,
                padding= (self.hp.F0_Predictor.Postnet.Kernel_Size - 1) // 2,
                bias= False,                
                ),
            torch.nn.Tanh(),
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
        contents: torch.FloatTensor,
        styles: torch.FloatTensor,
        lengths: torch.Tensor,
        ) -> torch.Tensor:
        bool_masks= None
        float_masks = 1.0
        if not lengths is None:
            bool_masks = Mask_Generate(
                lengths= lengths,
                max_length= contents.size(2)
                )
            float_masks = (~bool_masks)[:, None].float()

        x = (self.prenet(contents) + self.style(styles)) * float_masks
        for upsample_block, residual_blocks, upsample_rate in zip(
            self.upsample_blocks,
            self.residual_blocks,
            self.hp.F0_Predictor.Upsample.Rate
            ):
            x = upsample_block(x)
            lengths = lengths * upsample_rate
            
            float_masks = (~Mask_Generate(
                lengths= lengths,
                max_length= x.size(2)
                ))[:, None].float()

            x = torch.stack(
                [block(x, float_masks) for block in residual_blocks],
                dim= 1
                ).mean(dim= 1)
            
        prediction_f0s = self.postnet(x)

        return prediction_f0s

class F0_Predictor_Residual_Block(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilations: Union[List, Tuple],
        negative_slope: float= 0.1
        ):
        super().__init__()

        self.in_convs = torch.nn.ModuleList()
        self.out_convs = torch.nn.ModuleList()
        for dilation in dilations:
            self.in_convs.append(torch.nn.utils.weight_norm(torch.nn.Conv1d(
                in_channels= channels,
                out_channels= channels,
                kernel_size= kernel_size,
                dilation= dilation,
                padding= (kernel_size * dilation - dilation) // 2
                )))
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
        float_masks: Union[torch.Tensor, float]= 1.0
        ):
        for in_conv, out_conv in zip(self.in_convs, self.out_convs):
            residuals = x
            x = self.leaky_relu(x) * float_masks
            x = in_conv(x)
            x = self.leaky_relu(x) * float_masks
            x = out_conv(x)
            x = x + residuals
        
        return x * float_masks

class Token_Predictor(torch.nn.Module):
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.lstm = torch.nn.LSTM(
            input_size= self.hp.Token_Encoder.Size,
            hidden_size= self.hp.Token_Predictor.Size,
            num_layers= self.hp.Token_Predictor.LSTM.Stack
            )
        self.lstm_dropout = torch.nn.Dropout(
            p= self.hp.Token_Predictor.LSTM.Dropout_Rate
            )
        self.projection = Conv_Init(torch.nn.Conv1d(
            in_channels= self.hp.Token_Predictor.Size,
            out_channels= self.hp.Tokens + 1,
            kernel_size= 1,
            ), w_init_gain= 'linear')
        
    def forward(
        self,
        encodings: torch.Tensor    
        ) -> torch.Tensor:
        encodings = encodings.permute(2, 0, 1)
        self.lstm.flatten_parameters()
        encodings = self.lstm(encodings)[0]

        predictions = self.projection(encodings.permute(1, 2, 0))
        predictions = torch.nn.functional.log_softmax(predictions, dim= 1)

        return predictions
