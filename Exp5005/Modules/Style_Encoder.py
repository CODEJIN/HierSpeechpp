import torch
from typing import Optional, List, Dict, Tuple, Union

from .LinearAttention import LinearAttention
from .Layer import Conv_Init

class Style_Encoder(torch.nn.Module): 
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_kernel_size: int,
        num_heads: int,
        dropout_rate: float
        ):
        super().__init__()

        self.spectral_conv_0 = Conv_Init(torch.nn.Conv1d(
            in_channels= in_channels,
            out_channels= out_channels,
            kernel_size= 1
            ), w_init_gain= 'relu'
            )
        self.spectral_conv_1 = Conv_Init(torch.nn.Conv1d(
            in_channels= out_channels,
            out_channels= out_channels,
            kernel_size= 1
            ), w_init_gain= 'relu'
            )
        
        self.temporal_conv_0 = Conv_Init(torch.nn.Conv1d(
            in_channels= out_channels,
            out_channels= out_channels * 2,
            kernel_size= temporal_kernel_size,
            padding= (temporal_kernel_size - 1) // 2
            ), w_init_gain= 'glu'
            )
        self.temporal_conv_1 = Conv_Init(torch.nn.Conv1d(
            in_channels= out_channels,
            out_channels= out_channels * 2,
            kernel_size= temporal_kernel_size,
            padding= (temporal_kernel_size - 1) // 2
            ), w_init_gain= 'glu'
            )
        
        self.attention = LinearAttention(
            query_channels= out_channels,
            key_channels= out_channels,
            value_channels= out_channels,
            calc_channels= out_channels,
            num_heads= num_heads,
            )

        self.projection = Conv_Init(torch.nn.Conv1d(
            in_channels= out_channels,
            out_channels= out_channels,
            kernel_size= 1,
            ), w_init_gain= 'linear')
        
        self.mish = torch.nn.Mish()
        self.glu = torch.nn.GLU(dim= 1)
        self.dropout = torch.nn.Dropout(p= dropout_rate)

    def forward(
        self,
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
                max_length= styles.size(2)
                )
            float_masks = (~bool_masks)[:, None].float()

        styles = self.spectral_conv_0(styles * float_masks)
        styles = self.mish(styles * float_masks)
        styles = self.dropout(styles * float_masks)
        styles = self.spectral_conv_1(styles * float_masks)
        styles = self.mish(styles * float_masks)
        styles = self.dropout(styles * float_masks)

        residuals = styles
        styles = self.temporal_conv_0(styles * float_masks)
        styles = self.glu(styles * float_masks)
        styles = self.dropout(styles * float_masks)
        styles = styles + residuals
        residuals = styles
        styles = self.temporal_conv_1(styles * float_masks)
        styles = self.glu(styles * float_masks)
        styles = self.dropout(styles * float_masks)
        styles = styles + residuals

        styles = self.attention(
            queries= styles,
            keys= styles,
            values= styles,
            key_padding_masks= bool_masks
            )
        
        styles = self.projection(styles)

        if not lengths is None:
            styles = (styles * float_masks).sum(dim= 2) / lengths[:, None]
        else:
            styles = styles.mean(dim= 2)
            
        return styles

def Mask_Generate(lengths: torch.Tensor, max_length: Optional[Union[int, torch.Tensor]]= None):
    '''
    lengths: [Batch]
    max_lengths: an int value. If None, max_lengths == max(lengths)
    '''
    max_length = max_length or torch.max(lengths)
    sequence = torch.arange(max_length)[None, :].to(lengths.device)
    return sequence >= lengths[:, None]    # [Batch, Time]
