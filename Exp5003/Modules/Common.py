import torch
from typing import Optional, Union

from .Layer import Conv_Init
from .LinearAttention import LinearAttention

class WaveNet(torch.nn.Module):
    def __init__(
        self,
        calc_channels: int,
        style_channels: int,
        conv_stack: int,
        kernel_size: int,
        dilation_rate: int,
        dropout_rate: float= 0.0,
        ):
        super().__init__()
        self.calc_channels = calc_channels
        self.conv_stack = conv_stack

        def weight_norm_initialize_weight(module):
            if 'Conv' in module.__class__.__name__:
                module.weight.data.normal_(0.0, 0.01)

        self.style = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            in_channels= style_channels,
            out_channels= calc_channels * conv_stack * 2,
            kernel_size= 1
            ))
        self.style.apply(weight_norm_initialize_weight)
        
        self.input_convs = torch.nn.ModuleList()
        self.residual_and_skip_convs = torch.nn.ModuleList()
        for index in range(conv_stack):
            dilation = dilation_rate ** index
            padding = (kernel_size - 1) * dilation // 2
            self.input_convs.append(torch.nn.utils.weight_norm(torch.nn.Conv1d(
                in_channels= calc_channels,
                out_channels= calc_channels * 2,
                kernel_size= kernel_size,
                dilation= dilation,
                padding= padding
                )))
            self.residual_and_skip_convs.append(torch.nn.utils.weight_norm(torch.nn.Conv1d(
                in_channels= calc_channels,
                out_channels= calc_channels * 2,
                kernel_size= 1
                )))

        self.dropout = torch.nn.Dropout(p= dropout_rate)
        
        self.input_convs.apply(weight_norm_initialize_weight)
        self.residual_and_skip_convs.apply(weight_norm_initialize_weight)

    def forward(
        self,
        x: torch.FloatTensor,
        conditions: torch.FloatTensor,
        float_masks: torch.Tensor,
        ):
        conditions_list = self.style(conditions).chunk(chunks= self.conv_stack, dim= 1)  # [Batch, Calc_d * 2, Time] * Stack

        skips_list = []
        for in_conv, conditions, residual_and_skip_conv in zip(self.input_convs, conditions_list, self.residual_and_skip_convs):
            ins = in_conv(x)
            acts = Fused_Gate(ins + conditions)
            acts = self.dropout(acts)
            residuals, skips = residual_and_skip_conv(acts).chunk(chunks= 2, dim= 1)
            x = (x + residuals) * float_masks
            skips_list.append(skips)

        skips = torch.stack(skips_list, dim= 1).sum(dim= 1) * float_masks

        return skips

# @torch.compile not yet supported windows
@torch.jit.script
def Fused_Gate(x):
    x_tanh, x_sigmoid = x.chunk(chunks= 2, dim= 1)
    x = x_tanh.tanh() * x_sigmoid.sigmoid()

    return x


class FFT_Block(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        num_head: int,
        ffn_kernel_size: int,
        dropout_rate: float
        ) -> None:
        super().__init__()

        self.attention = LinearAttention(
            query_channels= channels,
            key_channels= channels,
            value_channels= channels,
            calc_channels= channels,
            num_heads= num_head,
            dropout_rate= dropout_rate
            )
        
        self.ffn = FFN(
            channels= channels,
            kernel_size= ffn_kernel_size,
            dropout_rate= dropout_rate
            )
        
    def forward(
        self,
        x: torch.FloatTensor,
        lengths: Optional[torch.IntTensor]= None
        ) -> torch.Tensor:
        '''
        x: [Batch, Dim, Time]
        '''
        bool_masks= None
        float_masks = 1.0
        if not lengths is None:
            bool_masks = Mask_Generate(
                lengths= lengths,
                max_length= x.size(2)
                )
            float_masks = (~bool_masks)[:, None].float()

        # Attention + Dropout + Norm
        x = self.attention(
            queries= x,
            keys= x,
            values= x,
            key_padding_masks= bool_masks
            )
        
        # FFN + Dropout + Norm
        x = self.ffn(x, float_masks)

        return x * float_masks

class FFN(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dropout_rate: float= 0.1,
        ) -> None:
        super().__init__()
        self.conv_0 = Conv_Init(torch.nn.Conv1d(
            in_channels= channels,
            out_channels= channels,
            kernel_size= kernel_size,
            padding= (kernel_size - 1) // 2,
            ), w_init_gain= 'relu'
            )
        self.silu = torch.nn.SiLU()
        self.dropout = torch.nn.Dropout(p= dropout_rate)
        self.conv_1 = Conv_Init(torch.nn.Conv1d(
            in_channels= channels,
            out_channels= channels,
            kernel_size= kernel_size,
            padding= (kernel_size - 1) // 2,
            ), w_init_gain= 'linear')
        self.norm = torch.nn.LayerNorm(channels)
        
    def forward(
        self,
        x: torch.Tensor,
        float_masks: torch.Tensor
        ) -> torch.Tensor:
        '''
        x: [Batch, Dim, Time]
        '''
        residuals = x

        x = self.conv_0(x * float_masks)
        x = self.silu(x)
        x = self.dropout(x)
        x = self.conv_1(x * float_masks)
        x = self.dropout(x)
        x = self.norm((x + residuals).permute(0, 2, 1)).permute(0, 2, 1)

        return x * float_masks


def Mask_Generate(lengths: torch.Tensor, max_length: Optional[Union[int, torch.Tensor]]= None):
    '''
    lengths: [Batch]
    max_lengths: an int value. If None, max_lengths == max(lengths)
    '''
    max_length = max_length or torch.max(lengths)
    sequence = torch.arange(max_length)[None, :].to(lengths.device)
    return sequence >= lengths[:, None]    # [Batch, Time]
