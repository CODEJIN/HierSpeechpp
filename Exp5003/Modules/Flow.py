import torch
import logging, sys
from typing import Optional

from .LinearAttention import LinearAttention
from .Layer import Conv_Init, Mask_Generate

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout,
    format= '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
    )

class FlowBlock(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        calc_channels: int,
        flow_stack: int,
        flow_wavenet_conv_stack: int,
        flow_wavenet_kernel_size: int,
        flow_wavnet_dilation_rate: int,
        flow_wavenet_dropout_rate: float= 0.0,
        condition_channels: Optional[int]= None,
        ):
        super().__init__()

        self.flows = torch.nn.ModuleList()
        for _ in range(flow_stack):
            self.flows.append(Flow(
                channels= channels,
                calc_channels= calc_channels,
                wavenet_conv_stack= flow_wavenet_conv_stack,
                wavenet_kernel_size= flow_wavenet_kernel_size,
                wavnet_dilation_rate= flow_wavnet_dilation_rate,
                wavenet_dropout_rate= flow_wavenet_dropout_rate,
                condition_channels= condition_channels,
                ))
            self.flows.append(Flip())

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor]= None,
        conditions: Optional[torch.Tensor]= None,
        reverse: bool= False
        ):
        '''
        x: [Batch, Dim, Time]
        lengths: [Batch],
        conditions: [Batch, Cond_d], This may be a speaker or emotion embedding vector.
        reverse: a boolean
        '''
        masks = 1.0
        if not lengths is None:
            masks = (~Mask_Generate(lengths, max_length= x.size(2))).unsqueeze(1).float()

        if not conditions is None and conditions.ndim == 2:
            conditions = conditions.unsqueeze(2)    # [Batch, Cond_d, 1]
        
        for flow in (self.flows if not reverse else reversed(self.flows)):
            x = flow(
                x= x,
                masks= masks,
                conditions= conditions,
                reverse= reverse,
                )

        return x

class Flow(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        calc_channels: int,
        wavenet_conv_stack: int,
        wavenet_kernel_size: int,
        wavnet_dilation_rate: int,
        wavenet_dropout_rate: float= 0.0,
        condition_channels: Optional[int]= None,
        ):
        super().__init__()
        assert channels % 2 == 0, 'in_channels must be a even.'

        # Don't use init. This become a reason lower quality.
        self.prenet = torch.nn.Conv1d( 
            in_channels= channels // 2,
            out_channels= calc_channels,
            kernel_size= 1,
            )
        
        self.wavenet = WaveNet(
            calc_channels= calc_channels,
            conv_stack= wavenet_conv_stack,
            kernel_size= wavenet_kernel_size,
            dilation_rate= wavnet_dilation_rate,
            dropout_rate= wavenet_dropout_rate,
            condition_channels= condition_channels,
            )
        
        self.postnet = Conv_Init(torch.nn.Conv1d(
            in_channels= calc_channels,
            out_channels= channels // 2,
            kernel_size= 1,
            ), w_init_gain= 'zero')

    def forward(
        self,
        x: torch.Tensor,
        masks: torch.Tensor,
        conditions: Optional[torch.Tensor]= None,
        reverse: bool= False
        ):
        x_0, x_1 = x.chunk(chunks= 2, dim= 1)   # [Batch, Dim // 2, Time] * 2
        x_hiddens = self.prenet(x_0) * masks    # [Batch, Calc_d, Time]
        x_hiddens = self.wavenet(
            x= x_hiddens,
            masks= masks,
            conditions= conditions
            )   # [Batch, Calc_d, Time]
        means = self.postnet(x_hiddens) # [Batch, Dim // 2, Time]

        if not reverse:
            x_1 = (x_1 + means) * masks
        else:
            x_1 = (x_1 - means) * masks

        x = torch.cat([x_0, x_1], dim= 1)   # [Batch, Dim, Time]

        return x

class Flip(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        *args,
        **kwargs
        ):
        return x.flip(dims= [1,])

class WaveNet(torch.nn.Module):
    def __init__(
        self,
        calc_channels: int,
        conv_stack: int,
        kernel_size: int,
        dilation_rate: int,
        dropout_rate: float= 0.0,
        condition_channels: Optional[int]= None,
        ):
        super().__init__()
        self.calc_channels = calc_channels
        self.conv_stack = conv_stack
        self.use_condition = not condition_channels is None

        def weight_norm_initialize_weight(module):
            if 'Conv' in module.__class__.__name__:
                module.weight.data.normal_(0.0, 0.01)

        if self.use_condition:
            self.condition = torch.nn.utils.weight_norm(torch.nn.Conv1d(
                in_channels= condition_channels,
                out_channels= calc_channels * conv_stack * 2,
                kernel_size= 1
                ))
            self.condition.apply(weight_norm_initialize_weight)
        
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
        x: torch.Tensor,
        masks: torch.Tensor,
        conditions: Optional[torch.Tensor]= None,    
        ):
        if self.use_condition:
            conditions_list = self.condition(conditions).chunk(chunks= self.conv_stack, dim= 1)  # [Batch, Calc_d * 2, Time] * Stack
        else:
            conditions_list = [torch.zeros(
                size= (x.size(0), self.calc_channels * 2, x.size(2)),
                dtype= x.dtype,
                device= x.device
                )] * self.conv_stack

        skips_list = []
        for in_conv, conditions, residual_and_skip_conv in zip(self.input_convs, conditions_list, self.residual_and_skip_convs):
            ins = in_conv(x)
            acts = Fused_Gate(ins + conditions)
            acts = self.dropout(acts)
            residuals, skips = residual_and_skip_conv(acts).chunk(chunks= 2, dim= 1)
            x = (x + residuals) * masks
            skips_list.append(skips)

        skips = torch.stack(skips_list, dim= 1).sum(dim= 1) * masks

        return skips

@torch.jit.script
def Fused_Gate(x):
    x_tanh, x_sigmoid = x.chunk(chunks= 2, dim= 1)
    x = x_tanh.tanh() * x_sigmoid.sigmoid()

    return x

def Flow_KL_Loss(
    encoding_means: torch.FloatTensor,
    encoding_log_stds: torch.FloatTensor,
    flows: torch.FloatTensor,
    flow_log_stds: torch.FloatTensor,
    float_masks: Optional[torch.FloatTensor]= None
    ):
    encoding_means = encoding_means.float()
    encoding_log_stds = encoding_log_stds.float()
    flows = flows.float()
    flow_log_stds = flow_log_stds.float()    

    loss = encoding_log_stds - flow_log_stds - 0.5
    loss += 0.5 * (flows - encoding_means).pow(2.0) * (-2.0 * encoding_log_stds).exp()

    if float_masks is None:
        loss = loss.mean()
    else:
        float_masks = float_masks.float()
        loss = (loss * float_masks).sum() / float_masks.sum()
    
    return loss


class FlowBlock_Transformer(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        calc_channels: int,
        condition_channels: int,
        flow_stack: int,
        flow_ditblock_stack: int,
        flow_ditblock_num_heads: int,
        flow_ditblock_ffn_kernel_size: int,
        flow_ditblock_dropout_rate: float= 0.0,
        ):
        super().__init__()

        self.styles = torch.nn.Sequential(
            Conv_Init(torch.nn.Conv1d(
                in_channels= condition_channels,
                out_channels= channels * 4,
                kernel_size= 1
                ), w_init_gain= 'relu'),
            torch.nn.SiLU(),
            Conv_Init(torch.nn.Conv1d(
                in_channels= channels * 4,
                out_channels= channels,
                kernel_size= 1
                ), w_init_gain= 'linear')            
            )

        self.flows = torch.nn.ModuleList()
        for _ in range(flow_stack):
            self.flows.append(Flow_Transformer(
                channels= channels,
                calc_channels= calc_channels,
                num_ditblock= flow_ditblock_stack,
                ditblock_num_heads= flow_ditblock_num_heads,
                ditblock_ffn_kernel_size= flow_ditblock_ffn_kernel_size,
                ditblock_dropout_rate= flow_ditblock_dropout_rate,
                ))
            self.flows.append(Flip())

    def forward(
        self,
        x: torch.Tensor,
        styles: torch.FloatTensor,
        lengths: Optional[torch.Tensor]= None,
        reverse: bool= False
        ):
        '''
        x: [Batch, Dim, Time]
        lengths: [Batch],
        conditions: [Batch, Cond_d], This may be a speaker or emotion embedding vector.
        reverse: a boolean
        '''
        styles = self.styles(styles)    # [Batch, Dim, 1]
        
        for flow in (self.flows if not reverse else reversed(self.flows)):
            x = flow(
                x= x,
                styles= styles,
                lengths= lengths,
                reverse= reverse,
                )

        return x

class Flow_Transformer(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        calc_channels: int,
        num_ditblock: int,
        ditblock_num_heads: int,
        ditblock_ffn_kernel_size: int,
        ditblock_dropout_rate: float= 0.0
        ):
        super().__init__()
        assert channels % 2 == 0, 'in_channels must be a even.'

        self.prenet = torch.nn.Conv1d(
            in_channels= channels // 2,
            out_channels= calc_channels,
            kernel_size= 1,
            )   # Don't use init. This become a reason lower quality.
        
        self.blocks = torch.nn.ModuleList([
            DiTBlock(
                channels= calc_channels,
                num_heads= ditblock_num_heads,
                ffn_kernel_size= ditblock_ffn_kernel_size,
                dropout_rate= ditblock_dropout_rate
                )
            for _ in range(num_ditblock)
            ])
        
        self.postnet = Conv_Init(torch.nn.Conv1d(
            in_channels= calc_channels,
            out_channels= channels // 2,
            kernel_size= 1,
            ), w_init_gain= 'zero'
            )

    def forward(
        self,
        x: torch.FloatTensor,
        styles: torch.FloatTensor,
        lengths: Optional[torch.IntTensor],
        reverse: bool= False
        ):
        bool_masks= None
        float_masks = 1.0
        if not lengths is None:
            bool_masks = Mask_Generate(
                lengths= lengths,
                max_length= x.size(2)
                )
            float_masks = (~bool_masks)[:, None].float()

        x_0, x_1 = x.chunk(chunks= 2, dim= 1)   # [Batch, Dim // 2, Time] * 2
        x_hiddens = self.prenet(x_0) * float_masks    # [Batch, Calc_d, Time]

        for block in self.blocks:
            x = x_hiddens = block(
                x= x_hiddens,
                styles= styles,
                lengths= lengths
                )
            
        means = self.postnet(x_hiddens) * float_masks

        if not reverse:
            x_1 = (x_1 + means) * float_masks
        else:
            x_1 = (x_1 - means) * float_masks

        x = torch.cat([x_0, x_1], dim= 1)   # [Batch, Dim, Time]

        return x

class DiTBlock(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ffn_kernel_size: int,
        dropout_rate: float
        ):
        super().__init__()
        
        self.ada_ln_modulation = torch.nn.Sequential(
            torch.nn.SiLU(),
            Conv_Init(torch.nn.Conv1d(
                in_channels= channels,
                out_channels= channels * 6,
                kernel_size= 1
                ), w_init_gain= 'zero')
            )
        
        self.norm_0 = torch.nn.LayerNorm(channels)
        self.attention = LinearAttention(
            query_channels= channels,
            key_channels= channels,
            value_channels= channels,
            calc_channels= channels,
            num_heads= num_heads
            )
        
        self.norm_1 = torch.nn.LayerNorm(channels)
        self.ffn = FFN(
            channels= channels,
            calc_channels= channels * 4,
            kernel_size= ffn_kernel_size,
            dropout_rate= dropout_rate
            )
        
    def forward(
        self,
        x: torch.FloatTensor,
        styles: torch.FloatTensor,
        lengths: Optional[torch.IntTensor]= None
        ):
        bool_masks= None
        float_masks = 1.0
        if not lengths is None:
            bool_masks = Mask_Generate(
                lengths= lengths,
                max_length= x.size(2)
                )
            float_masks = (~bool_masks)[:, None].float()

        scales_0, shifts_0, gates_0, scales_1, shifts_1, gates_1 = \
            self.ada_ln_modulation(styles).chunk(chunks= 6, dim= 1)
        
        residuals = x
        x = self.norm_0(x.permute(0, 2, 1)).permute(0, 2, 1) * float_masks
        x = x * (1.0 + scales_0) + shifts_0
        x = self.attention.forward(
            queries= x,
            keys= x,
            values= x,
            key_padding_masks= bool_masks
            ) * float_masks
        x = x * gates_0 + residuals

        residuals = x
        x = self.norm_1(x.permute(0, 2, 1)).permute(0, 2, 1) * float_masks
        x = x * (1.0 + scales_1) + shifts_1
        x = self.ffn(x, float_masks) * float_masks
        x = x * gates_1 + residuals

        return x

class FFN(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        calc_channels: int,
        kernel_size: int,
        dropout_rate: float= 0.1,
        ) -> None:
        super().__init__()
        self.conv_0 = Conv_Init(torch.nn.Conv1d(
            in_channels= channels,
            out_channels= calc_channels,
            kernel_size= kernel_size,
            padding= (kernel_size - 1) // 2,
            ), w_init_gain= 'relu'
            )
        self.silu = torch.nn.GELU(approximate= 'tanh')
        self.dropout = torch.nn.Dropout(p= dropout_rate)
        self.conv_1 = Conv_Init(torch.nn.Conv1d(
            in_channels= calc_channels,
            out_channels= channels,
            kernel_size= kernel_size,
            padding= (kernel_size - 1) // 2,
            ), w_init_gain= 'linear')
        
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

        return x * float_masks

