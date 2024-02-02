import torch
import transformers

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
