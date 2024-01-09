import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'    # This is ot prevent to be called Fortran Ctrl+C crash in Windows.
import torch
import numpy as np
import logging, yaml, os, sys, argparse, math, pickle, wandb
from functools import partial
from tqdm import tqdm
from collections import defaultdict
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.io import wavfile
from typing import List

from Exp5004.Modules.Text2Wav2Vec import Text2Wav2Vec

from Noam_Scheduler import Noam_Scheduler
from Exp5004.Datasets_Text2Wav2Vec import Dataset, Collater
from Logger import Logger

from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
from Arg_Parser import Recursive_Parse, To_Non_Recursive_Dict


import matplotlib as mpl
# 유니코드 깨짐현상 해결
mpl.rcParams['axes.unicode_minus'] = False
# 나눔고딕 폰트 적용
plt.rcParams["font.family"] = 'NanumGothic'

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout,
    format= '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
    )

# torch.autograd.set_detect_anomaly(True)

class Trainer:
    def __init__(self, hp_path, steps= 0):
        self.hp_path = hp_path
        self.gpu_id = int(os.getenv('RANK', '0'))
        self.num_gpus = int(os.getenv("WORLD_SIZE", '1'))
        
        self.hp = Recursive_Parse(yaml.load(
            open(self.hp_path, encoding='utf-8'),
            Loader=yaml.Loader
            ))

        if not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:{}'.format(self.gpu_id))
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = False
            torch.cuda.set_device(self.gpu_id)
        
        self.steps = steps

        self.Dataset_Generate()
        self.Model_Generate()
        self.Load_Checkpoint()
        self._Set_Distribution()

        self.scalar_dict = {
            'Train': defaultdict(float),
            'Evaluation': defaultdict(float),
            }

        if self.gpu_id == 0:
            self.writer_dict = {
                'Train': Logger(os.path.join(self.hp.Log_Path, 'Train')),
                'Evaluation': Logger(os.path.join(self.hp.Log_Path, 'Evaluation')),
                }

            if self.hp.Weights_and_Biases.Use:
                wandb.init(
                    project= self.hp.Weights_and_Biases.Project,
                    entity= self.hp.Weights_and_Biases.Entity,
                    name= self.hp.Weights_and_Biases.Name,
                    config= To_Non_Recursive_Dict(self.hp)
                    )
                wandb.watch(self.model_dict['Text2Wav2Vec'])

    def Dataset_Generate(self):
        token_dict = yaml.load(open(self.hp.Token_Path, 'r', encoding= 'utf-8-sig'), Loader=yaml.Loader)
        
        train_dataset = Dataset(
            token_dict= token_dict,
            pattern_path= self.hp.Train.Train_Pattern.Path,
            metadata_file= self.hp.Train.Train_Pattern.Metadata_File,
            text_length_min= self.hp.Train.Train_Pattern.Text_Length.Min,
            text_length_max= self.hp.Train.Train_Pattern.Text_Length.Max,
            audio_length_min= self.hp.Train.Train_Pattern.Audio_Length.Min,
            audio_length_max= self.hp.Train.Train_Pattern.Audio_Length.Max,
            accumulated_dataset_epoch= self.hp.Train.Train_Pattern.Accumulated_Dataset_Epoch,
            augmentation_ratio= self.hp.Train.Train_Pattern.Augmentation_Ratio,
            use_pattern_cache= self.hp.Train.Pattern_Cache
            )
        eval_dataset = Dataset(
            token_dict= token_dict,
            pattern_path= self.hp.Train.Eval_Pattern.Path,
            metadata_file= self.hp.Train.Eval_Pattern.Metadata_File,
            text_length_min= self.hp.Train.Eval_Pattern.Text_Length.Min,
            text_length_max= self.hp.Train.Eval_Pattern.Text_Length.Max,
            audio_length_min= self.hp.Train.Eval_Pattern.Audio_Length.Min,
            audio_length_max= self.hp.Train.Eval_Pattern.Audio_Length.Max,
            use_pattern_cache= self.hp.Train.Pattern_Cache
            )
        
        if self.gpu_id == 0:
            logging.info('The number of train patterns = {}.'.format(len(train_dataset) // self.hp.Train.Train_Pattern.Accumulated_Dataset_Epoch))
            logging.info('The number of development patterns = {}.'.format(len(eval_dataset)))

        collater = Collater(token_dict= token_dict)
        
        self.dataloader_dict = {}
        self.dataloader_dict['Train'] = torch.utils.data.DataLoader(
            dataset= train_dataset,
            sampler= torch.utils.data.DistributedSampler(train_dataset, shuffle= True) \
                     if self.hp.Use_Multi_GPU else \
                     torch.utils.data.RandomSampler(train_dataset),
            collate_fn= collater,
            batch_size= self.hp.Train.Batch_Size,
            num_workers= self.hp.Train.Num_Workers,
            pin_memory= True
            )
        self.dataloader_dict['Eval'] = torch.utils.data.DataLoader(
            dataset= eval_dataset,
            sampler= torch.utils.data.DistributedSampler(eval_dataset, shuffle= True) \
                     if self.num_gpus > 1 else \
                     torch.utils.data.RandomSampler(eval_dataset),
            collate_fn= collater,
            batch_size= self.hp.Train.Batch_Size,
            num_workers= self.hp.Train.Num_Workers,
            pin_memory= True,
            drop_last= True
            )
        
    def Model_Generate(self):
        self.model_dict = {
            'Text2Wav2Vec': Text2Wav2Vec(self.hp).to(self.device)
            }
        
        self.optimizer_dict = {
            'Text2Wav2Vec': torch.optim.AdamW(
                params= self.model_dict['Text2Wav2Vec'].parameters(),
                lr= self.hp.Train.Learning_Rate.Initial,
                betas=(self.hp.Train.ADAM.Beta1, self.hp.Train.ADAM.Beta2),
                eps= self.hp.Train.ADAM.Epsilon
                ),
            }
        self.scheduler_dict = {
            'Text2Wav2Vec': Noam_Scheduler(
                optimizer= self.optimizer_dict['Text2Wav2Vec'],
                warmup_steps= self.hp.Train.Learning_Rate.Warmup_Step
                ),
            }

        self.scaler = torch.cuda.amp.GradScaler(enabled= self.hp.Use_Mixed_Precision)

        # if self.gpu_id == 0:
        #     logging.info(self.model_dict['Text2Wav2Vec'])

    def Train_Step(
        self,
        tokens_with_between_pad: torch.IntTensor,
        token_lengths_with_between_pad: torch.IntTensor,
        audios: torch.FloatTensor,
        audio_lengths: torch.IntTensor,
        tokens_without_between_pad: torch.IntTensor,
        token_lengths_without_between_pad: torch.IntTensor,
        ):
        loss_dict = {}
        tokens_with_between_pad = tokens_with_between_pad.to(self.device, non_blocking=True)
        token_lengths_with_between_pad = token_lengths_with_between_pad.to(self.device, non_blocking=True)
        audios = audios.to(self.device, non_blocking=True)
        audio_lengths = audio_lengths.to(self.device, non_blocking=True)
        tokens_without_between_pad = tokens_without_between_pad.to(self.device, non_blocking=True)
        token_lengths_without_between_pad = token_lengths_without_between_pad.to(self.device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled= self.hp.Use_Mixed_Precision):
            content_loss, content_to_encoding_kld_loss, encoding_to_content_kld_loss, \
            duration_loss, f0_loss, token_ctc_loss = self.model_dict['Text2Wav2Vec'](
                tokens_with_between_pad= tokens_with_between_pad,
                token_lengths_with_between_pad= token_lengths_with_between_pad,
                audios= audios,
                audio_lengths= audio_lengths,
                tokens_without_between_pad= tokens_without_between_pad,
                token_lengths_without_between_pad= token_lengths_without_between_pad
                )

        loss_dict['Content'] = content_loss
        loss_dict['Content_to_Encoding_KLD'] = content_to_encoding_kld_loss
        loss_dict['Encoding_to_Content_KLD'] = encoding_to_content_kld_loss
        loss_dict['Duration'] = duration_loss
        loss_dict['F0'] = f0_loss
        loss_dict['Token'] = token_ctc_loss

        self.optimizer_dict['Text2Wav2Vec'].zero_grad()
        self.scaler.scale(
            loss_dict['Content'] +
            loss_dict['Content_to_Encoding_KLD'] +
            loss_dict['Encoding_to_Content_KLD'] +
            loss_dict['Duration'] +
            loss_dict['F0'] +
            loss_dict['Token']
            ).backward()

        self.scaler.unscale_(self.optimizer_dict['Text2Wav2Vec'])

        if self.hp.Train.Gradient_Norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                parameters= self.model_dict['Text2Wav2Vec'].parameters(),
                max_norm= self.hp.Train.Gradient_Norm
                )

        self.scaler.step(self.optimizer_dict['Text2Wav2Vec'])
        self.scaler.update()
        self.scheduler_dict['Text2Wav2Vec'].step()

        self.steps += 1
        self.tqdm.update(1)

        for tag, loss in loss_dict.items():
            loss = reduce_tensor(loss.data, self.num_gpus).item() if self.num_gpus > 1 else loss.item()
            self.scalar_dict['Train']['Loss/{}'.format(tag)] += loss

    def Train_Epoch(self):
        for (
            tokens_with_between_pad,
            token_lengths_with_between_pad,
            audios,
            audio_lengths,
            tokens_without_between_pad,
            token_lengths_without_between_pad
            ) in self.dataloader_dict['Train']:
            self.Train_Step(
                tokens_with_between_pad= tokens_with_between_pad,
                token_lengths_with_between_pad= token_lengths_with_between_pad,
                audios= audios,
                audio_lengths= audio_lengths,
                tokens_without_between_pad= tokens_without_between_pad,
                token_lengths_without_between_pad= token_lengths_without_between_pad
                )

            if self.steps % self.hp.Train.Checkpoint_Save_Interval == 0:
                self.Save_Checkpoint()

            if self.steps % self.hp.Train.Logging_Interval == 0 and self.gpu_id == 0:
                self.scalar_dict['Train'] = {
                    tag: loss / self.hp.Train.Logging_Interval
                    for tag, loss in self.scalar_dict['Train'].items()
                    }
                self.scalar_dict['Train']['Learning_Rate'] = self.scheduler_dict['Text2Wav2Vec'].get_last_lr()[0]
                self.writer_dict['Train'].add_scalar_dict(self.scalar_dict['Train'], self.steps)
                if self.hp.Weights_and_Biases.Use:
                    wandb.log(
                        data= {
                            f'Train.{key}': value
                            for key, value in self.scalar_dict['Train'].items()
                            },
                        step= self.steps,
                        commit= self.steps % self.hp.Train.Evaluation_Interval != 0
                        )
                self.scalar_dict['Train'] = defaultdict(float)

            if self.steps % self.hp.Train.Evaluation_Interval == 0:
                self.Evaluation_Epoch()

            if self.steps >= self.hp.Train.Max_Step:
                return

    def Evaluation_Step(
        self,
        tokens_with_between_pad: torch.IntTensor,
        token_lengths_with_between_pad: torch.IntTensor,
        audios: torch.FloatTensor,
        audio_lengths: torch.IntTensor,
        tokens_without_between_pad: torch.IntTensor,
        token_lengths_without_between_pad: torch.IntTensor,
        ):
        loss_dict = {}
        tokens_with_between_pad = tokens_with_between_pad.to(self.device, non_blocking=True)
        token_lengths_with_between_pad = token_lengths_with_between_pad.to(self.device, non_blocking=True)
        audios = audios.to(self.device, non_blocking=True)
        audio_lengths = audio_lengths.to(self.device, non_blocking=True)
        tokens_without_between_pad = tokens_without_between_pad.to(self.device, non_blocking=True)
        token_lengths_without_between_pad = token_lengths_without_between_pad.to(self.device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled= self.hp.Use_Mixed_Precision):
            content_loss, content_to_encoding_kld_loss, encoding_to_content_kld_loss, \
            duration_loss, f0_loss, token_ctc_loss = self.model_dict['Text2Wav2Vec'](
                tokens_with_between_pad= tokens_with_between_pad,
                token_lengths_with_between_pad= token_lengths_with_between_pad,
                audios= audios,
                audio_lengths= audio_lengths,
                tokens_without_between_pad= tokens_without_between_pad,
                token_lengths_without_between_pad= token_lengths_without_between_pad
                )

        loss_dict['Content'] = content_loss
        loss_dict['Content_to_Encoding_KLD'] = content_to_encoding_kld_loss
        loss_dict['Encoding_to_Content_KLD'] = encoding_to_content_kld_loss
        loss_dict['Duration'] = duration_loss
        loss_dict['F0'] = f0_loss
        loss_dict['Token'] = token_ctc_loss

        for tag, loss in loss_dict.items():
            loss = reduce_tensor(loss.data, self.num_gpus).item() if self.num_gpus > 1 else loss.item()
            self.scalar_dict['Evaluation']['Loss/{}'.format(tag)] += loss

    def Evaluation_Epoch(self):
        logging.info('(Steps: {}) Start evaluation in GPU {}.'.format(self.steps, self.gpu_id))

        for model in self.model_dict.values():
            model.eval()

        for step, (
            tokens_with_between_pad,
            token_lengths_with_between_pad,
            audios,
            audio_lengths,
            tokens_without_between_pad,
            token_lengths_without_between_pad
            ) in tqdm(
            enumerate(self.dataloader_dict['Eval'], 1),
            desc='[Evaluation]',
            total= math.ceil(len(self.dataloader_dict['Eval'].dataset) / self.hp.Train.Batch_Size / self.num_gpus)
            ):
            self.Evaluation_Step(
                tokens_with_between_pad= tokens_with_between_pad,
                token_lengths_with_between_pad= token_lengths_with_between_pad,
                audios= audios,
                audio_lengths= audio_lengths,
                tokens_without_between_pad= tokens_without_between_pad,
                token_lengths_without_between_pad= token_lengths_without_between_pad
                )

        if self.gpu_id == 0:
            self.scalar_dict['Evaluation'] = {
                tag: loss / step
                for tag, loss in self.scalar_dict['Evaluation'].items()
                }
            self.writer_dict['Evaluation'].add_scalar_dict(self.scalar_dict['Evaluation'], self.steps)
            # self.writer_dict['Evaluation'].add_histogram_model(self.model_dict['Text2Wav2Vec'], 'Text2Wav2Vec', self.steps, delete_keywords=[])

            index = np.random.randint(0, tokens_with_between_pad.size(0))
            with torch.inference_mode():
                _, prediction_f0s, prediction_alignments = self.model_dict['Text2Wav2Vec'].Inference(
                    tokens= tokens_with_between_pad[index, None].to(self.device),
                    token_lengths= token_lengths_with_between_pad[index, None].to(self.device),
                    audios= audios[index, None].to(self.device),
                    audio_lengths= audio_lengths[index, None].to(self.device),
                    )
                
            token_length = token_lengths_with_between_pad[index]
            feature_length = prediction_alignments[0, :, :token_length -1].sum().long()
            f0_length = feature_length * (self.hp.Sound.Hop_Size // self.hp.Sound.F0_Hop_Size)

            prediction_f0 = prediction_f0s[0, :f0_length].cpu().numpy()
            prediction_alignment = prediction_alignments[0, :feature_length, :token_length - 1].T.cpu().numpy()

            image_dict = {
                'Duration': (prediction_alignment, None, 'auto', None, None, None),
                'F0': (prediction_f0, None, 'auto', None, None, None),
                }
            self.writer_dict['Evaluation'].add_image_dict(image_dict, self.steps)

            if self.hp.Weights_and_Biases.Use:
                wandb.log(
                    data= {
                        f'Evaluation.{key}': value
                        for key, value in self.scalar_dict['Evaluation'].items()
                        },
                    step= self.steps,
                    commit= False
                    )
                wandb.log(
                    data= {
                        'Evaluation.Alignment': wandb.Image(prediction_alignment),
                        'Evaluation.F0': wandb.plot.line_series(
                            xs= np.arange(max(f0_length)),
                            ys= [prediction_f0],
                            keys= ['Prediction'],
                            title= 'F0',
                            xname= 'F0_t'
                            ),
                        },
                    step= self.steps,
                    commit= True
                    )

        self.scalar_dict['Evaluation'] = defaultdict(float)

        for model in self.model_dict.values():
            model.train()

    def Load_Checkpoint(self):
        if self.steps == 0:
            paths = [
                os.path.join(root, file).replace('\\', '/')
                for root, _, files in os.walk(self.hp.Checkpoint_Path)
                for file in files
                if os.path.splitext(file)[1] == '.pt'
                ]
            if len(paths) > 0:
                path = max(paths, key = os.path.getctime)
            else:
                return  # Initial training
        else:
            path = os.path.join(self.hp.Checkpoint_Path, 'S_{}.pt'.format(self.steps).replace('\\', '/'))

        state_dict = torch.load(path, map_location= 'cpu')
        self.model_dict['Text2Wav2Vec'].load_state_dict(state_dict['Model']['Text2Wav2Vec'])
        self.optimizer_dict['Text2Wav2Vec'].load_state_dict(state_dict['Optimizer']['Text2Wav2Vec'])
        self.scheduler_dict['Text2Wav2Vec'].load_state_dict(state_dict['Scheduler']['Text2Wav2Vec'])
        self.steps = state_dict['Steps']

        logging.info('Checkpoint loaded at {} steps in GPU {}.'.format(self.steps, self.gpu_id))

    def Save_Checkpoint(self):
        if self.gpu_id != 0:
            return

        os.makedirs(self.hp.Checkpoint_Path, exist_ok= True)
        state_dict = {
            'Model': {
                'Text2Wav2Vec': self.model_dict['Text2Wav2Vec'].state_dict(),
                },
            'Optimizer': {
                'Text2Wav2Vec': self.optimizer_dict['Text2Wav2Vec'].state_dict(),
                },
            'Scheduler': {
                'Text2Wav2Vec': self.scheduler_dict['Text2Wav2Vec'].state_dict(),
                },
            'Steps': self.steps
            }
        checkpoint_path = os.path.join(self.hp.Checkpoint_Path, 'S_{}.pt'.format(self.steps).replace('\\', '/'))

        torch.save(state_dict, checkpoint_path)

        logging.info('Checkpoint saved at {} steps.'.format(self.steps))

        if all([
            self.hp.Weights_and_Biases.Use,
            self.hp.Weights_and_Biases.Save_Checkpoint.Use,
            self.steps % self.hp.Weights_and_Biases.Save_Checkpoint.Interval == 0
            ]):
            wandb.save(checkpoint_path)

    def _Set_Distribution(self):
        if self.num_gpus > 1:
            self.model = apply_gradient_allreduce(self.model)

    def Train(self):
        hp_path = os.path.join(self.hp.Checkpoint_Path, 'Hyper_Parameters.yaml').replace('\\', '/')
        if not os.path.exists(hp_path):
            from shutil import copyfile
            os.makedirs(self.hp.Checkpoint_Path, exist_ok= True)
            copyfile(self.hp_path, hp_path)

        if self.steps == 0:
            self.Evaluation_Epoch()

        self.tqdm = tqdm(
            initial= self.steps,
            total= self.hp.Train.Max_Step,
            desc='[Training]'
            )

        while self.steps < self.hp.Train.Max_Step:
            try:
                self.Train_Epoch()
            except KeyboardInterrupt:
                self.Save_Checkpoint()
                exit(1)

        self.tqdm.close()
        logging.info('Finished training.')

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-hp', '--hyper_parameters', required= True, type= str)
    argParser.add_argument('-s', '--steps', default= 0, type= int)
    argParser.add_argument('-r', '--local_rank', default= 0, type= int)
    args = argParser.parse_args()
    
    hp = Recursive_Parse(yaml.load(
        open(args.hyper_parameters, encoding='utf-8'),
        Loader=yaml.Loader
        ))
    os.environ['CUDA_VISIBLE_DEVICES'] = hp.Device

    if hp.Use_Multi_GPU:
        init_distributed(
            rank= int(os.getenv('RANK', '0')),
            num_gpus= int(os.getenv("WORLD_SIZE", '1')),
            dist_backend= 'nccl'
            )
    new_Trainer = Trainer(hp_path= args.hyper_parameters, steps= args.steps)
    new_Trainer.Train()