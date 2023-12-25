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

from Exp5003.Modules.Modules import Synthesizer
from Exp5003.Modules.Discriminator import Discriminator, R1_Regulator, Feature_Map_Loss, Generator_Loss, Discriminator_Loss

from Noam_Scheduler import Noam_Scheduler
from Exp5003.Datasets import Dataset, Inference_Dataset, Collater, Inference_Collater
from Logger import Logger

from meldataset import mel_spectrogram
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
                wandb.watch(self.model_dict['Synthesizer'])

    def Dataset_Generate(self):
        token_dict = yaml.load(open(self.hp.Token_Path, 'r', encoding= 'utf-8-sig'), Loader=yaml.Loader)
        
        train_dataset = Dataset(
            pattern_path= self.hp.Train.Train_Pattern.Path,
            metadata_file= self.hp.Train.Train_Pattern.Metadata_File,
            audio_length_min= max(
                self.hp.Train.Train_Pattern.Audio_Length.Min,
                self.hp.Train.Segment_Size * self.hp.Sound.Hop_Size
                ),
            audio_length_max= self.hp.Train.Train_Pattern.Audio_Length.Max,
            accumulated_dataset_epoch= self.hp.Train.Train_Pattern.Accumulated_Dataset_Epoch,
            augmentation_ratio= self.hp.Train.Train_Pattern.Augmentation_Ratio,
            use_pattern_cache= self.hp.Train.Pattern_Cache
            )
        eval_dataset = Dataset(
            pattern_path= self.hp.Train.Eval_Pattern.Path,
            metadata_file= self.hp.Train.Eval_Pattern.Metadata_File,
            audio_length_min= max(
                self.hp.Train.Eval_Pattern.Audio_Length.Min,
                self.hp.Train.Segment_Size * self.hp.Sound.Hop_Size
                ),
            audio_length_max= self.hp.Train.Eval_Pattern.Audio_Length.Max,
            use_pattern_cache= self.hp.Train.Pattern_Cache
            )
        inference_dataset = Inference_Dataset(
            source_audio_paths= self.hp.Train.Inference_in_Train.Source_Audio_Path,
            sample_rate= self.hp.Sound.Sample_Rate,
            hop_size= self.hp.Sound.Hop_Size
            )

        if self.gpu_id == 0:
            logging.info('The number of train patterns = {}.'.format(len(train_dataset) // self.hp.Train.Train_Pattern.Accumulated_Dataset_Epoch))
            logging.info('The number of development patterns = {}.'.format(len(eval_dataset)))
            logging.info('The number of inference patterns = {}.'.format(len(inference_dataset)))

        collater = Collater()
        inference_collater = Inference_Collater()

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
            pin_memory= True
            )
        self.dataloader_dict['Inference'] = torch.utils.data.DataLoader(
            dataset= inference_dataset,
            sampler= torch.utils.data.SequentialSampler(inference_dataset),
            collate_fn= inference_collater,
            batch_size= self.hp.Inference_Batch_Size or self.hp.Train.Batch_Size,
            num_workers= self.hp.Train.Num_Workers,
            pin_memory= True
            )

    def Model_Generate(self):
        self.model_dict = {
            'Synthesizer': Synthesizer(self.hp).to(self.device),
            'Discriminator': Discriminator(
                use_stft_discriminator= self.hp.Discriminator.Use_STFT,
                period_list= self.hp.Discriminator.Period,
                stft_n_fft_list= self.hp.Discriminator.STFT_N_FFT,
                scale_pool_kernel_size_list= self.hp.Discriminator.Scale_Pool_Kernel_Size,
                ).to(self.device),
            }
        
        self.mel_func = partial(
            mel_spectrogram,
            n_fft= self.hp.Sound.N_FFT,
            num_mels= self.hp.Sound.N_Mel,
            sampling_rate= self.hp.Sound.Sample_Rate,
            hop_size= self.hp.Sound.Hop_Size,
            win_size= self.hp.Sound.Window_Size,
            fmin= 0,
            fmax= None
            )

        self.criterion_dict = {
            'MSE': torch.nn.MSELoss(reduce= None).to(self.device),
            'MAE': torch.nn.L1Loss(reduce= None).to(self.device),
            'TokenCTC': torch.nn.CTCLoss(
                blank= self.hp.Tokens,  # == Token length
                zero_infinity=True
                ),
            'R1': R1_Regulator()
            }
        self.optimizer_dict = {
            'Synthesizer': torch.optim.AdamW(
                params= self.model_dict['Synthesizer'].parameters(),
                lr= self.hp.Train.Learning_Rate.Initial,
                betas=(self.hp.Train.ADAM.Beta1, self.hp.Train.ADAM.Beta2),
                eps= self.hp.Train.ADAM.Epsilon
                ),
            'Discriminator': torch.optim.AdamW(
                params= self.model_dict['Discriminator'].parameters(),
                lr= self.hp.Train.Learning_Rate.Initial,
                betas=(self.hp.Train.ADAM.Beta1, self.hp.Train.ADAM.Beta2),
                eps= self.hp.Train.ADAM.Epsilon
                ),
            }
        self.scheduler_dict = {
            'Synthesizer': Noam_Scheduler(
                optimizer= self.optimizer_dict['Synthesizer'],
                warmup_steps= self.hp.Train.Learning_Rate.Warmup_Step
                ),
            'Discriminator': Noam_Scheduler(
                optimizer= self.optimizer_dict['Discriminator'],
                warmup_steps= self.hp.Train.Learning_Rate.Warmup_Step
                ),
            }

        self.scaler = torch.cuda.amp.GradScaler(enabled= self.hp.Use_Mixed_Precision)

        # if self.gpu_id == 0:
        #     logging.info(self.model_dict['Synthesizer'])

    def Train_Step(self, audios, audio_lengths):
        loss_dict = {}
        audios = audios.to(self.device, non_blocking=True)
        audio_lengths = audio_lengths.to(self.device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled= self.hp.Use_Mixed_Precision):
            prediction_audios_slice, audios_slice, \
            stft_loss, semantic_f0_loss, prosody_loss, \
            encoding_perturbed_to_clean_kld_loss, encoding_clean_to_perturbed_kld_loss, \
            encoding_clean_to_acoustic_kld_loss, acoustic_to_encoding_clean_kld_loss = self.model_dict['Synthesizer'](
                audios= audios,
                audio_lengths= audio_lengths
                )
            
            audios_slice.requires_grad_()
            discriminations_list_for_real, _ = self.model_dict['Discriminator'](audios_slice,)
            discriminations_list_for_fake, _ = self.model_dict['Discriminator'](prediction_audios_slice.detach())
            with torch.cuda.amp.autocast(enabled= False):
                loss_dict['Discrimination'] = Discriminator_Loss(discriminations_list_for_real, discriminations_list_for_fake)
                loss_dict['R1'] = self.criterion_dict['R1'](discriminations_list_for_real, audios_slice)

        self.optimizer_dict['Discriminator'].zero_grad()

        self.scaler.scale(loss_dict['Discrimination'] + loss_dict['R1']).backward()
        self.scaler.unscale_(self.optimizer_dict['Discriminator'])
        if self.hp.Train.Gradient_Norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                parameters= self.model_dict['Discriminator'].parameters(),
                max_norm= self.hp.Train.Gradient_Norm
                )

        self.scaler.step(self.optimizer_dict['Discriminator'])
        self.scaler.update()
        self.scheduler_dict['Discriminator'].step()

        with torch.cuda.amp.autocast(enabled= self.hp.Use_Mixed_Precision):
            discriminations_list_for_real, feature_maps_list_for_real = self.model_dict['Discriminator'](audios_slice,)
            discriminations_list_for_fake, feature_maps_list_for_fake = self.model_dict['Discriminator'](prediction_audios_slice)
            with torch.cuda.amp.autocast(enabled= False):
                loss_dict['STFT'] = stft_loss
                loss_dict['Semantic_F0'] = semantic_f0_loss
                loss_dict['Prosody'] = prosody_loss
                loss_dict['Encoding_Perturbed_to_Clean_KLD'] = encoding_perturbed_to_clean_kld_loss
                loss_dict['Encoding_Clean_to_Perturbed_KLD'] = encoding_clean_to_perturbed_kld_loss
                loss_dict['Encoding_Clean_to_Acoustic_KLD'] = encoding_clean_to_acoustic_kld_loss
                loss_dict['Acoustic_to_Encoding_Clean_KLD'] = acoustic_to_encoding_clean_kld_loss
                loss_dict['Adversarial'] = Generator_Loss(discriminations_list_for_fake)
                loss_dict['Feature_Map'] = Feature_Map_Loss(feature_maps_list_for_real, feature_maps_list_for_fake)

        self.optimizer_dict['Synthesizer'].zero_grad()
        self.scaler.scale(
            loss_dict['STFT'] * self.hp.Train.Learning_Rate.Lambda.STFT +
            loss_dict['Semantic_F0'] +
            loss_dict['Prosody'] +
            loss_dict['Encoding_Perturbed_to_Clean_KLD'] +
            loss_dict['Encoding_Clean_to_Perturbed_KLD'] +
            loss_dict['Encoding_Clean_to_Acoustic_KLD'] +
            loss_dict['Acoustic_to_Encoding_Clean_KLD'] +
            loss_dict['Adversarial'] +
            loss_dict['Feature_Map'] * self.hp.Train.Learning_Rate.Lambda.Feature_Map
            ).backward()

        self.scaler.unscale_(self.optimizer_dict['Synthesizer'])

        if self.hp.Train.Gradient_Norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                parameters= self.model_dict['Synthesizer'].parameters(),
                max_norm= self.hp.Train.Gradient_Norm
                )

        self.scaler.step(self.optimizer_dict['Synthesizer'])
        self.scaler.update()
        self.scheduler_dict['Synthesizer'].step()

        self.steps += 1
        self.tqdm.update(1)

        for tag, loss in loss_dict.items():
            loss = reduce_tensor(loss.data, self.num_gpus).item() if self.num_gpus > 1 else loss.item()
            self.scalar_dict['Train']['Loss/{}'.format(tag)] += loss

    def Train_Epoch(self):
        for audios, audio_lengths in self.dataloader_dict['Train']:
            self.Train_Step(
                audios= audios,
                audio_lengths= audio_lengths
                )

            if self.steps % self.hp.Train.Checkpoint_Save_Interval == 0:
                self.Save_Checkpoint()

            if self.steps % self.hp.Train.Logging_Interval == 0 and self.gpu_id == 0:
                self.scalar_dict['Train'] = {
                    tag: loss / self.hp.Train.Logging_Interval
                    for tag, loss in self.scalar_dict['Train'].items()
                    }
                self.scalar_dict['Train']['Learning_Rate'] = self.scheduler_dict['Synthesizer'].get_last_lr()[0]
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

            if self.steps % self.hp.Train.Inference_Interval == 0:
                self.Inference_Epoch()
            
            if self.steps >= self.hp.Train.Max_Step:
                return

    def Evaluation_Step(self, audios, audio_lengths):
        loss_dict = {}
        audios = audios.to(self.device, non_blocking=True)
        audio_lengths = audio_lengths.to(self.device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled= self.hp.Use_Mixed_Precision):
            prediction_audios_slice, audios_slice, \
            stft_loss, semantic_f0_loss, prosody_loss, \
            encoding_perturbed_to_clean_kld_loss, encoding_clean_to_perturbed_kld_loss, \
            encoding_clean_to_acoustic_kld_loss, acoustic_to_encoding_clean_kld_loss = self.model_dict['Synthesizer'](
                audios= audios,
                audio_lengths= audio_lengths
                )

            audios_slice.requires_grad_() # to calculate gradient penalty.
            discriminations_list_for_real, feature_maps_list_for_real = self.model_dict['Discriminator'](audios_slice)
            discriminations_list_for_fake, feature_maps_list_for_fake = self.model_dict['Discriminator'](prediction_audios_slice)
            with torch.cuda.amp.autocast(enabled= False):
                loss_dict['STFT'] = stft_loss
                loss_dict['Semantic_F0'] = semantic_f0_loss
                loss_dict['Prosody'] = prosody_loss
                loss_dict['Encoding_Perturbed_to_Clean_KLD'] = encoding_perturbed_to_clean_kld_loss
                loss_dict['Encoding_Clean_to_Perturbed_KLD'] = encoding_clean_to_perturbed_kld_loss
                loss_dict['Encoding_Clean_to_Acoustic_KLD'] = encoding_clean_to_acoustic_kld_loss
                loss_dict['Acoustic_to_Encoding_Clean_KLD'] = acoustic_to_encoding_clean_kld_loss
                loss_dict['Discrimination'] = Discriminator_Loss(discriminations_list_for_real, discriminations_list_for_fake)
                loss_dict['R1'] = self.criterion_dict['R1'](discriminations_list_for_real, audios_slice)
                loss_dict['Adversarial'] = Generator_Loss(discriminations_list_for_fake)
                loss_dict['Feature_Map'] = Feature_Map_Loss(feature_maps_list_for_real, feature_maps_list_for_fake)

        for tag, loss in loss_dict.items():
            loss = reduce_tensor(loss.data, self.num_gpus).item() if self.num_gpus > 1 else loss.item()
            self.scalar_dict['Evaluation']['Loss/{}'.format(tag)] += loss

    def Evaluation_Epoch(self):
        logging.info('(Steps: {}) Start evaluation in GPU {}.'.format(self.steps, self.gpu_id))

        for model in self.model_dict.values():
            model.eval()

        for step, (audios, audio_lengths) in tqdm(
            enumerate(self.dataloader_dict['Eval'], 1),
            desc='[Evaluation]',
            total= math.ceil(len(self.dataloader_dict['Eval'].dataset) / self.hp.Train.Batch_Size / self.num_gpus)
            ):
            self.Evaluation_Step(
                audios= audios,
                audio_lengths= audio_lengths
                )

        if self.gpu_id == 0:
            self.scalar_dict['Evaluation'] = {
                tag: loss / step
                for tag, loss in self.scalar_dict['Evaluation'].items()
                }
            self.writer_dict['Evaluation'].add_scalar_dict(self.scalar_dict['Evaluation'], self.steps)
            # self.writer_dict['Evaluation'].add_histogram_model(self.model_dict['Synthesizer'], 'Synthesizer', self.steps, delete_keywords=[])
            # self.writer_dict['Evaluation'].add_histogram_model(self.model_dict['Discriminator'], 'Discriminator', self.steps, delete_keywords=[])
        
            index = np.random.randint(0, audios.size(0))

            with torch.inference_mode():
                prediction_audios = self.model_dict['Synthesizer'].Inference(
                    source_audios= audios[index].unsqueeze(0).to(self.device),
                    source_audio_lengths= audio_lengths[index].unsqueeze(0).to(self.device),
                    )
            
            audio_length = audio_lengths[index]
            spectrogram_length = audio_length // self.hp.Sound.Hop_Size

            target_audio = audios[index, :audio_length]
            prediction_audio = prediction_audios[0, :audio_length]

            mel_func = partial(
                mel_spectrogram,
                n_fft= self.hp.Sound.N_FFT,
                num_mels= self.hp.Sound.N_Mel,
                sampling_rate= self.hp.Sound.Sample_Rate,
                hop_size= self.hp.Sound.Hop_Size,
                win_size= self.hp.Sound.Window_Size,
                fmin= 0,
                fmax= None
                )

            target_mel = mel_func(target_audio[None])[0, :, :spectrogram_length].cpu().numpy()
            prediction_mel = mel_func(prediction_audio[None])[0, :, :spectrogram_length].cpu().numpy()
            
            target_audio = target_audio.cpu().numpy()
            prediction_audio = prediction_audio.cpu().numpy()

            image_dict = {
                'Mel/Target': (target_mel, None, 'auto', None, None, None),
                'Mel/Prediction': (prediction_mel, None, 'auto', None, None, None),
                }
            audio_dict = {
                'Audio/Target': (target_audio, self.hp.Sound.Sample_Rate),
                'Audio/Linear': (prediction_audio, self.hp.Sound.Sample_Rate),
                }

            self.writer_dict['Evaluation'].add_image_dict(image_dict, self.steps)
            self.writer_dict['Evaluation'].add_audio_dict(audio_dict, self.steps)

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
                        'Evaluation.Mel.Target': wandb.Image(target_mel),
                        'Evaluation.Mel.Prediction': wandb.Image(prediction_mel),                        
                        'Evaluation.Audio.Target': wandb.Audio(
                            target_audio,
                            sample_rate= self.hp.Sound.Sample_Rate,
                            caption= 'Target_Audio'
                            ),
                        'Evaluation.Audio.Linear': wandb.Audio(
                            prediction_audio,
                            sample_rate= self.hp.Sound.Sample_Rate,
                            caption= 'Linear_Audio'
                            ),
                        },
                    step= self.steps,
                    commit= True
                    )

        self.scalar_dict['Evaluation'] = defaultdict(float)

        for model in self.model_dict.values():
            model.train()

    @torch.inference_mode()
    def Inference_Step(self, source_audios, source_audio_lengths, source_audio_paths, start_index= 0, tag_step= False):
        source_audios = source_audios.to(self.device, non_blocking=True)
        source_audio_lengths = source_audio_lengths.to(self.device, non_blocking=True)

        prediction_audios = self.model_dict['Synthesizer'].Inference(
            source_audios= source_audios,
            source_audio_lengths= source_audio_lengths
            )

        mel_lengths = source_audio_lengths // self.hp.Sound.Hop_Size
        prediction_mels = [
            mel[:, :length]
            for mel, length in zip(self.mel_func(prediction_audios.cpu()).numpy(), mel_lengths)
            ]
        prediction_audios = [
            audio[:length]
            for audio, length in zip(prediction_audios.cpu().numpy(), source_audio_lengths)
            ]

        files = []
        for index in range(source_audios.size(0)):
            tags = []
            if tag_step: tags.append('Step-{}'.format(self.steps))
            tags.append('IDX_{}'.format(index + start_index))
            files.append('.'.join(tags))

        os.makedirs(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'PNG').replace('\\', '/'), exist_ok= True)
        os.makedirs(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'WAV').replace('\\', '/'), exist_ok= True)
        for index, (
            mel,
            audio,
            source_audio_path,
            file
            ) in enumerate(zip(
            prediction_mels,
            prediction_audios,
            source_audio_paths,
            files
            )):
            title = '    '.join([
                f'Source: {source_audio_path if len(source_audio_path) < 90 else source_audio_path[-90:]}'
                ])
            new_figure = plt.figure(figsize=(20, 5 * 3), dpi=100)
            ax = plt.subplot2grid((2, 1), (0, 0))
            plt.imshow(mel, aspect='auto', origin='lower')
            plt.title(f'Prediction  {title}')
            plt.colorbar(ax= ax)
            ax = plt.subplot2grid((2, 1), (1, 0))
            plt.plot(audio)
            plt.margins(x= 0)
            plt.title(f'Audio    {title}')
            plt.colorbar(ax= ax)
            plt.tight_layout()
            plt.savefig(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'PNG', '{}.png'.format(file)).replace('\\', '/'))
            plt.close(new_figure)
            
            wavfile.write(
                os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'WAV', '{}.wav'.format(file)).replace('\\', '/'),
                self.hp.Sound.Sample_Rate,
                audio
                )
            
    def Inference_Epoch(self):
        if self.gpu_id != 0:
            return

        logging.info('(Steps: {}) Start inference.'.format(self.steps))

        for model in self.model_dict.values():
            model.eval()

        batch_size = self.hp.Inference_Batch_Size or self.hp.Train.Batch_Size
        for step, (source_audios, source_audio_lengths, source_audio_paths) in tqdm(
            enumerate(self.dataloader_dict['Inference']),
            desc='[Inference]',
            total= math.ceil(len(self.dataloader_dict['Inference'].dataset) / batch_size)
            ):
            self.Inference_Step(source_audios, source_audio_lengths, source_audio_paths, start_index= step * batch_size)

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
        self.model_dict['Synthesizer'].load_state_dict(state_dict['Model']['Synthesizer'])
        self.model_dict['Discriminator'].load_state_dict(state_dict['Model']['Discriminator'])
        self.optimizer_dict['Synthesizer'].load_state_dict(state_dict['Optimizer']['Synthesizer'])
        self.optimizer_dict['Discriminator'].load_state_dict(state_dict['Optimizer']['Discriminator'])
        self.scheduler_dict['Synthesizer'].load_state_dict(state_dict['Scheduler']['Synthesizer'])
        self.scheduler_dict['Discriminator'].load_state_dict(state_dict['Scheduler']['Discriminator'])
        self.steps = state_dict['Steps']

        logging.info('Checkpoint loaded at {} steps in GPU {}.'.format(self.steps, self.gpu_id))

    def Save_Checkpoint(self):
        if self.gpu_id != 0:
            return

        os.makedirs(self.hp.Checkpoint_Path, exist_ok= True)
        state_dict = {
            'Model': {
                'Synthesizer': self.model_dict['Synthesizer'].state_dict(),
                'Discriminator': self.model_dict['Discriminator'].state_dict(),
                },
            'Optimizer': {
                'Synthesizer': self.optimizer_dict['Synthesizer'].state_dict(),
                'Discriminator': self.optimizer_dict['Discriminator'].state_dict(),
                },
            'Scheduler': {
                'Synthesizer': self.scheduler_dict['Synthesizer'].state_dict(),
                'Discriminator': self.scheduler_dict['Discriminator'].state_dict(),
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

        if self.hp.Train.Initial_Inference:
            self.Inference_Epoch()

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