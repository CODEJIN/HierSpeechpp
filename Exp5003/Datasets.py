from argparse import Namespace
import torch
import numpy as np
import pickle, os, logging, librosa
from typing import Dict, List, Optional
import functools

from .Pattern_Generator import Text_Filtering, Phonemize

def Text_to_Token(text: str, token_dict: Dict[str, int]):
    return np.array([
        token_dict[letter]
        for letter in ['<S>'] + list(text) + ['<E>']
        ], dtype= np.int32)

def Token_Stack(tokens: List[np.ndarray], token_dict, max_length: Optional[int]= None):
    max_token_length = max_length or max([token.shape[0] for token in tokens])
    tokens = np.stack(
        [np.pad(token, [0, max_token_length - token.shape[0]], constant_values= token_dict['<E>']) for token in tokens],
        axis= 0
        )
    return tokens

def Feature_Stack(features: List[np.ndarray], max_length: Optional[int]= None):
    max_feature_length = max_length or max([feature.shape[1] for feature in features])
    features = np.stack(
        [np.pad(feature, [[0, 0], [0, max_feature_length - feature.shape[1]]], constant_values= feature.min()) for feature in features],
        axis= 0
        )
    return features

def Audio_Stack(audios: List[np.ndarray], max_length: Optional[int]= None):
    max_audio_length = max_length or max([audio.shape[0] for audio in audios])
    audios = np.stack(
        [np.pad(audio, [0, max_audio_length - audio.shape[0]], constant_values= 0.0) for audio in audios],
        axis= 0
        )
    return audios

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        pattern_path: str,
        metadata_file: str,
        audio_length_min: int,
        audio_length_max: int,
        accumulated_dataset_epoch: int= 1,
        augmentation_ratio: float= 0.0,
        use_pattern_cache: bool= False
        ):
        super().__init__()
        self.pattern_path = pattern_path

        metadata_dict = pickle.load(open(
            os.path.join(pattern_path, metadata_file).replace('\\', '/'), 'rb'
            ))
        
        self.patterns = []
        max_pattern_by_speaker = max([
            len(patterns)
            for patterns in metadata_dict['File_List_by_Speaker_Dict'].values()
            ])
        for patterns in metadata_dict['File_List_by_Speaker_Dict'].values():
            ratio = float(len(patterns)) / float(max_pattern_by_speaker)
            if ratio < augmentation_ratio:
                patterns *= int(np.ceil(augmentation_ratio / ratio))
            self.patterns.extend(patterns)

        self.patterns = [
            x for x in self.patterns
            if all([
                metadata_dict['Audio_Length_Dict'][x] >= audio_length_min,
                metadata_dict['Audio_Length_Dict'][x] <= audio_length_max
                ])
            ] * accumulated_dataset_epoch

        if use_pattern_cache:
            self.Pattern_LRU_Cache = functools.lru_cache(maxsize= None)(self.Pattern_LRU_Cache)
    
    def __getitem__(self, idx):
        path = os.path.join(self.pattern_path, self.patterns[idx]).replace('\\', '/')
        return self.Pattern_LRU_Cache(path)
    
    def Pattern_LRU_Cache(self, path: str):
        pattern_dict = pickle.load(open(path, 'rb'))
        
        return pattern_dict['Audio']

    def __len__(self):
        return len(self.patterns)    

class Inference_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        source_audio_paths: List[str],
        sample_rate: int,
        hop_size: int,
        ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_size = hop_size

        self.patterns = []
        for index, source_audio_path in enumerate(source_audio_paths):
            if not os.path.exists(source_audio_path):
                logging.warning('The source audio path of index {} is incorrect. This index is ignoired.'.format(index))
                continue
            self.patterns.append(source_audio_path)

    def __getitem__(self, idx):
        source_audio_path = self.patterns[idx]

        source_audio, _ = librosa.load(source_audio_path, sr= self.sample_rate)
        source_audio = librosa.util.normalize(source_audio) * 0.95
        source_audio = source_audio[:source_audio.shape[0] - (source_audio.shape[0] % self.hop_size)]

        return source_audio, source_audio_path

    def __len__(self):
        return len(self.patterns)

class Collater:
    def __call__(self, batch):
        audios = batch
        audio_lengths = np.array([audio.shape[0] for audio in audios])
        
        audios = Audio_Stack(
            audios= audios
            )
        
        audios = torch.FloatTensor(audios)    # [Batch, Audio_t], Audio_t == Feature_t * hop_size
        audio_lengths = torch.IntTensor(audio_lengths)   # [Batch]

        return audios, audio_lengths

class Inference_Collater:
    def __call__(self, batch):
        source_audios, source_audio_paths = zip(*batch)
        source_audio_lengths = np.array([audio.shape[0] for audio in source_audios])
        
        source_audios = Audio_Stack(
            audios= source_audios
            )
        
        source_audios = torch.FloatTensor(source_audios)    # [Batch, Audio_t], Audio_t == Feature_t * hop_size
        source_audio_lengths = torch.IntTensor(source_audio_lengths)   # [Batch]

        return source_audios, source_audio_lengths, source_audio_paths