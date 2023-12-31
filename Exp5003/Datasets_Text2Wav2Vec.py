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
        token_dict: Dict[str, int],
        pattern_path: str,
        metadata_file: str,
        text_length_min: int,
        text_length_max: int,
        audio_length_min: int,
        audio_length_max: int,
        accumulated_dataset_epoch: int= 1,
        augmentation_ratio: float= 0.0,
        use_pattern_cache: bool= False
        ):
        super().__init__()
        self.token_dict = token_dict
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
                metadata_dict['Text_Length_Dict'][x] >= text_length_min,
                metadata_dict['Text_Length_Dict'][x] <= text_length_max,
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

        token_without_between_pad = Text_to_Token(pattern_dict['Pronunciation'], self.token_dict)

        # padding between tokens
        token_with_between_pad = ['<P>'] * (len(pattern_dict['Pronunciation']) * 2 - 1)
        token_with_between_pad[0::2] = pattern_dict['Pronunciation']
        token_with_between_pad = Text_to_Token(token_with_between_pad, self.token_dict)
                
        return token_with_between_pad, pattern_dict['Audio'], token_without_between_pad

    def __len__(self):
        return len(self.patterns)    

class Inference_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        token_dict: Dict[str, int],
        texts: List[str],
        reference_audio_paths: List[str],
        sample_rate: int,
        hop_size: int,
        ):
        super().__init__()
        self.token_dict = token_dict
        self.sample_rate = sample_rate
        self.hop_size = hop_size

        pronunciations = Phonemize(texts, 'English', use_tqdm= False)
        self.patterns = []
        for index, (text, pronunciation, reference_audio_path) in enumerate(zip(texts, pronunciations, reference_audio_paths)):
            text = Text_Filtering(text)
            if text is None or text == '':
                logging.warning('The text of index {} is incorrect. This index is ignoired.'.format(index))
                continue
            if not os.path.exists(reference_audio_path):
                logging.warning('The source audio path of index {} is incorrect. This index is ignoired.'.format(index))
                continue
            self.patterns.append((text, pronunciation, reference_audio_path))

    def __getitem__(self, idx):
        text, pronunciation, reference_audio_path = self.patterns[idx]

        # padding between tokens
        token = ['<P>'] * (len(pronunciation) * 2 - 1)
        token[0::2] = pronunciation
        token = Text_to_Token(token, self.token_dict)

        reference_audio, _ = librosa.load(reference_audio_path, sr= self.sample_rate)
        reference_audio = librosa.util.normalize(reference_audio) * 0.95
        reference_audio = reference_audio[:reference_audio.shape[0] - (reference_audio.shape[0] % self.hop_size)]

        return token, reference_audio, text, pronunciation, reference_audio_path

    def __len__(self):
        return len(self.patterns)

class Collater:
    def __init__(
        self,
        token_dict: Dict[str, int],
        ):
        self.token_dict = token_dict

    def __call__(self, batch):
        tokens_with_between_pad, audios, tokens_without_between_pad = zip(*batch)

        token_lengths_with_between_pad = np.array([token.shape[0] for token in tokens_with_between_pad])
        audio_lengths = np.array([audio.shape[0] for audio in audios])
        token_lengths_without_between_pad = np.array([token.shape[0] for token in tokens_without_between_pad])
        
        tokens_with_between_pad = Token_Stack(
            tokens= tokens_with_between_pad,
            token_dict= self.token_dict
            )
        audios = Audio_Stack(
            audios= audios
            )
        tokens_without_between_pad = Token_Stack(
            tokens= tokens_without_between_pad,
            token_dict= self.token_dict
            )
        
        tokens_with_between_pad = torch.IntTensor(tokens_with_between_pad)   # [Batch, Token_t]
        token_lengths_with_between_pad = torch.IntTensor(token_lengths_with_between_pad)   # [Batch]
        audios = torch.FloatTensor(audios)    # [Batch, Audio_t], Audio_t == Feature_t * hop_size
        audio_lengths = torch.IntTensor(audio_lengths)   # [Batch]
        tokens_without_between_pad = torch.IntTensor(tokens_without_between_pad)   # [Batch, Token_t]
        token_lengths_without_between_pad = torch.IntTensor(token_lengths_without_between_pad)   # [Batch]

        return tokens_with_between_pad, token_lengths_with_between_pad, audios, audio_lengths, tokens_without_between_pad, token_lengths_without_between_pad

class Inference_Collater:
    def __init__(
        self,
        token_dict: Dict[str, int],
        ):
        self.token_dict = token_dict

    def __call__(self, batch):
        tokens, reference_audios, texts, pronunciations, reference_audio_paths = zip(*batch)
        
        token_lengths = np.array([token.shape[0] for token in tokens])
        reference_audio_lengths = np.array([audio.shape[0] for audio in reference_audios])
        
        tokens = Token_Stack(
            tokens= tokens,
            token_dict= self.token_dict
            )
        reference_audios = Audio_Stack(
            audios= reference_audios
            )
        
        tokens = torch.IntTensor(tokens)   # [Batch, Token_t]
        token_lengths = torch.IntTensor(token_lengths)   # [Batch]
        reference_audios = torch.FloatTensor(reference_audios)    # [Batch, Audio_t], Audio_t == Feature_t * hop_size
        reference_audio_lengths = torch.IntTensor(reference_audio_lengths)   # [Batch]

        return tokens, reference_audios, token_lengths, reference_audio_lengths, texts, pronunciations, reference_audio_paths