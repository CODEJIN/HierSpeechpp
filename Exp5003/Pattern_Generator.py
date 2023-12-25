import torch, torchaudio
import numpy as np
import yaml, os, pickle, librosa, re, argparse, math, logging, sys, asyncio, json
from concurrent.futures import ThreadPoolExecutor as PE
from random import shuffle
from tqdm import tqdm
from pysptk.sptk import rapt
from typing import List, Tuple, Dict, Union, Optional
from collections import defaultdict

from phonemizer import logger as phonemizer_logger
from phonemizer.backend import BACKENDS
from phonemizer.punctuation import Punctuation
from phonemizer.separator import default_separator
from phonemizer.phonemize import _phonemize
from unidecode import unidecode

from Arg_Parser import Recursive_Parse

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout,
    format= '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
    )
using_extension = [x.upper() for x in ['.wav', '.m4a', '.flac']]
regex_checker = re.compile('[가-힣A-Za-z,.?!\'\-\s]+')

logger = phonemizer_logger.get_logger()
logger.setLevel(logging.CRITICAL)
phonemizer_dict = {
    key: BACKENDS['espeak'](
        language= language,
        punctuation_marks= Punctuation.default_marks(),
        preserve_punctuation= True,
        with_stress= True,
        tie= False,
        language_switch= 'remove-flags',
        words_mismatch= 'ignore',
        logger= logger
        )
    for key, language in [
        ('English', 'en-us'),
        ('Korean', 'ko')
        ]
    }

if __name__ == '__main__':
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')

def Text_Filtering(text: str):
    remove_letter_list = ['(', ')', '\"', '[', ']', ':', ';']
    replace_list = [('  ', ' '), (' ,', ','), ('\' ', '\''), ('“', ''), ('”', ''), ('’', '\'')]

    text = text.strip()
    for filter in remove_letter_list:
        text= text.replace(filter, '')
    for filter, replace_STR in replace_list:
        text= text.replace(filter, replace_STR)

    text= text.strip()
    
    if len(regex_checker.findall(text)) != 1:
        return None
    elif text.startswith('\''):
        return None
    else:
        return regex_checker.findall(text)[0]

whitespace_re = re.compile(r'\s+')
english_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
    ]]
def expand_abbreviations(text: str):
    for regex, replacement in english_abbreviations:
        text = re.sub(regex, replacement, text)

    return text

def Phonemize(texts: Union[str, List[str]], language: str, chunk: int= 5000, use_tqdm: bool= True):
    if type(texts) == str:
        texts = [texts]

    if language == 'English':
        texts = [text.lower() for text in texts]
        texts = [expand_abbreviations(text) for text in texts]

    pronunciations = []

    indices = range(0, len(texts), chunk)
    if use_tqdm:
        indices = tqdm(indices, desc= 'Phonemize')
    for index in indices:
        pronunciations_chunk = _phonemize(
            backend= phonemizer_dict[language],
            text= texts[index:index + chunk],
            separator= default_separator,
            strip= True,
            njobs= 4,
            prepend_text= False,
            preserve_empty_lines= False
            )
        pronunciations.extend([
            re.sub(whitespace_re, ' ', pronunciation)
            for pronunciation in pronunciations_chunk
            ])

    return pronunciations

def Audio_Stack(audios: List[np.ndarray], max_length: Optional[int]= None) -> np.ndarray:
    max_audio_length = max_length or max([audio.shape[0] for audio in audios])
    audios = np.stack(
        [np.pad(audio, [0, max_audio_length - audio.shape[0]], constant_values= 0.0) for audio in audios],
        axis= 0
        )

    return audios

async def Read_audio_and_Trim(
    path: str,
    sample_rate: int,
    hop_size: int,
    f0_min: float,
    f0_max: float
    ):
    loop = asyncio.get_event_loop()

    def Read():
        audio, _ = librosa.load(path, sr=sample_rate)
        audio = librosa.util.normalize(audio) * 0.95

        audio = audio[:audio.shape[0] - (audio.shape[0] % hop_size)]

        f0 = rapt(
            x= audio * 32768,
            fs= sample_rate,
            hopsize= hop_size,
            min= f0_min,
            max= f0_max,
            otype= 1
            )
        
        nonsilence_frames = np.where(f0 > 0.0)[0]
        if len(nonsilence_frames) < 2:
            return None
        initial_silence_frame, *_, last_silence_frame = nonsilence_frames
        initial_silence_frame = max(initial_silence_frame - 21, 0)
        last_silence_frame = min(last_silence_frame + 21, f0.shape[0])
        audio = audio[initial_silence_frame * hop_size:last_silence_frame * hop_size]        

        return audio
    
    return await loop.run_in_executor(None, Read)

async def Pattern_Generate(
    paths,
    sample_rate: int,
    hop_size: int,
    f0_min: int,
    f0_max: int,
    ):
    tasks = [
        Read_audio_and_Trim(
            path= path,
            sample_rate= sample_rate,
            hop_size= hop_size,
            f0_min= f0_min,
            f0_max= f0_max
            )
        for path in paths
        ]
    audios = await asyncio.gather(*tasks)
    is_valid_list = [
        not audio is None
        for audio in audios
        ]
    valid_patterns = [
        (path, audio)
        for path, audio in zip(paths, audios)
        if not audio is None
        ]
    if len(valid_patterns) == 0:
        return [None] * len(paths)
    paths, audios = zip(*valid_patterns)
    
    audios_valid: List[np.ndarray] = []
    current_index = 0
    for is_valid in is_valid_list:
        if is_valid:
            audios_valid.append(audios[current_index])
            current_index += 1
        else:
            audios_valid.append(None)

    return audios_valid

def Pattern_File_Generate(
    paths: List[str],
    speakers: List[str],
    emotions: List[str],
    languages: List[str],
    genders: List[str],
    datasets: List[str],
    texts: List[str],
    pronunciations: List[str],
    tags: Optional[List[str]]= None,
    eval: bool= False
    ):
    pattern_path = hp.Train.Eval_Pattern.Path if eval else hp.Train.Train_Pattern.Path

    tags = tags or [''] * len(paths)
    files = [
        '{}.{}{}.PICKLE'.format(
        speaker if dataset in speaker else '{}.{}'.format(dataset, speaker),
        '{}.'.format(tag) if tag != '' else '',
        os.path.splitext(os.path.basename(path))[0]
        ).upper()
        for path, speaker, dataset, tag in zip(paths, speakers, datasets, tags)
        ]    
    non_existed_patterns = [
        (path, file, speaker, emotion, language, gender, dataset, text, pronunciation)
        for path, file, speaker, emotion, language, gender, dataset, text, pronunciation in zip(
            paths, files, speakers, emotions, languages, genders, datasets, texts, pronunciations
            )
        if not any([
            os.path.exists(os.path.join(x, dataset, speaker, file).replace("\\", "/"))
            for x in [hp.Train.Eval_Pattern.Path, hp.Train.Train_Pattern.Path]
            ])
        ]
    if len(non_existed_patterns) == 0:
        return
    paths, files, speakers, emotions, languages, genders, datasets, texts, pronunciations = zip(*non_existed_patterns)
    files = [
        os.path.join(pattern_path, dataset, speaker, file).replace('\\', '/')
        for file, speaker, dataset in zip(files, speakers, datasets)
        ]
    
    if len(files) == 0:
        return

    audios = asyncio.run(Pattern_Generate(
        paths= paths,
        sample_rate= hp.Sound.Sample_Rate,
        hop_size= hp.Sound.Hop_Size,
        f0_min= hp.Sound.F0_Min,
        f0_max= hp.Sound.F0_Max
        ))

    for file, audio, speaker, emotion, language, gender, dataset, text, pronunciation in zip(
        files, audios, speakers, emotions, languages, genders, datasets, texts, pronunciations
        ):
        if audio is None:
            logging.warning(f'\'{file}\' is skipped.')
            continue
        new_pattern_dict = {
            'Audio': audio,
            'Speaker': speaker,
            'Emotion': emotion,
            'Language': language,
            'Gender': gender,
            'Dataset': dataset,
            'Text': text,
            'Pronunciation': pronunciation
            }
        os.makedirs(os.path.join(pattern_path, dataset, speaker).replace('\\', '/'), exist_ok= True)
        with open(file, 'wb') as f:
            pickle.dump(new_pattern_dict, f, protocol=4)
        del new_pattern_dict

def Selvas_Info_Load(path: str):
    '''
    ema, emb, emf, emg, emh, nea, neb, nec, ned, nee, nek, nel, nem, nen, neo
    1-100: Neutral
    101-200: Happy
    201-300: Sad
    301-400: Angry

    lmy, ava, avb, avc, avd, ada, adb, adc, add:
    all neutral
    '''
    if os.path.exists('Selvas_Load.pickle'):
        pickled_info_dict = pickle.load(open('Selvas_Load.pickle', 'rb'))
        paths = pickled_info_dict['Paths']
        text_dict = pickled_info_dict['Text_Dict']
        pronunciation_dict = pickled_info_dict['Pronunciation_Dict']
        speaker_dict = pickled_info_dict['Speaker_Dict']
        emotion_dict = pickled_info_dict['Emotion_Dict']
        language_dict = pickled_info_dict['Language_Dict']
        gender_dict = pickled_info_dict['Gender_Dict']
        
        print(f'Selvas info generated: {len(paths)}')
        return paths, text_dict, pronunciation_dict, speaker_dict, emotion_dict, language_dict, gender_dict
    
    paths = []
    for root, _, files in os.walk(path):
        for file in files:
            file = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file)[1].upper() in using_extension:
                continue
            if any(['lmy04282' in file, 'lmy07365' in file, 'lmy05124' in file]):
                continue
            paths.append(file)

    text_dict = {}
    
    for wav_path in paths:
        text = open(wav_path.replace('/wav/', '/transcript/').replace('.wav', '.txt'), 'r', encoding= 'utf-8-sig').readlines()[0].strip()
        text = Text_Filtering(text)
        if text is None:
            continue

        text_dict[wav_path] = text

    paths = list(text_dict.keys())

    pronunciations = Phonemize(
        texts= [text_dict[path] for path in paths],
        language= 'Korean'
        )    
    pronunciation_dict = {path: pronunciation for path, pronunciation in zip(paths, pronunciations)}

    speaker_dict = {
        path: path.split('/')[-3].strip().upper()
        for path in paths
        }
    
    emotion_dict = {}
    for path in paths:
        if speaker_dict[path] in ['LMY', 'KIH', 'AVA', 'AVB', 'AVC', 'AVD', 'ADA', 'ADB', 'ADC', 'ADD', 'PFA', 'PFB', 'PFC', 'PFD', 'PFI', 'PFL', 'PFM', 'PFO', 'PFP', 'PMA', 'PMB', 'PMC', 'PMD', 'PMI', 'PMJ', 'PML']:
            emotion_dict[path] = 'Neutral'
        elif speaker_dict[path] in ['EMA', 'EMB', 'EMF', 'EMG', 'EMH', 'NEA', 'NEB', 'NEC', 'NED', 'NEE', 'NEK', 'NEL', 'NEM', 'NEN', 'NEO']:
            index = int(os.path.splitext(os.path.basename(path))[0][-5:])
            if index > 0 and index < 101:
                emotion_dict[path] = 'Neutral'
            elif index > 100 and index < 201:
                emotion_dict[path] = 'Happy'
            elif index > 200 and index < 301:
                emotion_dict[path] = 'Sad'
            elif index > 300 and index < 401:
                emotion_dict[path] = 'Angry'
            else:
                raise NotImplementedError(f'Unknown emotion index: {index}')
        else:
            raise NotImplementedError(f'Unknown speaker: {speaker_dict[path]}')

    language_dict = {path: 'Korean' for path in paths}

    gender_dict = {
        'ADA': 'Female',
        'ADB': 'Female',
        'ADC': 'Male',
        'ADD': 'Male',
        'AVA': 'Female',
        'AVB': 'Female',
        'AVC': 'Female',
        'AVD': 'Female',
        'EMA': 'Female',
        'EMB': 'Female',
        'EMF': 'Male',
        'EMG': 'Male',
        'EMH': 'Male',
        'KIH': 'Male',
        'LMY': 'Female',
        'NEA': 'Female',
        'NEB': 'Female',
        'NEC': 'Female',
        'NED': 'Female',
        'NEE': 'Female',
        'NEK': 'Male',
        'NEL': 'Male',
        'NEM': 'Male',
        'NEN': 'Male',
        'NEO': 'Male',
        'PFA': 'Female',
        'PFB': 'Female',
        'PFC': 'Female',
        'PFD': 'Female',
        'PFI': 'Female',
        'PFL': 'Female',
        'PFM': 'Female',
        'PFO': 'Female',
        'PFP': 'Female',
        'PMA': 'Male',
        'PMB': 'Male',
        'PMC': 'Male',
        'PMD': 'Male',
        'PMI': 'Male',
        'PMJ': 'Male',
        'PML': 'Male',
        }
    gender_dict = {
        path: gender_dict[speaker]
        for path, speaker in speaker_dict.items()
        }

    print(f'Selvas info generated: {len(paths)}')
    with open('Selvas_Load.pickle', 'wb') as f:
        pickle.dump({
            'Paths': paths,
            'Text_Dict': text_dict,
            'Pronunciation_Dict': pronunciation_dict,
            'Speaker_Dict': speaker_dict,
            'Emotion_Dict': emotion_dict,
            'Language_Dict': language_dict,
            'Gender_Dict': gender_dict
            }, f, protocol= 4)
    return paths, text_dict, pronunciation_dict, speaker_dict, emotion_dict, language_dict, gender_dict

def KSS_Info_Load(path: str):
    '''
    all neutral
    '''
    if os.path.exists('KSS_Load.pickle'):
        pickled_info_dict = pickle.load(open('KSS_Load.pickle', 'rb'))
        paths = pickled_info_dict['Paths']
        text_dict = pickled_info_dict['Text_Dict']
        pronunciation_dict = pickled_info_dict['Pronunciation_Dict']
        speaker_dict = pickled_info_dict['Speaker_Dict']
        emotion_dict = pickled_info_dict['Emotion_Dict']
        language_dict = pickled_info_dict['Language_Dict']
        gender_dict = pickled_info_dict['Gender_Dict']
        
        print(f'KSS info generated: {len(paths)}')
        return paths, text_dict, pronunciation_dict, speaker_dict, emotion_dict, language_dict, gender_dict

    paths, text_dict = [], {}
    for line in open(os.path.join(path, 'transcript.v.1.4.txt').replace('\\', '/'), 'r', encoding= 'utf-8-sig').readlines():
        line = line.strip().split('|')
        file, text = line[0].strip(), line[2].strip()
        text = Text_Filtering(text)
        if text is None:
            continue
        
        file = os.path.join(path, 'kss', file).replace('\\', '/')
        paths.append(file)
        text_dict[file] = text

    pronunciations = Phonemize(
        texts= [text_dict[path] for path in paths],
        language= 'Korean'
        )    
    pronunciation_dict = {path: pronunciation for path, pronunciation in zip(paths, pronunciations)}

    speaker_dict = {
        path: 'KSS'
        for path in paths
        }
    emotion_dict = {
        path: 'Neutral'
        for path in paths
        }
    language_dict = {
        path: 'Korean'
        for path in paths
        }
    gender_dict = {
        path: 'Female'
        for path in paths
        }

    print(f'KSS info generated: {len(paths)}')
    with open('KSS_Load.pickle', 'wb') as f:
        pickle.dump({
            'Paths': paths,
            'Text_Dict': text_dict,
            'Pronunciation_Dict': pronunciation_dict,
            'Speaker_Dict': speaker_dict,
            'Emotion_Dict': emotion_dict,
            'Language_Dict': language_dict,
            'Gender_Dict': gender_dict
            }, f, protocol= 4)
    return paths, text_dict, pronunciation_dict, speaker_dict, emotion_dict, language_dict, gender_dict

def AIHub_Info_Load(path: str, n_sample_by_speaker: Optional[int]= None):
    if os.path.exists('AIHub_Load.pickle'):
        pickled_info_dict = pickle.load(open('AIHub_Load.pickle', 'rb'))
        paths = pickled_info_dict['Paths']
        text_dict = pickled_info_dict['Text_Dict']
        pronunciation_dict = pickled_info_dict['Pronunciation_Dict']
        speaker_dict = pickled_info_dict['Speaker_Dict']
        emotion_dict = pickled_info_dict['Emotion_Dict']
        language_dict = pickled_info_dict['Language_Dict']
        gender_dict = pickled_info_dict['Gender_Dict']
        
        print(f'AIHub info generated: {len(paths)}')
        return paths, text_dict, pronunciation_dict, speaker_dict, emotion_dict, language_dict, gender_dict

    emotion_label_dict = {
        'Neutrality': 'Neutral'
        }

    skip_info_keys = []
    info_dict = {}
    n_sample_by_speaker_dict = defaultdict(int)
    for root, _, files in os.walk(path):
        for file in files:
            key, extension = os.path.splitext(file)
            if not key in info_dict.keys():
                info_dict[key] = {}
            
            path = os.path.join(root, file).replace('\\', '/')
            if extension.upper() == '.WAV':
                info_dict[key]['Path'] = path
            elif extension.upper() == '.JSON':
                pattern_info = json.load(open(path, encoding= 'utf-8-sig'))
                text = Text_Filtering(pattern_info['전사정보']['TransLabelText'].replace('\xa0', ' '))
                if text is None:
                    skip_info_keys.append(key)
                    continue

                info_dict[key]['Text'] = text
                info_dict[key]['Speaker'] = 'AIHub_{}'.format(pattern_info['화자정보']['SpeakerName'])
                info_dict[key]['Gender'] = pattern_info['화자정보']['Gender']
                info_dict[key]['Emotion'] = emotion_label_dict[pattern_info['화자정보']['Emotion']]

                if not n_sample_by_speaker is None and n_sample_by_speaker_dict[info_dict[key]['Speaker']] >= n_sample_by_speaker:
                    skip_info_keys.append(key)
                    continue
                n_sample_by_speaker_dict[info_dict[key]['Speaker']] += 1
            else:
                # raise ValueError(f'Unsupported file type: {path}')
                logging.info(f'Unsupported file type skipped: {path}')
                skip_info_keys.append(key)
                continue

    for key in skip_info_keys:
        info_dict.pop(key, None)
    
    info_dict = {
        key: value
        for key, value in info_dict.items()
        if all([x in value.keys() for x in ['Path', 'Text', 'Speaker', 'Gender', 'Emotion']])
        }
    paths = [info['Path'] for info in info_dict.values()]
    text_dict = {
        info['Path']: info['Text']  for info in info_dict.values()
        }
    pronunciations = Phonemize(
        texts= [text_dict[path] for path in paths],
        language= 'Korean'
        )    
    pronunciation_dict = {path: pronunciation for path, pronunciation in zip(paths, pronunciations)}

    speaker_dict = {
        info['Path']: info['Speaker']  for info in info_dict.values()
        }
    emotion_dict = {
        info['Path']: info['Emotion']  for info in info_dict.values()
        }
    language_dict = {
        path: 'Korean'
        for path in paths
        }
    gender_dict = {
        info['Path']: info['Gender']  for info in info_dict.values()
        }

    print(f'AIHub info generated: {len(paths)}')
    with open('AIHub_Load.pickle', 'wb') as f:
        pickle.dump({
            'Paths': paths,
            'Text_Dict': text_dict,
            'Pronunciation_Dict': pronunciation_dict,
            'Speaker_Dict': speaker_dict,
            'Emotion_Dict': emotion_dict,
            'Language_Dict': language_dict,
            'Gender_Dict': gender_dict
            }, f, protocol= 4)
    return paths, text_dict, pronunciation_dict, speaker_dict, emotion_dict, language_dict, gender_dict

def Basic_Info_Load(
    path: str,
    dataset_label: str,
    language: Optional[Union[str, Dict[str, str]]]= None,
    gender: Optional[Union[str, Dict[str, str]]]= None
    ):
    '''
    This is to use a customized dataset.
    path: Dataset path. In the path, there must be a 'script.txt' file. 'script.txt' must have the following structure: 'Path\tScript\tSpeaker\tEmotion'.
    dataset_label: The lable of dataset
    language: The language of dataset or speaker. When dataset is multi language, this parameter is dictionary that key and value are speaker and language, resplectly.
    gender: The gender of dataset or speaker. When dataset is multi language, this parameter is dictionary that key and value are speaker and gender, resplectly.
    '''
    if os.path.exists(f'{dataset_label}_Load.pickle'):
        pickled_info_dict = pickle.load(open(f'{dataset_label}_Load.pickle', 'rb'))
        paths = pickled_info_dict['Paths']
        text_dict = pickled_info_dict['Text_Dict']
        pronunciation_dict = pickled_info_dict['Pronunciation_Dict']
        speaker_dict = pickled_info_dict['Speaker_Dict']
        emotion_dict = pickled_info_dict['Emotion_Dict']
        language_dict = pickled_info_dict['Language_Dict']
        gender_dict = pickled_info_dict['Gender_Dict']
        
        print(f'{dataset_label} info generated: {len(paths)}')
        return paths, text_dict, pronunciation_dict, speaker_dict, emotion_dict, language_dict, gender_dict
    
    paths = []
    for root, _, files in os.walk(path):
        for file in files:
            file = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file)[1].upper() in using_extension:
                continue
            paths.append(file)
    
    text_dict = {}
    speaker_dict = {}
    emotion_dict = {}

    for line in open(os.path.join(path, 'scripts.txt').replace('\\', '/'), 'r', encoding= 'utf-8-sig').readlines()[1:]:
        file, text, speaker, emotion = line.strip().split('\t')
        if (type(language) == str and language == 'English') or (type(language) == dict and language[speaker] == 'English'):
            text = unidecode(text)  # When English script, unidecode called.
        text = Text_Filtering(text)
        if text is None:
            continue
        
        text_dict[os.path.join(path, file).replace('\\', '/')] = text
        speaker_dict[os.path.join(path, file).replace('\\', '/')] = speaker.strip()
        emotion_dict[os.path.join(path, file).replace('\\', '/')] = emotion.strip()

    paths = list(text_dict.keys())

    language_dict = {path: language for path in paths}

    if type(language) == str or language is None:
        language_dict = {path: language for path in paths}
    elif type(language) == dict:
        language_dict = {
            path: language[speaker]
            for path, speaker in speaker_dict.items()
            }

    if type(gender) == str or gender is None:
        gender_dict = {path: gender for path in paths}
    elif type(gender) == dict:
        gender_dict = {
            path: gender[speaker]
            for path, speaker in speaker_dict.items()
            }

    pronunciation_dict = {}
    for language in set(language_dict.values()):
        language_paths = [path for path in paths if language_dict[path] == language]
        language_pronunciations = Phonemize(
            texts= [text_dict[path] for path in language_paths],
            language= language
            )
        pronunciation_dict.update({
            path: pronunciation
            for path, pronunciation in zip(language_paths, language_pronunciations)
            })

    print(f'{dataset_label} info generated: {len(paths)}')
    with open(f'{dataset_label}_Load.pickle', 'wb') as f:
        pickle.dump({
            'Paths': paths,
            'Text_Dict': text_dict,
            'Pronunciation_Dict': pronunciation_dict,
            'Speaker_Dict': speaker_dict,
            'Emotion_Dict': emotion_dict,
            'Language_Dict': language_dict,
            'Gender_Dict': gender_dict
            }, f, protocol= 4)
    return paths, text_dict, pronunciation_dict, speaker_dict, emotion_dict, language_dict, gender_dict


def VCTK_Info_Load(path: str):
    '''
    VCTK v0.92 is distributed as flac files.
    '''
    if os.path.exists('VCTK_Load.pickle'):
        pickled_info_dict = pickle.load(open('VCTK_Load.pickle', 'rb'))
        paths = pickled_info_dict['Paths']
        text_dict = pickled_info_dict['Text_Dict']
        pronunciation_dict = pickled_info_dict['Pronunciation_Dict']
        speaker_dict = pickled_info_dict['Speaker_Dict']
        emotion_dict = pickled_info_dict['Emotion_Dict']
        language_dict = pickled_info_dict['Language_Dict']
        gender_dict = pickled_info_dict['Gender_Dict']
        
        print(f'VCTK info generated: {len(paths)}')
        return paths, text_dict, pronunciation_dict, speaker_dict, emotion_dict, language_dict, gender_dict
    
    path = os.path.join(path, 'wav48').replace('\\', '/')
    
    paths = []
    for root, _, files in os.walk(path):
        for file in files:
            file = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file)[1].upper() in using_extension:
                continue
            elif '_mic1' in file:
                continue

            paths.append(file)

    text_dict = {}
    for path in paths:
        if 'p315'.upper() in path.upper():  #Officially, 'p315' text is lost in VCTK dataset.
            continue
        text = Text_Filtering(unidecode(open(path.replace('wav48', 'txt').replace('flac', 'txt').replace('_mic2', ''), 'r').readlines()[0]))
        if text is None:
            continue
        
        text_dict[path] = text
            
    paths = list(text_dict.keys())
    pronunciations = Phonemize(
        texts= [text_dict[path] for path in paths],
        language= 'English'
        )    
    pronunciation_dict = {path: pronunciation for path, pronunciation in zip(paths, pronunciations)}

    speaker_dict = {
        path: 'VCTK.{}'.format(path.split('/')[-2].strip().upper())
        for path in paths
        }

    emotion_dict = {path: 'Neutral' for path in paths}
    language_dict = {path: 'English' for path in paths}

    gender_dict = {
        'VCTK.P225': 'Female',
        'VCTK.P226': 'Male',
        'VCTK.P227': 'Male',
        'VCTK.P228': 'Female',
        'VCTK.P229': 'Female',
        'VCTK.P230': 'Female',
        'VCTK.P231': 'Female',
        'VCTK.P232': 'Male',
        'VCTK.P233': 'Female',
        'VCTK.P234': 'Female',
        'VCTK.P236': 'Female',
        'VCTK.P237': 'Male',
        'VCTK.P238': 'Female',
        'VCTK.P239': 'Female',
        'VCTK.P240': 'Female',
        'VCTK.P241': 'Male',
        'VCTK.P243': 'Male',
        'VCTK.P244': 'Female',
        'VCTK.P245': 'Male',
        'VCTK.P246': 'Male',
        'VCTK.P247': 'Male',
        'VCTK.P248': 'Female',
        'VCTK.P249': 'Female',
        'VCTK.P250': 'Female',
        'VCTK.P251': 'Male',
        'VCTK.P252': 'Male',
        'VCTK.P253': 'Female',
        'VCTK.P254': 'Male',
        'VCTK.P255': 'Male',
        'VCTK.P256': 'Male',
        'VCTK.P257': 'Female',
        'VCTK.P258': 'Male',
        'VCTK.P259': 'Male',
        'VCTK.P260': 'Male',
        'VCTK.P261': 'Female',
        'VCTK.P262': 'Female',
        'VCTK.P263': 'Male',
        'VCTK.P264': 'Female',
        'VCTK.P265': 'Female',
        'VCTK.P266': 'Female',
        'VCTK.P267': 'Female',
        'VCTK.P268': 'Female',
        'VCTK.P269': 'Female',
        'VCTK.P270': 'Male',
        'VCTK.P271': 'Male',
        'VCTK.P272': 'Male',
        'VCTK.P273': 'Male',
        'VCTK.P274': 'Male',
        'VCTK.P275': 'Male',
        'VCTK.P276': 'Female',
        'VCTK.P277': 'Female',
        'VCTK.P278': 'Male',
        'VCTK.P279': 'Male',
        'VCTK.P280': 'Female',
        'VCTK.P281': 'Male',
        'VCTK.P282': 'Female',
        'VCTK.P283': 'Male',
        'VCTK.P284': 'Male',
        'VCTK.P285': 'Male',
        'VCTK.P286': 'Male',
        'VCTK.P287': 'Male',
        'VCTK.P288': 'Female',
        'VCTK.P292': 'Male',
        'VCTK.P293': 'Female',
        'VCTK.P294': 'Female',
        'VCTK.P295': 'Female',
        'VCTK.P297': 'Female',
        'VCTK.P298': 'Male',
        'VCTK.P299': 'Female',
        'VCTK.P300': 'Female',
        'VCTK.P301': 'Female',
        'VCTK.P302': 'Male',
        'VCTK.P303': 'Female',
        'VCTK.P304': 'Male',
        'VCTK.P305': 'Female',
        'VCTK.P306': 'Female',
        'VCTK.P307': 'Female',
        'VCTK.P308': 'Female',
        'VCTK.P310': 'Female',
        'VCTK.P311': 'Male',
        'VCTK.P312': 'Female',
        'VCTK.P313': 'Female',
        'VCTK.P314': 'Female',
        'VCTK.P316': 'Male',
        'VCTK.P317': 'Female',
        'VCTK.P318': 'Female',
        'VCTK.P323': 'Female',
        'VCTK.P326': 'Male',
        'VCTK.P329': 'Female',
        'VCTK.P330': 'Female',
        'VCTK.P333': 'Female',
        'VCTK.P334': 'Male',
        'VCTK.P335': 'Female',
        'VCTK.P336': 'Female',
        'VCTK.P339': 'Female',
        'VCTK.P340': 'Female',
        'VCTK.P341': 'Female',
        'VCTK.P343': 'Female',
        'VCTK.P345': 'Male',
        'VCTK.P347': 'Male',
        'VCTK.P351': 'Female',
        'VCTK.P360': 'Male',
        'VCTK.P361': 'Female',
        'VCTK.P362': 'Female',
        'VCTK.P363': 'Male',
        'VCTK.P364': 'Male',
        'VCTK.P374': 'Male',
        'VCTK.P376': 'Male',
        'VCTK.S5': 'Female',
        }
    gender_dict = {
        path: gender_dict[speaker]
        for path, speaker in speaker_dict.items()
        }

    print(f'VCTK info generated: {len(paths)}')
    with open('VCTK_Load.pickle', 'wb') as f:
        pickle.dump({
            'Paths': paths,
            'Text_Dict': text_dict,
            'Pronunciation_Dict': pronunciation_dict,
            'Speaker_Dict': speaker_dict,
            'Emotion_Dict': emotion_dict,
            'Language_Dict': language_dict,
            'Gender_Dict': gender_dict
            }, f, protocol= 4)
    return paths, text_dict, pronunciation_dict, speaker_dict, emotion_dict, language_dict, gender_dict

def Libri_Info_Load(path: str, n_sample_by_speaker: Optional[int]= None):
    if os.path.exists('Libri_Load.pickle'):
        pickled_info_dict = pickle.load(open('Libri_Load.pickle', 'rb'))
        paths = pickled_info_dict['Paths']
        text_dict = pickled_info_dict['Text_Dict']
        pronunciation_dict = pickled_info_dict['Pronunciation_Dict']
        speaker_dict = pickled_info_dict['Speaker_Dict']
        emotion_dict = pickled_info_dict['Emotion_Dict']
        language_dict = pickled_info_dict['Language_Dict']
        gender_dict = pickled_info_dict['Gender_Dict']
        
        print(f'Libri info generated: {len(paths)}')
        return paths, text_dict, pronunciation_dict, speaker_dict, emotion_dict, language_dict, gender_dict

    paths = []
    for root, _, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file_path)[1].upper() in using_extension:
                continue
            paths.append(file_path)

    text_dict = {}
    for file_path in paths:
        text = Text_Filtering(unidecode(open(f'{os.path.splitext(file_path)[0]}.normalized.txt', 'r', encoding= 'utf-8-sig').readlines()[0]))
        if text is None:
            continue
        
        text_dict[file_path] = text

    paths = list(text_dict.keys())

    speaker_dict = {
        path: 'Libri.{:04d}'.format(int(path.split('/')[-3].strip().upper()))
        for path in paths
        }

    if not n_sample_by_speaker is None:
        n_sample_by_speaker_dict = defaultdict(int)
        paths_sample = []
        text_dict_sample = {}
        speaker_dict_sample = {}    
        shuffle(paths)
        for path in paths:
            speaker = speaker_dict[path]
            if n_sample_by_speaker_dict[speaker] >= n_sample_by_speaker:
                continue
            paths_sample.append(paths)
            text_dict_sample[path] = text_dict[path]
            speaker_dict_sample[path] = speaker_dict[path]
            n_sample_by_speaker_dict[speaker] += 1
        paths = paths_sample
        text_dict = text_dict_sample
        speaker_dict = speaker_dict_sample

    pronunciations = Phonemize(
        texts= [text_dict[path] for path in paths],
        language= 'English'
        )    
    pronunciation_dict = {path: pronunciation for path, pronunciation in zip(paths, pronunciations)}

    emotion_dict = {path: 'Neutral' for path in paths}
    language_dict = {path: 'English' for path in paths}

    gender_dict = {}
    for line in open(os.path.join(os.path.join(path, 'SPEAKERS.txt').replace('\\', '/')), 'r', encoding= 'utf-8-sig').readlines()[12:]:
        speaker, gender, *_ = [x.strip() for x in line.strip().split('|')]
        gender_dict[f'Libri.{int(speaker):04d}'] = 'Male' if gender == 'M' else 'Female'
    gender_dict = {
        path: gender_dict[speaker]
        for path, speaker in speaker_dict.items()
        }

    print(f'Libri info generated: {len(paths)}')
    with open('Libri_Load.pickle', 'wb') as f:
        pickle.dump({
            'Paths': paths,
            'Text_Dict': text_dict,
            'Pronunciation_Dict': pronunciation_dict,
            'Speaker_Dict': speaker_dict,
            'Emotion_Dict': emotion_dict,
            'Language_Dict': language_dict,
            'Gender_Dict': gender_dict
            }, f, protocol= 4)
    return paths, text_dict, pronunciation_dict, speaker_dict, emotion_dict, language_dict, gender_dict

def LJ_Info_Load(path: str):
    if os.path.exists('LJ_Load.pickle'):
        pickled_info_dict = pickle.load(open('LJ_Load.pickle', 'rb'))
        paths = pickled_info_dict['Paths']
        text_dict = pickled_info_dict['Text_Dict']
        pronunciation_dict = pickled_info_dict['Pronunciation_Dict']
        speaker_dict = pickled_info_dict['Speaker_Dict']
        emotion_dict = pickled_info_dict['Emotion_Dict']
        language_dict = pickled_info_dict['Language_Dict']
        gender_dict = pickled_info_dict['Gender_Dict']
        
        print(f'LJ info generated: {len(paths)}')
        return paths, text_dict, pronunciation_dict, speaker_dict, emotion_dict, language_dict, gender_dict

    paths = []
    for root, _, files in os.walk(path):
        for file in files:
            file = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file)[1].upper() in using_extension:
                continue
            paths.append(file)

    text_dict = {}
    for line in open(os.path.join(path, 'metadata.csv').replace('\\', '/'), 'r', encoding= 'utf-8-sig').readlines():
        line = line.strip().split('|')        
        text = Text_Filtering(unidecode(line[2].strip()))
        if text is None:
            continue
        wav_path = os.path.join(path, 'wavs', f'{line[0]}.wav')
        
        text_dict[wav_path] = text

    paths = list(text_dict.keys())

    pronunciations = Phonemize(
        texts= [text_dict[path] for path in paths],
        language= 'English'
        )    
    pronunciation_dict = {path: pronunciation for path, pronunciation in zip(paths, pronunciations)}
    
    speaker_dict = {
        path: 'LJ'
        for path in paths
        }
    emotion_dict = {path: 'Neutral' for path in paths}
    language_dict = {path: 'English' for path in paths}
    gender_dict = {path: 'Female' for path in paths}

    print(f'LJ info generated: {len(paths)}')
    with open('LJ_Load.pickle', 'wb') as f:
        pickle.dump({
            'Paths': paths,
            'Text_Dict': text_dict,
            'Pronunciation_Dict': pronunciation_dict,
            'Speaker_Dict': speaker_dict,
            'Emotion_Dict': emotion_dict,
            'Language_Dict': language_dict,
            'Gender_Dict': gender_dict
            }, f, protocol= 4)
    return paths, text_dict, pronunciation_dict, speaker_dict, emotion_dict, language_dict, gender_dict


def Split_Eval(paths: List[str], eval_ratio: float= 0.001, min_eval: int= 1):
    shuffle(paths)
    index = max(int(len(paths) * eval_ratio), min_eval)
    return paths[index:], paths[:index]

def Metadata_Generate(eval: bool= False):
    pattern_path = hp.Train.Eval_Pattern.Path if eval else hp.Train.Train_Pattern.Path
    metadata_File = hp.Train.Eval_Pattern.Metadata_File if eval else hp.Train.Train_Pattern.Metadata_File

    speakers = []
    emotions = []
    languages = []
    genders = []
    language_and_gender_dict_by_speaker = {}

    new_metadata_dict = {
        'Hop_Size': hp.Sound.Hop_Size,
        'Sample_Rate': hp.Sound.Sample_Rate,
        'File_List': [],
        'Audio_Length_Dict': {},
        'Speaker_Dict': {},
        'Emotion_Dict': {},
        'Dataset_Dict': {},
        'File_List_by_Speaker_Dict': {},
        'Text_Length_Dict': {},
        }

    files_tqdm = tqdm(
        total= sum([len(files) for root, _, files in os.walk(pattern_path, followlinks=True)]),
        desc= 'Eval_Pattern' if eval else 'Train_Pattern'
        )

    for root, _, files in os.walk(pattern_path, followlinks=True):
        for file in files:
            with open(os.path.join(root, file).replace("\\", "/"), "rb") as f:
                pattern_dict = pickle.load(f)

            file = os.path.join(root, file).replace("\\", "/").replace(pattern_path, '').lstrip('/')

            try:
                if not all([
                    key in pattern_dict.keys()
                    for key in ('Audio', 'Speaker', 'Emotion', 'Language', 'Gender', 'Dataset', 'Text', 'Pronunciation')
                    ]):
                    files_tqdm.update(1)
                    continue
                new_metadata_dict['Audio_Length_Dict'][file] = pattern_dict['Audio'].shape[0]
                new_metadata_dict['Speaker_Dict'][file] = pattern_dict['Speaker']
                new_metadata_dict['Emotion_Dict'][file] = pattern_dict['Emotion']
                new_metadata_dict['Dataset_Dict'][file] = pattern_dict['Dataset']
                new_metadata_dict['File_List'].append(file)
                if not pattern_dict['Speaker'] in new_metadata_dict['File_List_by_Speaker_Dict'].keys():
                    new_metadata_dict['File_List_by_Speaker_Dict'][pattern_dict['Speaker']] = []
                new_metadata_dict['File_List_by_Speaker_Dict'][pattern_dict['Speaker']].append(file)
                new_metadata_dict['Text_Length_Dict'][file] = len(pattern_dict['Text'])

                speakers.append(pattern_dict['Speaker'])
                emotions.append(pattern_dict['Emotion'])
                languages.append(pattern_dict['Language'])
                genders.append(pattern_dict['Gender'])
                language_and_gender_dict_by_speaker[pattern_dict['Speaker']] = {
                    'Language': pattern_dict['Language'],
                    'Gender': pattern_dict['Gender']
                    }
            except:
                print('File \'{}\' is not correct pattern file. This file is ignored.'.format(file))

            files_tqdm.update(1)

    with open(os.path.join(pattern_path, metadata_File.upper()).replace("\\", "/"), 'wb') as f:
        pickle.dump(new_metadata_dict, f, protocol= 4)

    if not eval:
        speaker_index_dict = {
            speaker: index
            for index, speaker in enumerate(sorted(set(speakers)))
            }
        yaml.dump(
            speaker_index_dict,
            open(hp.Speaker_Info_Path, 'w')
            )
            
        emotion_index_dict = {
            emotion: index
            for index, emotion in enumerate(sorted(set(emotions)))
            }
        yaml.dump(
            emotion_index_dict,
            open(hp.Emotion_Info_Path, 'w')
            )

        language_index_dict = {
            language: index
            for index, language in enumerate(sorted(set(languages)))
            }
        yaml.dump(
            language_index_dict,
            open(hp.Language_Info_Path, 'w')
            )

        gender_index_dict = {
            gender: index
            for index, gender in enumerate(sorted(set(genders)))
            }
        yaml.dump(
            gender_index_dict,
            open(hp.Gender_Info_Path, 'w')
            )

        yaml.dump(
            language_and_gender_dict_by_speaker,
            open(hp.Language_and_Gender_Info_by_Speaker_Path, 'w')
            )

    print('Metadata generate done.')

def Token_dict_Generate(tokens: Union[List[str], str]):
    tokens = ['<S>', '<E>', '<P>'] + sorted(list(tokens))        
    token_dict = {token: index for index, token in enumerate(tokens)}
    
    os.makedirs(os.path.dirname(hp.Token_Path), exist_ok= True)    
    yaml.dump(token_dict, open(hp.Token_Path, 'w', encoding='utf-8-sig'), allow_unicode= True)

    return token_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-hp", "--hyper_parameters", required=True, type= str)
    parser.add_argument("-selvas", "--selvas_path", required=False)
    parser.add_argument("-kss", "--kss_path", required=False)
    parser.add_argument("-aihub", "--aihub_path", required=False)

    parser.add_argument("-vctk", "--vctk_path", required=False)
    parser.add_argument("-libri", "--libri_path", required=False)
    parser.add_argument("-lj", "--lj_path", required=False)
    
    parser.add_argument("-soundworks", "--soundworks_path", required=False)

    parser.add_argument("-evalr", "--eval_ratio", default= 0.001, type= float)
    parser.add_argument("-evalm", "--eval_min", default= 1, type= int)
    parser.add_argument("-batch", "--batch_size", default= 64, required=False, type= int)
    parser.add_argument("-sample", "--sample", required=False, type= int)

    args = parser.parse_args()

    global hp
    hp = Recursive_Parse(yaml.load(
        open(args.hyper_parameters, encoding='utf-8'),
        Loader=yaml.Loader
        ))

    train_paths, eval_paths = [], []
    text_dict = {}
    pronunciation_dict = {}
    speaker_dict = {}
    emotion_dict = {}
    language_dict = {}
    gender_dict = {}
    dataset_dict = {}
    tag_dict = {}

    if not args.selvas_path is None:
        selvas_paths, selvas_text_dict, selvas_pronunciation_dict, selvas_speaker_dict, selvas_emotion_dict, selvas_language_dict, selvas_gender_dict = Selvas_Info_Load(path= args.selvas_path)
        selvas_paths = Split_Eval(selvas_paths, args.eval_ratio, args.eval_min)
        train_paths.extend(selvas_paths[0])
        eval_paths.extend(selvas_paths[1])
        text_dict.update(selvas_text_dict)
        pronunciation_dict.update(selvas_pronunciation_dict)
        speaker_dict.update(selvas_speaker_dict)
        emotion_dict.update(selvas_emotion_dict)
        language_dict.update(selvas_language_dict)
        gender_dict.update(selvas_gender_dict)
        dataset_dict.update({path: 'Selvas' for paths in selvas_paths for path in paths})
        tag_dict.update({path: '' for paths in selvas_paths for path in paths})

    if not args.kss_path is None:
        kss_paths, kss_text_dict, kss_pronunciation_dict, kss_speaker_dict, kss_emotion_dict, kss_language_dict, kss_gender_dict = KSS_Info_Load(path= args.kss_path)
        kss_paths = Split_Eval(kss_paths, args.eval_ratio, args.eval_min)
        train_paths.extend(kss_paths[0])
        eval_paths.extend(kss_paths[1])
        text_dict.update(kss_text_dict)
        pronunciation_dict.update(kss_pronunciation_dict)
        speaker_dict.update(kss_speaker_dict)
        emotion_dict.update(kss_emotion_dict)
        language_dict.update(kss_language_dict)
        gender_dict.update(kss_gender_dict)
        dataset_dict.update({path: 'KSS' for paths in kss_paths for path in paths})
        tag_dict.update({path: '' for paths in kss_paths for path in paths})

    if not args.aihub_path is None:
        aihub_paths, aihub_text_dict, aihub_pronunciation_dict, aihub_speaker_dict, aihub_emotion_dict, aihub_language_dict, aihub_gender_dict = AIHub_Info_Load(
            path= args.aihub_path,
            n_sample_by_speaker= args.sample
            )
        aihub_paths = Split_Eval(aihub_paths, args.eval_ratio, args.eval_min)
        train_paths.extend(aihub_paths[0])
        eval_paths.extend(aihub_paths[1])
        text_dict.update(aihub_text_dict)
        pronunciation_dict.update(aihub_pronunciation_dict)
        speaker_dict.update(aihub_speaker_dict)
        emotion_dict.update(aihub_emotion_dict)
        language_dict.update(aihub_language_dict)
        gender_dict.update(aihub_gender_dict)
        dataset_dict.update({path: 'AIHub' for paths in aihub_paths for path in paths})
        tag_dict.update({path: '' for paths in aihub_paths for path in paths})



    if not args.vctk_path is None:
        vctk_paths, vctk_text_dict, vctk_pronunciation_dict, vctk_speaker_dict, vctk_emotion_dict, vctk_language_dict, vctk_gender_dict = VCTK_Info_Load(path= args.vctk_path)
        vctk_paths = Split_Eval(vctk_paths, args.eval_ratio, args.eval_min)
        train_paths.extend(vctk_paths[0])
        eval_paths.extend(vctk_paths[1])
        text_dict.update(vctk_text_dict)
        pronunciation_dict.update(vctk_pronunciation_dict)
        speaker_dict.update(vctk_speaker_dict)
        emotion_dict.update(vctk_emotion_dict)
        language_dict.update(vctk_language_dict)
        gender_dict.update(vctk_gender_dict)
        dataset_dict.update({path: 'VCTK' for paths in vctk_paths for path in paths})
        tag_dict.update({path: '' for paths in vctk_paths for path in paths})

    if not args.libri_path is None:
        libri_paths, libri_text_dict, libri_pronunciation_dict, libri_speaker_dict, libri_emotion_dict, libri_language_dict, libri_gender_dict = Libri_Info_Load(
            path= args.libri_path,
            n_sample_by_speaker= args.sample
            )
        libri_paths = Split_Eval(libri_paths, args.eval_ratio, args.eval_min)
        train_paths.extend(libri_paths[0])
        eval_paths.extend(libri_paths[1])
        text_dict.update(libri_text_dict)
        pronunciation_dict.update(libri_pronunciation_dict)
        speaker_dict.update(libri_speaker_dict)
        emotion_dict.update(libri_emotion_dict)
        language_dict.update(libri_language_dict)
        gender_dict.update(libri_gender_dict)
        dataset_dict.update({path: 'Libri' for paths in libri_paths for path in paths})
        tag_dict.update({path: '' for paths in libri_paths for path in paths})

    if not args.lj_path is None:
        lj_paths, lj_text_dict, lj_pronunciation_dict, lj_speaker_dict, lj_emotion_dict, lj_language_dict, lj_gender_dict = LJ_Info_Load(path= args.lj_path)
        lj_paths = Split_Eval(lj_paths, args.eval_ratio, args.eval_min)
        train_paths.extend(lj_paths[0])
        eval_paths.extend(lj_paths[1])
        text_dict.update(lj_text_dict)
        pronunciation_dict.update(lj_pronunciation_dict)
        speaker_dict.update(lj_speaker_dict)
        emotion_dict.update(lj_emotion_dict)
        language_dict.update(lj_language_dict)
        gender_dict.update(lj_gender_dict)
        dataset_dict.update({path: 'LJ' for paths in lj_paths for path in paths})
        tag_dict.update({path: '' for paths in lj_paths for path in paths})

    if not args.soundworks_path is None:
        soundworks_paths, soundworks_text_dict, soundworks_pronunciation_dict, soundworks_speaker_dict, \
        soundworks_emotion_dict, soundworks_language_dict, soundworks_gender_dict = Basic_Info_Load(
            path= args.soundworks_path,
            dataset_label= 'Soundworks',
            language= {
                'Soundworks_Eng_Female': 'English',
                'Soundworks_Eng_Male': 'English',
                'Soundworks_Eng_Kid': 'English',
                'Soundworks_Kor_Female': 'English',
                'Soundworks_Kor_Male': 'Korean',
                'Soundworks_Kor_Kid': 'Korean',
                },
            gender= {
                'Soundworks_Eng_Female': 'Female',
                'Soundworks_Eng_Male': 'Male',
                'Soundworks_Eng_Kid': 'Male',
                'Soundworks_Kor_Female': 'Female',
                'Soundworks_Kor_Male': 'Male',
                'Soundworks_Kor_Kid': 'Male',
                }
            )
        soundworks_paths = Split_Eval(soundworks_paths, args.eval_ratio, args.eval_min)
        train_paths.extend(soundworks_paths[0])
        eval_paths.extend(soundworks_paths[1])
        text_dict.update(soundworks_text_dict)
        pronunciation_dict.update(soundworks_pronunciation_dict)
        speaker_dict.update(soundworks_speaker_dict)
        emotion_dict.update(soundworks_emotion_dict)
        language_dict.update(soundworks_language_dict)
        gender_dict.update(soundworks_gender_dict)
        dataset_dict.update({path: 'Soundworks' for paths in soundworks_paths for path in paths})
        tag_dict.update({path: '' for paths in soundworks_paths for path in paths})

    # if len(train_paths) == 0 or len(eval_paths) == 0:
    #     raise ValueError('Total info count must be bigger than 0.')

    logging.info('Sorting...')
    train_paths, eval_paths = sorted(train_paths), sorted(eval_paths)
    logging.info('Sorting...Done')

    logging.info('Token dict generating')
    tokens = set()
    for phonemes in pronunciation_dict.values():
        tokens = tokens.union(set(phonemes))
    tokens = sorted(list(tokens))
    token_dict = Token_dict_Generate(tokens= tokens)
    logging.info('Token dict generating...Done')

    for index in tqdm(range(0, len(train_paths), args.batch_size)):
        batch_paths = train_paths[index:index + args.batch_size]
        Pattern_File_Generate(
            paths= batch_paths,
            speakers= [speaker_dict[path] for path in batch_paths],
            emotions= [emotion_dict[path] for path in batch_paths],
            languages= [language_dict[path] for path in batch_paths],
            genders= [gender_dict[path] for path in batch_paths],
            datasets= [dataset_dict[path] for path in batch_paths],
            texts= [text_dict[path] for path in batch_paths],
            pronunciations= [pronunciation_dict[path] for path in batch_paths],
            tags= [tag_dict[path] for path in batch_paths],
            eval= False
            )
    for index in tqdm(range(0, len(eval_paths), args.batch_size)):
        batch_paths = eval_paths[index:index + args.batch_size]
        Pattern_File_Generate(
            paths= batch_paths,
            speakers= [speaker_dict[path] for path in batch_paths],
            emotions= [emotion_dict[path] for path in batch_paths],
            languages= [language_dict[path] for path in batch_paths],
            genders= [gender_dict[path] for path in batch_paths],
            datasets= [dataset_dict[path] for path in batch_paths],
            texts= [text_dict[path] for path in batch_paths],
            pronunciations= [pronunciation_dict[path] for path in batch_paths],
            tags= [tag_dict[path] for path in batch_paths],
            eval= True
            )

    Metadata_Generate()
    Metadata_Generate(eval= True)

# python -m Exp5000.Pattern_Generator -hp Exp5000/Hyper_Parameters.yaml -lj D:\Rawdata\LJSpeech
# python -m Exp5003.Pattern_Generator -hp Exp5003/Hyper_Parameters.yaml -libri F:\Rawdata\LibriTTS_Clean