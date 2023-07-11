#!/usr/bin/env python3

# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import io
import logging
import os
import tarfile
import time
import multiprocessing

import torch
import torchaudio
import torchaudio.backend.sox_io_backend as sox
from pydub import AudioSegment
import pydub
import wave
import numpy as np
# import torchaudio.backend.soundfile_backend as sox

AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])


def m4a2wav(input_file):
   audio = AudioSegment.from_file(input_file, format="m4a")


def write_tar_file(data_list,
                   tar_file,
                   resample=16000,
                   index=0,
                   total=1):
    logging.info('Processing {} {}/{}'.format(tar_file, index, total))
    read_time = 0.0
    save_time = 0.0
    write_time = 0.0
    with tarfile.open(tar_file, "w") as tar:
        prev_wav = None
        for item in data_list:
            
            # if len(item)==3:
            #     key, txt, wav = item
            # elif len(item)==4:
            #     key, txt, wav ,lang = item
            # elif len(item)==5:
            #     key, txt, wav, start, end = item
            # elif len(item)==6:
            #     key, txt, wav, start, end ,lang= item
            try:
                if len(item)==6 or len(item)==5:
                    no_segments=False
                else:
                    no_segments=True

                if no_segments:
                    if len(item)==4:
                        key, txt, wav ,lang= item
                    else:
                        key, txt, wav = item
                else:
                    if len(item)==6:
                        key, txt, wav, start, end ,lang= item
                    else:
                        key, txt, wav, start, end = item

                suffix = wav.split('.')[-1]
                # assert suffix in AUDIO_FORMAT_SETS
                if no_segments:
                    # ts = time.time()
                    # with open(wav, 'rb') as fin:
                    #     data = fin.read()
                    
                    # read_time += (time.time() - ts)


                    ts = time.time()
                    # with open(wav, 'rb') as fin:
                    #     data = fin.read()
                    waveforms, sample_rate = sox.load(wav, normalize=True)
                    audio = waveforms[:1, :]

                    read_time += (time.time() - ts)
                    ts = time.time()
                    f = io.BytesIO()
                    sox.save(f, audio, resample, format="wav", bits_per_sample=16)
                    # Save to wav for segments file
                    suffix = "wav"
                    f.seek(0)
                    data = f.read()
                    save_time += (time.time() - ts)


                else:
                    if wav != prev_wav:
                        ts = time.time()
                    
                        if os.path.isfile(wav):
                            waveforms, sample_rate = sox.load(wav, normalize=True)
                        elif os.path.isfile(wav.replace(".wav",".mp3")):
                            waveforms, sample_rate = sox.load(wav.replace(".wav",".mp3"), normalize=True)
                        elif os.path.isfile(wav.replace(".wav",".WAV")):
                            waveforms, sample_rate = sox.load(wav.replace(".wav",".WAV"), normalize=True)
                        elif os.path.isfile(wav.replace(".wav",".m4a")):
                            wav=wav.replace(".wav",".m4a")

                            audio = pydub.AudioSegment.from_file(wav, format="m4a")
                            audio_array = np.array(audio.get_array_of_samples())

                            # 将numpy数组转换成PyTorch张量
                            waveforms = torch.from_numpy(audio_array)
                            sample_rate = audio.frame_rate
                        else:


                            waveforms, sample_rate = sox.load(wav, normalize=True)
            


                        read_time += (time.time() - ts)
                        prev_wav = wav
                    start = int(start * sample_rate)
                    end = int(end * sample_rate)
                    if len(waveforms.shape )==1:
                        waveforms=waveforms.unsqueeze(0)
                        audio = waveforms[:1, start:end]
                    else:
                        if waveforms.shape[0]==1:
                            audio = waveforms[:1, start:end]
                        else:
                            waveforms = waveforms.to(torch.float32)
                            waveforms = waveforms.sum(0).unsqueeze(0)
                            audio = waveforms[:1, start:end]

                    # resample
                    if sample_rate != resample:
                        if not audio.is_floating_point():
                            # normalize the audio before resample
                            # because resample can't process int audio
                            audio = audio / (1 << 15)
                            audio = torchaudio.transforms.Resample(
                                sample_rate, resample)(audio)
                            audio = (audio * (1 << 15)).short()
                        else:
                            audio = torchaudio.transforms.Resample(
                                sample_rate, resample)(audio)

                    ts = time.time()
                    f = io.BytesIO()
                    sox.save(f, audio, resample, format="wav", bits_per_sample=16)
                    # Save to wav for segments file
                    suffix = "wav"
                    f.seek(0)
                    data = f.read()
                    save_time += (time.time() - ts)

                assert isinstance(txt, str)
                ts = time.time()
                txt_file = key + '.txt'
                txt = txt.encode('utf8')
                txt_data = io.BytesIO(txt)
                txt_info = tarfile.TarInfo(txt_file)
                txt_info.size = len(txt)
                tar.addfile(txt_info, txt_data)

                lang_file = key + '.lang'
                lang = lang.encode('utf8')
                lang_data = io.BytesIO(lang)
                lang_info = tarfile.TarInfo(lang_file)
                lang_info.size = len(lang)
                tar.addfile(lang_info, lang_data)

                wav_file = key + '.' + suffix
                wav_data = io.BytesIO(data)
                wav_info = tarfile.TarInfo(wav_file)
                wav_info.size = len(data)
                tar.addfile(wav_info, wav_data)
                write_time += (time.time() - ts)
            except Exception as e:
                print(e)
                continue
        print('read {} save {} write {}'.format(read_time, save_time,
                                                       write_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num_utts_per_shard',
                        type=int,
                        default=1000,
                        help='num utts per shard')
    parser.add_argument('--num_threads',
                        type=int,
                        default=1,
                        help='num threads for make shards')
    parser.add_argument('--prefix',
                        default='shards',
                        help='prefix of shards tar file')

    parser.add_argument('--resample',
                        type=int,
                        default=16000,
                        help='segments file')
    parser.add_argument('--data_list', help='data.list')

    parser.add_argument('shards_dir', help='output shards dir')
    parser.add_argument('shards_list', help='output shards list file')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    torch.set_num_threads(1)


    data = []
    with open(args.data_list, 'r', encoding='utf8') as fin:
        for line in fin:
            line=eval(line.strip())
            
            if "lang" in line.keys():
                if "start" in line.keys():
                     data.append((line["key"], line["txt"],line["wav"],line["start"],line["end"],line["lang"] ))   #(line["key"], line["txt"],line["wav"],line["start"],line["end"],line["lang"] )
                else:
                    data.append((line["key"], line["txt"],line["wav"],line["lang"] ))   #(line["key"], line["txt"],line["wav"],line["start"],line["end"],line["lang"] )
            else:
                if "start" in line.keys():
                     data.append((line["key"], line["txt"],line["wav"],line["start"],line["end"] ))   #(line["key"], line["txt"],line["wav"],line["start"],line["end"],line["lang"] )
                else:
                    data.append((line["key"], line["txt"],line["wav"] ))   #(line["key"], line["txt"],line["wav"],line["start"],line["end"],line["lang"] )




    num = args.num_utts_per_shard
    chunks = [data[i:i + num] for i in range(0, len(data), num)]
    os.makedirs(args.shards_dir, exist_ok=True)

    # Using thread pool to speedup
    pool = multiprocessing.Pool(processes=args.num_threads)
    shards_list = []
    tasks_list = []
    num_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        tar_file = os.path.join(args.shards_dir,
                                '{}_{:09d}.tar'.format(args.prefix, i))
        shards_list.append(tar_file)
        # pool.apply(
        #     write_tar_file,
        #     (chunk, tar_file,args.resample, i, num_chunks ))
        pool.apply_async(
            write_tar_file,
            (chunk, tar_file,args.resample, i, num_chunks ))

    pool.close()
    pool.join()

    with open(args.shards_list, 'w', encoding='utf8') as fout:
        for name in shards_list:
            fout.write(name + '\n')
