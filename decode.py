# !pip install torch==1.12.1 accelerate torchaudio datasets
# !pip install --upgrade transformers
# !pip install git+https://github.com/huggingface/transformers.git


from datasets import load_dataset, DatasetDict


import torch
from torch.utils.data import DataLoader
import os


import datetime

import sys 
wav_file = sys.argv[1]
# wav_file="/home/yuhang001/w2023/wenet-eng-id/wenet/data/yt_test/data.list"
wav_name=wav_file.split("/")[-2]

def avg_parameters2():
    
    models = []
    with open("/home/yuhang001/eng_whisper/whisper-small-eng3/need_path") as f :
        for i in f:
            i=i.strip()
            models.append(    torch.load(i +"/pytorch_model.bin")  )


    avg_model = torch.load("/home/yuhang001/eng_whisper/whisper-small-eng3/checkpoint-avg/pytorch_model.bin")
    for key in avg_model:
        avg_model[key] = torch.true_divide(sum([_[key] for _ in models]), len(models))
    torch.save(avg_model,  "/home/yuhang001/eng_whisper/whisper-small-eng3/checkpoint-avg/pytorch_model.bin")


# avg_parameters2()


from transformers import Wav2Vec2ForCTC, AutoProcessor
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
model_id = "facebook/mms-1b-all"
# target_lang="ind"
target_lang="ind"

# model = Wav2Vec2ForCTC.from_pretrained(model_id ,target_lang=target_lang,  ).to("cuda")

model = Wav2Vec2ForCTC.from_pretrained("mms-1b-all-fine-tune3/checkpoint-11200" ,  ).to("cuda")
# model.load_adapter("ind")
processor = AutoProcessor.from_pretrained(model_id ,target_lang=target_lang , )
processor.tokenizer.set_target_lang("ind")
model.load_adapter("ind")

from wenet.dataset.dataset import Dataset
from wenet.dataset.dataset import Dataset_dev ,Dataset_test
dev_data=wav_file
data_type="shard" #shard
dev_dataset = Dataset_test(data_type,dev_data,processor)


def to_cuda(data_dict):
    """将字典内的所有张量放在 CUDA 上"""
    for key in data_dict:
        if isinstance(data_dict[key], torch.Tensor):
            data_dict[key] = data_dict[key].to('cuda')
    return data_dict



ref=open("./decoder_"+wav_name+"_ref","w")
reg=open("./decoder_"+wav_name+"_reg","w")

with torch.no_grad():
    index=1
    for all_inputs in dev_dataset:

        key=all_inputs['key']
        inputs=to_cuda(all_inputs['input_values'])
        label=all_inputs['label']
    

        ref.writelines("utt_" + str(index).zfill(5)  + " " +str(label)+"\n")
        
        outputs = model(**inputs).logits
        ids = torch.argmax(outputs, dim=-1)[0]
        transcription = processor.decode(ids)



        print( str(label) )
        print( str(transcription) )
        now = datetime.datetime.now()
        print("Current date and time: ")
        print(now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
        reg.writelines("utt_" + str(index).zfill(5)  + " " +str(transcription)+"\n")
        print(index)
        index=index+1

ref.close()
reg.close()





    




exit()








# #任务，写一个，可以给到transformer 的dataset
from transformers import AutoProcessor
# model = WhisperForConditionalGeneration.from_pretrained("/home/yuhang001/eng_whisper/whisper-small-eng/checkpoint-900")
# processor = AutoProcessor.from_pretrained("/home/yuhang001/eng_whisper/whisper-small-eng")


processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("/home/yuhang001/id_whisper/whisper-small-eng6_200_end/checkpoint-30000").to("cuda")
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to("cuda")
# model = WhisperForConditionalGeneration.from_pretrained("/home/yuhang001/eng_whisper/whisper-small-eng4/checkpoint-800").to("cuda")
#model = WhisperForConditionalGeneration.from_pretrained("/home/yuhang001/eng_whisper/whisper-small-eng5_add_noise/checkpoint-150").to("cuda")

# model = WhisperForConditionalGeneration.from_pretrained("/home/yuhang001/eng_man_whisper/whisper-small-eng6_200/checkpoint-700").to("cuda")
# model = WhisperForConditionalGeneration.from_pretrained("/home/yuhang001/eng_man_whisper/whisper-small-eng6_200_end/checkpoint-3000").to("cuda")
# model = WhisperForConditionalGeneration.from_pretrained("/home/yuhang001/eng_man_whisper/whisper-small-eng6_200_end/checkpoint-1600").to("cuda")

# processor = WhisperProcessor.from_pretrained("/home/yuhang001/eng_whisper/whisper-small-eng")
# model = WhisperForConditionalGeneration.from_pretrained("/home/yuhang001/eng_whisper/whisper-small-eng/checkpoint-4050").to("cuda")
# model = WhisperForConditionalGeneration.from_pretrained("/home/yuhang001/eng_man_whisper/whisper-small-eng6_200_end/checkpoint-5200").to("cuda")

forced_decoder_ids = processor.get_decoder_prompt_ids(language="indonesian", task="transcribe")

# forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")

# #common voice 解码
# ref=open("./decoder_clean_ref","w")
# reg=open("./decoder_clean_reg","w")
# from datasets import load_dataset
# from transformers import WhisperForConditionalGeneration, WhisperProcessor
# import torch
# from evaluate import load
# librispeech_test_clean = load_dataset("librispeech_asr", 'clean', split="test", streaming=True)
# index=1
# for batch in iter(librispeech_test_clean):
#     audio = batch["audio"]
#     input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
#     batch["reference"] = processor.tokenizer._normalize(batch['text'])
#     ref.writelines("utt_" + str(index).zfill(5)  + " " +batch["reference"]+"\n")

#     with torch.no_grad():
#         predicted_ids = model.generate(input_features.to("cuda")          ,forced_decoder_ids=forced_decoder_ids  )[0]
#     transcription = processor.decode(predicted_ids)
#     batch["prediction"] = processor.tokenizer._normalize(transcription)
#     reg.writelines("utt_" + str(index).zfill(5)  + " " +batch["prediction"]+"\n")
#     print(index)
#     index=index+1
# ref.close()
# reg.close()










#自定义dataset解码
ref=open("./decoder_"+wav_name+"ref","w")
reg=open("./decoder_"+wav_name+"reg","w")
index=1
for i in dev_dataset:
    ref_utt=processor.batch_decode(torch.tensor(i["labels"]).unsqueeze(0).to("cuda")   , skip_special_tokens=True)[0] 
    ref_utt=processor.tokenizer._normalize(ref_utt)

    ref.writelines("utt_" + str(index).zfill(5)  + " " +str(ref_utt)+"\n")
    input=i["input_features"]

    with torch.no_grad():
        generated_ids = model.generate(inputs=input.to("cuda") ,forced_decoder_ids=forced_decoder_ids  )  #forced_decoder_ids=forced_decoder_ids
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    transcription=processor.tokenizer._normalize(transcription)
    
    print(str(transcription))
    now = datetime.datetime.now()
    print("Current date and time: ")
    print(now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
    reg.writelines("utt_" + str(index).zfill(5)  + " " +str(transcription)+"\n")
    print(index)
    index=index+1
ref.close()
reg.close()


# cat /home/yuhang001/eng_whisper/decoder_ntu_ref | awk '{printf "utt_" NR " "; print $0}' > u1
# cat /home/yuhang001/eng_whisper/decoder_ntu_reg | awk '{printf "utt_" NR " "; print $0}' > u2
# python tools/compute-wer.py --char 1 --v 1  decoder_yt_testref  decoder_yt_testreg
#wer 37.6


#100 step fine-turn: 30
#150 step  : 28

#200 step  24.83
#1950 step 17.69  ,2600 18.40, 17.21   ,14.94


# 原本的是  11.46  ; 4.18 ;3.45
#test_clean  5.61  ;4.30 ;








# #自定义dataset解码_bak
# ref=open("./decoder_ntu_ref","w")
# reg=open("./decoder_ntu_reg","w")
# index=1
# for i in dev_dataset:
#     ref_utt=processor.batch_decode(torch.tensor(i["labels"]).unsqueeze(0).to("cuda")   , skip_special_tokens=True)[0] 
#     ref.writelines("utt_" + str(index).zfill(5)  + " " +str(ref_utt)+"\n")
#     input=i["input_features"]
#     input_features = input["input_features"][0]
#     with torch.no_grad():
#         generated_ids = model.generate(inputs=torch.tensor(input_features).unsqueeze(0).to("cuda")  ,forced_decoder_ids=forced_decoder_ids)
#     transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     reg.writelines("utt_" + str(index).zfill(5)  + " " +str(transcription)+"\n")
#     print(index)
#     index=index+1
# ref.close()



# librispeech_clean
# old model:
    # comvoice: 3.45
    # self_dataset: 3.45
#my model:
    # comvoice: 
    # self_dataset: 4.3   , 4.96(4400step)


#ntu
#old model: 30.53
#my model: 14.94     ,14.49 ;14.00 4000step    13.63 4400step  13.31  12.77 ,12.55(avg)


#ny-eng
#old model :  60.74
#self_model:  30.06


#avg_model:
#ntu -conversition结果  ： 11.08
#text clean 4.03
