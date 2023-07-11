from datasets import load_dataset, DatasetDict



from torch.utils.data import DataLoader

from wenet.dataset.dataset import Dataset
from wenet.dataset.dataset import Dataset_dev
import numpy as np


train_data="/home/yuhang001/w2023/wenet-eng-id/wenet/data/5-time-id-imda-500/data.list"
dev_data="/home/yuhang001/w2023/wenet-eng-id/wenet/data/yt_test_small_10/data.list"

data_type="shard"
#如果存在lang,那就处理lang



import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"


# #获取tokenizer
# from transformers import Wav2Vec2CTCTokenizer
# mms_adapter_repo ="facebook/mms-1b-all"
# tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(mms_adapter_repo)
# print(tokenizer.vocab.keys())

# #获取特征提取器
# from transformers import Wav2Vec2FeatureExtractor
# feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
from transformers import Wav2Vec2Processor
# processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

#直接复用之前的processor和model
from transformers import Wav2Vec2ForCTC, AutoProcessor
target_lang="eng"
# target_lang="cmn-script_simplified"
model_id = "facebook/mms-1b-all"
processor = AutoProcessor.from_pretrained(model_id ,target_lang=target_lang )
# ,target_lang=target_lang
# from office


train_dataset = Dataset(data_type,train_data,processor)
dev_dataset = Dataset_dev(data_type,dev_data,processor)



import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Any
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [ {"input_values":feature["input_values"]["input_values"][0]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)


from evaluate import load
wer_metric = load("wer")
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}



# # model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
# model = WhisperForConditionalGeneration.from_pretrained("/home/yuhang001/id_whisper/whisper-small-eng6_200_end/checkpoint-10000")
# # model = WhisperForConditionalGeneration.from_pretrained("/home/yuhang001/eng_whisper/whisper-small-eng5_add_noise/checkpoint-800")



# model = Wav2Vec2ForCTC.from_pretrained(
#     "facebook/mms-1b-all",
#     attention_dropout=0.1,
#     hidden_dropout=0.1,
#     feat_proj_dropout=0.1,
#     layerdrop=0.1,
#     pad_token_id=processor.tokenizer.pad_token_id,
#     vocab_size=len(processor.tokenizer),
#     ignore_mismatched_sizes=True,
#     target_lang=target_lang,
# )

model = Wav2Vec2ForCTC.from_pretrained(
    model_id,
    attention_dropout=0.0,
    hidden_dropout=0.0,
    feat_proj_dropout=0.0,
    layerdrop=0.0,
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
    ignore_mismatched_sizes=True,
    target_lang=target_lang,
)



# target_lang=target_lang,
# model.init_adapter_layers()
#ignore_mismatched_sizes=True,
model.freeze_base_model()

adapter_weights = model._get_adapters()
for param in adapter_weights.values():
    param.requires_grad = True


from transformers import TrainingArguments

# training_args = TrainingArguments(
#     output_dir="./mms-1b-all-fine-tune",
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=64,
#     evaluation_strategy="steps",
#     gradient_checkpointing=True,
#     dataloader_num_workers=6,
#     fp16=True,
#     save_steps=100,
#     eval_steps=100,
#     logging_steps=100,
#     learning_rate=1e-4,
#     warmup_steps=100,
#     push_to_hub=False,
#     greater_is_better=False,
#     metric_for_best_model="wer",
#     report_to=["tensorboard"],
#     load_best_model_at_end=True,
#     per_device_eval_batch_size=2,

# )


training_args = TrainingArguments(
  output_dir="./mms-1b-all-fine-tune_eng",
  per_device_train_batch_size=4,
  evaluation_strategy="steps",
  num_train_epochs=10,
  gradient_checkpointing=True,
  fp16=False,
  save_steps=400,
  eval_steps=200,
  logging_steps=200,
  learning_rate=2e-3,
  warmup_steps=1000,
  push_to_hub=False,
  max_grad_norm=1,
  save_total_limit=2,
  
)

training_args.set_training(batch_size=4,max_steps=200000,gradient_accumulation_steps=4,learning_rate=2e-3)

from transformers import Trainer

trainer = Trainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

processor.save_pretrained(training_args.output_dir)



trainer.train()


from safetensors.torch import save_file as safe_save_file
from transformers.models.wav2vec2.modeling_wav2vec2 import WAV2VEC2_ADAPTER_SAFE_FILE
import os

adapter_file = WAV2VEC2_ADAPTER_SAFE_FILE.format(target_lang)
adapter_file = os.path.join(training_args.output_dir, adapter_file)

safe_save_file(model._get_adapters(), adapter_file, metadata={"format": "pt"})
