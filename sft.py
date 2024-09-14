from datasets import load_dataset, DatasetDict, Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig

import evaluate
import wandb
from wandb.sdk.integration_utils.data_logging import ValidationDataLogger
from loguru import logger

# %env WANDB_ENTITY=your-username/your-team-name
# %env WANDB_PROJECT=your-project-name


MAX_LENGTH = 512
MODEL_NAME = ""


def convert_to_features(example_batch):
    input_encodings = tokenizer.batch_encode_plus(
        example_batch['prompt'], padding='max_length', max_length=MAX_LENGTH, truncation=True)

    encodings = {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': input_encodings['input_ids'],
    }

    return encodings


def combine_premise_and_hypothesis(example):
    label_map = {
        0: "entailment",
        1: "neutral",
        2: "contradiction"
    }

    return {
        'prompt': '{}{}{}'.format(
            '### Premise:{}{}'.format(example['premise'], "\n"),
            '### Label:{}{}'.format(label_map[example['label']], "\n"),
            '### Claim:{}{}'.format(example['hypothesis'], "\n"),
            # '### Manipulated : {}{}'.format(example['hypothesis'], "\n"),
        )
    }


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['query'])):
        output_texts.append(example['query'][i])
    return output_texts


logger.info("Loading model and tokenizer")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# new_tokens = ['<|premise|>', '<|label|>', '<|claim|>', '<|manipulated_claim|>']
# new_tokens = set(new_tokens) - set(tokenizer.get_vocab().keys())
# tokenizer.add_tokens(list(new_tokens))
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

logger.info('eos token: {}', tokenizer.eos_token)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    use_cache=False,
    # load_in_8bit=True,
    # device_map="auto",
)
model.config.eos_token_id = tokenizer.eos_token_id
# model.config.end_token_id = tokenizer.eos_token_id
model.config.pad_token_id = model.config.eos_token_id
model.resize_token_embeddings(len(tokenizer))

logger.info("Loading datasets")
# Create the comparisons datasets
train_dataset = load_dataset("anli", split="train_r3[:20000]")
eval_dataset = load_dataset("anli", split="dev_r3[:30]")
train_dataset = train_dataset.map(combine_premise_and_hypothesis)
eval_dataset = eval_dataset.map(combine_premise_and_hypothesis)

logger.info('prompt: {}', train_dataset[0]['prompt'])

# upload dataset to huggingface
# logger.info("Uploading datasets to HuggingFace Datasets Hub")
# train_dataset.push_to_hub("Tverous/anli", split="train")
# eval_dataset.push_to_hub("Tverous/anli", split="eval")

train_dataset = train_dataset.map(convert_to_features, batched=True)
eval_dataset = eval_dataset.map(convert_to_features, batched=True)
train_dataset = train_dataset.map(lambda x: {"query": tokenizer.decode(
    x["input_ids"], skip_special_token=True).strip()}, batched=False)
eval_dataset = eval_dataset.map(lambda x: {"query": tokenizer.decode(
    x["input_ids"], skip_special_token=True).strip()}, batched=False)

logger.info("query: {}", train_dataset[0]["query"])

train_dataset = train_dataset.remove_columns(['label'])
eval_dataset = eval_dataset.remove_columns(['label'])

# For W&B visualization
validation_inputs = eval_dataset.remove_columns(
    ['hypothesis', 'prompt', 'query', 'uid', 'reason', 'input_ids', 'attention_mask', 'labels'])
validation_targets = [x for x in eval_dataset['hypothesis']]
validation_logger = ValidationDataLogger(
    inputs=validation_inputs[:],
    targets=validation_targets
)
# validation_logger.log_predictions("test")

rouge = evaluate.load("rouge")


def compute_metrics(eval_preds):
    predictions, labels = eval_preds

    logger.debug(predictions)
    logger.debug(labels)

    # pred_str = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

    logger.debug('log predictions')

    # log predictions
    validation_logger.log_predictions(label_str)

    logger.debug('write to wandb')

    score = rouge.compute(predictions=label_str, references=label_str)

    logger.debug('score: {}', score)

    return score


training_args = TrainingArguments(
    output_dir="claim-gen",
    auto_find_batch_size=True,
    warmup_steps=100,

    logging_steps=1,

    evaluation_strategy='steps',
    eval_steps=1000,
    eval_accumulation_steps=1,

    save_strategy='steps',
    save_steps=1000,
    push_to_hub=True,
    hub_model_id="",
    # report_to='wandb',
    # run_name='trlxxx'  # name of the W&B run
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# get trainer
trainer = SFTTrainer(
    model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # dataset_text_field="prompt",
    formatting_func=formatting_prompts_func,
    max_seq_length=MAX_LENGTH,
    # compute_metrics=compute_metrics,

    # peft_config=peft_config,

    # Training arguments
    args=training_args
)

# train
trainer.train()

trainer.push_to_hub("")

wandb.finish()
