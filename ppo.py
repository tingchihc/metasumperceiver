from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler

from loguru import logger
import evaluate

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO    
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable
    model_name: Optional[str] = field(default="", metadata={"help": "the model name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    merge_model_adapter: Optional[bool] = field(default=False, metadata={"help": "the learning rate"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    batch_size=2,
    mini_batch_size=2,
    optimize_cuda_cache=True,
)

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
# sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": config.mini_batch_size}


# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(config, dataset_name="anli"):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # load anli with datasets
    ds = load_dataset(dataset_name, split="train_r1[:1000]")

    label_map = {
        0: "entailment",
        1: "neutral",
        2: "contradiction"
    }
    def tokenize(sample):
        # input query: |premise|{}|label|{}|claim|{}
        
        prompt = '{}{}{}'.format(
            '### Premise:{}{}'.format(sample['premise'], "\n"),
            '### Label:{}{}'.format(label_map[sample['label']], "\n"),
            '### Original:{}{}'.format(sample['hypothesis'], "\n"),
            # '### Claim: {}{}'.format(sample['hypothesis'], "\n"),
            # '### Manipulated: {}{}'.format(sample['hypothesis'], "\n"),
        )

        sample["input_ids"] = tokenizer.encode(prompt)
        sample["query"] = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(config)

logger.info('query: {}', dataset[0]['query'])

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# set seed before initializing value head for deterministic eval
set_seed(config.seed)


"""### Apply LoRA
Here comes the magic with `peft`! Let's load a `PeftModel` and specify that we are going to use low-rank adapters (LoRA) using `get_peft_model` utility function from `peft`.
"""


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.debug(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Now let's build the model, the reference model, and the tokenizer.
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    load_in_8bit=True,
    device_map="balanced",
    max_memory={0: "800MB", 1: "800MB"},
    peft_config=lora_config,
)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)

print_trainable_parameters(model)

# GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
# only for this model.
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)
# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(
    config, model, ref_model=None, tokenizer=tokenizer, dataset=dataset, data_collator=collator, optimizer=optimizer
)

# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug


entailment_pipe = pipeline("entailment-classification",
                           model="Tverous/entailment-classification", device=device, trust_remote_code=True)
rouge = evaluate.load('rouge')

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "eos_token_id": tokenizer.eos_token_id,
}
# TODO
output_min_length = 32
output_max_length = 512
output_length_sampler = LengthSampler(output_min_length, output_max_length)

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    model.gradient_checkpointing_disable()
    model.pretrained_model.config.use_cache = True
    # Get response from Causal LM
    response_tensors = []
    for query in query_tensors:
        
        gen_len = output_length_sampler()
        generation_kwargs["max_new_tokens"] = gen_len
        
        response = ppo_trainer.generate(query, **generation_kwargs)
        
        logger.info('premise: {}', tokenizer.decode(response.squeeze(), skip_special_tokens=True).split("\n")[0].split(":")[1])
        logger.info('label: {}', tokenizer.decode(response.squeeze(), skip_special_tokens=True).split("\n")[1].split(":")[1])
        logger.info('original: {}', tokenizer.decode(response.squeeze(), skip_special_tokens=True).split("\n")[2].split(":")[1])
        logger.info('claim: {}', tokenizer.decode(response.squeeze(), skip_special_tokens=True).split("\n")[3].split(":")[1])
        logger.info('-----------------------------------------------')
        # logger.info('response: {}', tokenizer.decode(response.squeeze(), skip_special_tokens=True).split("\n")[2].split(":")[1])
        
        response_tensors.append(response.squeeze())
    batch["premise"] = [tokenizer.decode(r.squeeze(), skip_special_token=True).split("\n")[0].split(":")[1] for r in response_tensors]
    batch["label"] = [tokenizer.decode(r.squeeze(), skip_special_token=True).split("\n")[1].split(":")[1] for r in response_tensors]
    batch["original"] = [tokenizer.decode(r.squeeze(), skip_special_token=True).split("\n")[2].split(":")[1] for r in response_tensors]
    batch["claim"] = [tokenizer.decode(r.squeeze(), skip_special_token=True).split("\n")[3].split(":")[1] for r in response_tensors]

    rewards = []
    for premise, label, original, claim in zip(batch["premise"], batch["label"], batch["original"], batch["claim"]):
        pipe_outputs = entailment_pipe(premise, claim)
        
        fg_score = 0.1*(pipe_outputs[label] - 0.5*sum(pipe_outputs.values()))
        logger.info('fg_score: {}', fg_score)
        rouge_score = 10*(rouge.compute(predictions=[claim], references=[original])['rouge1'])
        logger.info('rouge score: {}', rouge_score)
        
        rewards.append(torch.tensor(fg_score + rouge_score))

    logger.info('rewards: {}', rewards)
    
    # Run PPO step
    model.gradient_checkpointing_enable()
    model.pretrained_model.config.use_cache = False

    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)


model.push_to_hub(f"{script_args.model_name}-ppo-entailment")