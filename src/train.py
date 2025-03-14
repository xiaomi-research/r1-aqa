import logging
from dataclasses import dataclass, field
from typing import Optional

import transformers
from transformers import HfArgumentParser
from trl import GRPOConfig

from trainer.grpo_trainer import GRPOTrainer
from utils.rewards import accuracy_reward, format_reward
from dataset.dataset import AudioDataset


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    config_path: Optional[str] = field(default=None, metadata={"help": "config path"})
    model_name_or_path : Optional[str] = field(default=None, metadata={"help": "model name or path"})
    out_dir: Optional[str] = field(default=None, metadata={"help": "output dir for model"})
    data_file: Optional[str] = field(default=None, metadata={"help": "train data file"})
    use_wandb: Optional[str] = field(default="flase", metadata={"help": "whether use wandb to report logs"})

    def __post_init__(self):
        if self.config_path is None:
            raise ValueError("config path should not none")


def main():
    parser = HfArgumentParser(DataTrainingArguments)
    data_args = parser.parse_args_into_dataclasses()[0]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    transformers.logging.set_verbosity_info()
    logging.info(data_args)

    reward_funcs_registry = {"accuracy": accuracy_reward, "format": format_reward}
    reward_funcs = [reward_funcs_registry["accuracy"], reward_funcs_registry["format"]]

    train_dataset = AudioDataset(data_args.data_file)

    training_args = GRPOConfig(
        seed=42,
        data_seed=42,
        output_dir=data_args.out_dir, 
        deepspeed=data_args.config_path, 
        max_prompt_length=512, 
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=2, 
        logging_steps=1, 
        bf16=True,
        report_to="wandb" if data_args.use_wandb == "true" else [],
        gradient_checkpointing=False, 
        num_train_epochs=2, 
        max_steps=1000,
        run_name="AQA-GRPO", 
        save_steps=100, 
        save_only_model=True, 
        temperature=1.0,
        num_generations=8)
    
    trainer = GRPOTrainer(
        model=data_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None)

    trainer.train()
    trainer.save_model(data_args.out_dir)


if __name__ == "__main__":
    main()
