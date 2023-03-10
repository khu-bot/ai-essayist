import argparse
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies import DeepSpeedStrategy
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from essayist.data import Datum, LanguageModelingDataset, load_jsonl_data
from essayist.task import LanguageModeling
from essayist.utils import get_logger, normalize_text

# fmt: off
parser = argparse.ArgumentParser(prog="train")

g = parser.add_argument_group("Train Parameter")
g.add_argument("--model", type=str, required=True, help="huggingface model")
g.add_argument("--dataset-dir", type=str, required=True, help="dataset name")
g.add_argument("--tokenizer", type=str, help="huggingface tokenizer, use model by default")
g.add_argument("--batch-size", type=int, default=3, help="training batch size")
g.add_argument("--valid-batch-size", type=int, default=4, help="validation batch size")
g.add_argument("--accumulate-grad-batches", type=int, default=1, help="the number of gradident accumulation steps")
g.add_argument("--prompt-max-length", type=int, default=128, help="prompt max length")
g.add_argument("--min-content-char", type=int, default=100, help="filter less than this number of content")
g.add_argument("--max-length", type=int, default=512, help="max sequence length")
g.add_argument("--epochs", type=int, default=1, help="the number of training epochs")
g.add_argument("--learning-rate", type=float, default=5e-5, help="learning rate")
g.add_argument("--weight-decay", type=float, default=0.01, help="weight decay")
g.add_argument("--warmup-rate", type=float, default=0.06, help="warmup step rate")
g.add_argument("--seed", type=int, default=42, help="random seed")
g.add_argument("--precision", type=int, default=32, choices=[16, 32])
g.add_argument("--strategy", type=str, default="ddp", choices=["deepspeed", "ddp"])
g.add_argument("--use-token-type-ids", action="store_true")

g = parser.add_argument_group("Personal Options")
g.add_argument("--output-dir", type=str, help="output directory path to save artifacts")
g.add_argument("--gpus", type=int, help="the number of gpus, use all devices by default")
g.add_argument("--logging-interval", type=int, default=10, help="logging interval")
g.add_argument("--evaluate-interval", type=int, default=5000, help="validation interval")

g = parser.add_argument_group("Wandb Options")
g.add_argument("--wandb-run-name", type=str, help="wanDB run name")
g.add_argument("--wandb-entity", type=str, default="khu-bot", help="wanDB entity name")
g.add_argument("--wandb-project", type=str, default="ai-bookathon", help="wanDB project name")
# fmt: on


def normalize_datum(datum: Datum) -> Datum:
    datum["title"] = datum["title"].strip()
    datum["content"] = normalize_text(datum["content"])
    return datum


def filter_datum(datum: Datum, min_length: int) -> Datum:
    return not datum["title"] or len(datum["content"]) < min_length


def main(args: argparse.Namespace):
    logger = get_logger("train")

    if args.output_dir:
        os.makedirs(args.output_dir)
        logger.info(f'[+] Save output to "{args.output_dir}"')

    logger.info(" ====== Arguements ======")
    for k, v in vars(args).items():
        logger.info(f"{k:25}: {v}")

    logger.info(f"[+] Set Random Seed to {args.seed}")
    pl.seed_everything(args.seed, workers=True)

    logger.info(f"[+] GPU: {args.gpus}")

    logger.info(f'[+] Use Dataset: "{args.dataset_dir}"')
    train_data_path = os.path.join(args.dataset_dir, "train.jsonl")
    dev_data_path = os.path.join(args.dataset_dir, "dev.jsonl")
    test_data_path = os.path.join(args.dataset_dir, "test.jsonl")

    train_data = [normalize_datum(datum) for datum in load_jsonl_data(train_data_path)]
    dev_data = [normalize_datum(datum) for datum in load_jsonl_data(dev_data_path)]
    test_data = [normalize_datum(datum) for datum in load_jsonl_data(test_data_path)]

    logger.info(f"[+] Loaded train data: {len(train_data)}")
    logger.info(f"[+] Loaded dev data: {len(dev_data)}")
    logger.info(f"[+] Loaded test data: {len(test_data)}")

    train_data = [datum for datum in train_data if not filter_datum(datum, args.min_content_char)]
    dev_data = [datum for datum in dev_data if not filter_datum(datum, args.min_content_char)]
    test_data = [datum for datum in test_data if not filter_datum(datum, args.min_content_char)]

    logger.info(f"[+] Remain train data after filtering: {len(train_data)}")
    logger.info(f"[+] Remain dev data after filtering: {len(dev_data)}")
    logger.info(f"[+] Remain test data after filtering: {len(test_data)}")

    if args.tokenizer is None:
        args.tokenizer = args.model
    logger.info(f'[+] Load Tokenizer: "{args.tokenizer}"')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    logger.info(f'[+] Load Model: "{args.model}"')
    model = AutoModelForCausalLM.from_pretrained(args.model)

    train_dataset = LanguageModelingDataset(
        train_data, tokenizer, args.prompt_max_length, args.max_length, args.use_token_type_ids
    )
    dev_dataset = LanguageModelingDataset(
        dev_data, tokenizer, args.prompt_max_length, args.max_length, args.use_token_type_ids
    )
    test_dataset = LanguageModelingDataset(
        test_data, tokenizer, args.prompt_max_length, args.max_length, args.use_token_type_ids
    )

    logger.info(f"[+] # of train examples: {len(train_dataset)}")
    logger.info(f"[+] # of dev examples: {len(dev_dataset)}")
    logger.info(f"[+] # of test examples: {len(test_dataset)}")

    if args.gpus is None:
        args.gpus = torch.cuda.device_count()

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=os.cpu_count() // 2,
        pin_memory=True,
    )
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.valid_batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.valid_batch_size)

    total_steps = len(train_dataloader) * args.epochs

    if args.output_dir:
        model_dir = os.path.join(args.output_dir, "models")
        os.makedirs(model_dir)

        train_loggers = [TensorBoardLogger(args.output_dir, "", "logs")]
    else:
        model_dir = None
        train_loggers = []

    if args.strategy == "deepspeed":
        args.strategy = "deepspeed_stage_2_offload"
        optimizer_name = "deepspeed"
    else:
        optimizer_name = "adam"

    language_modeling = LanguageModeling(
        model=model,
        total_steps=total_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_rate=args.warmup_rate,
        model_save_dir=model_dir,
        optimizer_name=optimizer_name,
        tokenizer=tokenizer,
    )

    logger.info(f"[+] Start Training")
    if args.wandb_project and (args.wandb_run_name or args.output_dir):
        train_loggers.append(
            WandbLogger(
                name=args.wandb_run_name or os.path.basename(args.output_dir),
                project=args.wandb_project,
                entity=args.wandb_entity,
                save_dir=args.output_dir if args.output_dir else None,
            )
        )

    trainer = pl.Trainer(
        logger=train_loggers,
        max_epochs=args.epochs,
        log_every_n_steps=args.logging_interval,
        val_check_interval=args.evaluate_interval,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[LearningRateMonitor(logging_interval="step")] if train_loggers else [],
        strategy=args.strategy,
        accelerator="gpu" if args.gpus else None,
        devices=max(args.gpus, 1),
        precision=args.precision,
    )
    trainer.fit(language_modeling, train_dataloader, dev_dataloader)
    model.save_pretrained(os.path.join(model_dir, "last_checkpoint"))
    trainer.test(language_modeling, test_dataloader)


if __name__ == "__main__":
    exit(main(parser.parse_args()))
