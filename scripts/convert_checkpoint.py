import argparse
import tempfile

from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from transformers import AutoTokenizer

from essayist.task import LanguageModeling
from essayist.utils import get_logger

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--output-path", type=str, required=True)
parser.add_argument("--tokenizer", type=str, help="Save with tokenizer")
parser.add_argument("--deepspeed", action="store_true", help="Use this flag for deepspeed checkpoint")


def main(args: argparse.Namespace):
    logger = get_logger("checkpoint_convert")

    logger.info(f"[+] Load Lightning module checkpoint from: {args.checkpoint}")
    if args.deepspeed:
        f = tempfile.NamedTemporaryFile()
        convert_zero_checkpoint_to_fp32_state_dict(args.checkpoint, f.name)
        args.checkpoint = f.name

    lm = LanguageModeling.load_from_checkpoint(args.checkpoint)
    logger.info(f"[+] Save huggingface pretrained model to: {args.output_path}")
    lm.model.save_pretrained(args.output_path)

    if args.deepspeed:
        f.close()

    if args.tokenizer:
        logger.info(f"[+] Load Tokenizer from: {args.tokenizer}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

        logger.info(f"[+] Save huggingface tokenizer to: {args.output_path}")
        tokenizer.save_pretrained(args.output_path)


if __name__ == "__main__":
    exit(main(parser.parse_args()))
