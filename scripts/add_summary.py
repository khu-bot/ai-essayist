import argparse
import json

from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--data-file-path", type=str, required=True, help="jsonl data file")
parser.add_argument("--output-path", type=str, required=True, help="output jsonl path")
parser.add_argument("--model", type=str, default="digit82/kobart-summarization", help="Summarization model")
parser.add_argument("--tokenizer", type=str)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--sum-input-max-length", type=int, default=512)
parser.add_argument("--sum-max-token-length", type=int, default=64)
parser.add_argument("--max-content-length", type=str, default=800)
parser.add_argument("--max-summarization", type=int, default=10, help="maximun summarization numbers")
parser.add_argument("--num-beams", type=int, default=4)
parser.add_argument("--device")


def main(args: argparse.Namespace):
    print(f"[+] Load data: {args.data_file_path}")

    with open(args.data_file_path) as f:
        data = [json.loads(l) for l in f]
    print(f"Found {len(data)} examples")

    if args.tokenizer is None:
        args.tokenizer = args.model

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(args.device)

    print("[+] Chunk data...")
    docs = []
    for doc_id, datum in enumerate(data):
        doc = []
        sentences = datum["content"].split("\n")
        cur_content = ""
        part_id = 0

        while sentences:
            sentence = sentences.pop(0)
            if len(cur_content) + len(sentence) + 1 > args.max_content_length:
                doc.append(
                    {
                        "title": datum["title"],
                        "content": cur_content,
                        "doc_id": doc_id,
                        "part_id": part_id,
                        "summarizations": [],
                    }
                )
                cur_content = sentence
                part_id += 1
                continue
            cur_content += "\n" + sentence
        if cur_content:
            doc.append(
                {
                    "title": datum["title"],
                    "content": cur_content,
                    "doc_id": doc_id,
                    "part_id": part_id,
                    "summarizations": [],
                }
            )
        docs.append(doc)
    all_data = [d for doc in docs for d in doc]
    print(f"[+] Get {len(all_data)} chunks")

    print(f"[+] Start to append summarizations")
    for i in tqdm(range(0, len(all_data), args.batch_size)):
        batch_data = all_data[i : i + args.batch_size]
        texts = [tokenizer.bos_token + d["content"] + tokenizer.eos_token for d in batch_data]

        input_ids = tokenizer(
            texts,
            padding="longest",
            return_tensors="pt",
            max_length=args.sum_input_max_length,
            truncation=True,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=False,
        )["input_ids"].to(args.device)

        sum_tokens = model.generate(
            input_ids,
            num_beams=args.num_beams,
            max_length=args.sum_max_token_length,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        summarizations = tokenizer.batch_decode(sum_tokens, skip_special_tokens=True)

        for doc_part, summarization in zip(batch_data, summarizations):
            doc_id = doc_part["doc_id"]
            part_id = doc_part["part_id"]

            doc = docs[doc_id]
            for after_part_index in range(part_id + 1, min(len(doc), part_id + 1 + args.max_summarization)):
                doc[after_part_index]["summarizations"].append(summarization)
                doc[after_part_index]["summarizations"] = doc[after_part_index]["summarizations"][
                    -args.max_summarization :
                ]

    print(f"[+] Save data to: {args.output_path}")
    with open(args.output_path, "w", encoding="utf-8") as f:
        for data in all_data:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    exit(main(parser.parse_args()))
