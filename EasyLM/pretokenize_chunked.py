"""
Stream C4, tokenize in chunks, save each chunk to GCS.
Compatible with load_from_disk in training scripts.
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset, Dataset, DatasetDict, Features, Sequence, Value
from transformers import AutoTokenizer
import mlxu
from EasyLM.pretokenize import TextProcessor

def pretokenize_chunked(output_dir, split, tokenizer_name, chunk_size=50000):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    text_processor_config = {
        'fields_from_example': '',
        'fields': 'text',
        'subfield_separator': ' ',
        'add_bos_token': False,
        'add_eos_token': True,
        'prepend_text': '',
        'base64_token_dtype': 'i4',
    }
    text_processor = TextProcessor(text_processor_config, tokenizer)

    print(f"Streaming C4 {split}...")
    dataset = load_dataset('allenai/c4', 'en', split=split, streaming=True)

    chunk_input_tokens = []
    chunk_loss_masks = []
    chunk_idx = 0
    example_idx = 0

    for example in dataset:
        result = text_processor(example)
        chunk_input_tokens.append(result['input_tokens'])
        chunk_loss_masks.append(result['loss_masks'])
        example_idx += 1

        if example_idx % 10000 == 0:
            print(f"Processed {example_idx} examples...")

        if len(chunk_input_tokens) >= chunk_size:
            chunk_path = f"{output_dir}/chunk_{chunk_idx:06d}"
            print(f"Saving chunk {chunk_idx} to {chunk_path}...")
            ds = Dataset.from_dict({
                'input_tokens': chunk_input_tokens,
                'loss_masks': chunk_loss_masks,
            })
            ds.save_to_disk(chunk_path)
            chunk_input_tokens = []
            chunk_loss_masks = []
            chunk_idx += 1

    # Save remaining
    if chunk_input_tokens:
        chunk_path = f"{output_dir}/chunk_{chunk_idx:06d}"
        print(f"Saving final chunk {chunk_idx} to {chunk_path}...")
        ds = Dataset.from_dict({
            'input_tokens': chunk_input_tokens,
            'loss_masks': chunk_loss_masks,
        })
        ds.save_to_disk(chunk_path)

    print(f"Done! {chunk_idx+1} chunks, {example_idx} examples total.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--split', default='train')
    parser.add_argument('--tokenizer', default='google-t5/t5-base')
    parser.add_argument('--chunk_size', type=int, default=50000)
    args = parser.parse_args()
    pretokenize_chunked(args.output_dir, args.split, args.tokenizer, args.chunk_size)
