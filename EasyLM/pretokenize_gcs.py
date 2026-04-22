import argparse
import pickle
from datasets import load_dataset
from transformers import AutoTokenizer
from google.cloud import storage

def tokenize_and_save_gcs(output_dir, split, tokenizer_name, num_examples, chunk_size=10000):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    print(f"Streaming C4 {split} split...")
    dataset = load_dataset('allenai/c4', 'en', split=split, streaming=True)
    bucket_name = output_dir[5:].split('/')[0]
    prefix = '/'.join(output_dir[5:].split('/')[1:])
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    chunk_tokens = []
    chunk_masks = []
    chunk_idx = 0
    example_idx = 0
    for example in dataset:
        if example_idx >= num_examples:
            break
        tokens = tokenizer(example['text'], truncation=False)['input_ids']
        for i in range(0, len(tokens) - 1024, 1024):
            chunk_tokens.append(tokens[i:i+1024])
            chunk_masks.append([1.0] * 1024)
        example_idx += 1
        if example_idx % 10000 == 0:
            print(f"Processed {example_idx} examples, {len(chunk_tokens)} sequences...")
        if len(chunk_tokens) >= chunk_size:
            blob_name = f"{prefix}/chunk_{chunk_idx:06d}.pkl"
            blob = bucket.blob(blob_name)
            data = {'input_tokens': chunk_tokens, 'loss_masks': chunk_masks}
            blob.upload_from_string(pickle.dumps(data))
            print(f"Saved chunk {chunk_idx} to gs://{bucket_name}/{blob_name}")
            chunk_tokens = []
            chunk_masks = []
            chunk_idx += 1
    if chunk_tokens:
        blob_name = f"{prefix}/chunk_{chunk_idx:06d}.pkl"
        blob = bucket.blob(blob_name)
        data = {'input_tokens': chunk_tokens, 'loss_masks': chunk_masks}
        blob.upload_from_string(pickle.dumps(data))
        print(f"Saved final chunk {chunk_idx}")
    print(f"Done! Saved {chunk_idx+1} chunks, {example_idx} examples total.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--split', default='train')
    parser.add_argument('--tokenizer', default='google-t5/t5-base')
    parser.add_argument('--num_examples', type=int, default=10000000)
    parser.add_argument('--chunk_size', type=int, default=10000)
    args = parser.parse_args()
    tokenize_and_save_gcs(args.output_dir, args.split, args.tokenizer, args.num_examples, args.chunk_size)
