import argparse
import json
import os
import sys
import time
import requests
from io import BytesIO

from PIL import Image
import paddle
import paddle.distributed as dist
from tqdm import tqdm
import Levenshtein

from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor, AutoConfig
from paddleformers.generation import GenerationConfig


def parse_args():
    parser = argparse.ArgumentParser(description="PaddleFormers & PaddleOCR-VL Model Evaluation Script")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Model path or name")
    parser.add_argument("--data_path", type=str, required=True, help="Test data path (jsonl format)")
    parser.add_argument("--data_dir", type=str, default="", help="Image directory prefix (e.g., ./data/data01/)")
    parser.add_argument("--output_path", type=str, default="eval_results.jsonl", help="Result save path")
    parser.add_argument("--max_length", type=int, default=1024, help="Max generation length")
    parser.add_argument("--device", type=str, default="gpu", help="Device: gpu / cpu / xpu / iluvatar_gpu")
    parser.add_argument("--checkpoint_interval", type=int, default=0, help="Save checkpoint every N samples (0 to disable)")
    return parser.parse_args()


def load_model_and_processor(model_path, device):
    print(f"Loading model: {model_path} ...")
    paddle.set_device(device)

    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForConditionalGeneration.from_pretrained(model_path, convert_from_hf=True)
    model.config._attn_implementation = "flashmask"
    model.visual.config._attn_implementation = "flashmask"
    model.eval()
    print("Model loaded successfully!")
    return model, processor


def compute_metrics(predictions, references):
    total_ned = 0
    num_samples = len(predictions)

    if num_samples == 0:
        return 0.0

    for pred, ref in zip(predictions, references):
        dist = Levenshtein.distance(pred, ref)
        max_len = max(len(pred), len(ref))
        if max_len > 0:
            total_ned += dist / max_len

    avg_ned = total_ned / num_samples
    return avg_ned


def generate_response(model, processor, messages, max_length=1024):
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pd",
    )

    generation_config = GenerationConfig(
        do_sample=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        use_cache=True
    )

    with paddle.no_grad():
        generated_ids = model.generate(**inputs, generation_config=generation_config, max_new_tokens=max_length)
        generated_ids = generated_ids[0].tolist()[0]
        output_text = processor.decode(generated_ids, skip_special_tokens=True)

    return output_text


def main():
    start_time = time.time()
    args = parse_args()

    try:
        dist.init_parallel_env()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    except Exception:
        rank = 0
        world_size = 1
        print("Distributed environment not detected, using single card mode.")

    model, processor = load_model_and_processor(args.model_name_or_path, args.device)

    if rank == 0:
        print(f"Reading data: {args.data_path}")
    samples = []
    with open(args.data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    total_samples = len(samples)
    samples = samples[rank::world_size]

    if rank == 0:
        print(f"Total test samples loaded: {total_samples}")
    print(f"[Rank {rank}] Assigned {len(samples)} samples")

    results = []
    checkpoint_interval = args.checkpoint_interval
    if checkpoint_interval > 0:
        part_interval = max(1, checkpoint_interval)
    else:
        part_interval = max(1, total_samples // 10) if total_samples > 0 else 1
    part_count = 0

    for idx, sample in enumerate(tqdm(samples, desc=f"[Rank {rank}] Inferencing", position=rank)):
        query = sample["messages"][0]["content"]
        image_path = args.data_dir + sample["images"][0]
        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": query.replace('<image>', '')},
                ],
            }
        ]
        output = generate_response(model, processor, messages, args.max_length)
        sample["answer"] = output
        sample["label"] = sample["messages"][1]["content"]

        results.append(sample)

        if checkpoint_interval > 0 and ((idx + 1) % part_interval == 0 or idx == len(samples) - 1):
            part_file = f"{args.output_path}.part{rank}.{part_count}"
            with open(part_file, 'w', encoding='utf-8') as f:
                for res in results:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
            print(f"[Rank {rank}] Checkpoint saved ({idx + 1}/{len(samples)}): {part_file}")
            results = []
            part_count += 1

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        all_results = []
        print("Aggregating results from all Ranks...")
        for r in range(world_size):
            part_count = 0
            while True:
                part_file_r = f"{args.output_path}.part{r}.{part_count}"
                if os.path.exists(part_file_r):
                    with open(part_file_r, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                all_results.append(json.loads(line))
                    try:
                        os.remove(part_file_r)
                    except OSError as e:
                        print(f"Warning: Unable to remove temporary file {part_file_r}: {e}")
                    part_count += 1
                else:
                    break

        if not all_results:
            all_results = results

        predictions = [res.get("answer", "") for res in all_results]
        references = [res.get("label", "") for res in all_results]

        print("Computing evaluation metrics...")
        avg_ned = compute_metrics(predictions, references)

        print("\n" + "="*40)
        print("        Evaluation Report")
        print("="*40)
        print(f"Model: {args.model_name_or_path}")
        print(f"Total Samples: {len(all_results)}")
        print("-" * 40)
        print(f"Avg. NED: {avg_ned:.4f} (Lower is better)")
        print("="*40)

        with open(args.output_path, 'w', encoding='utf-8') as f:
            for res in all_results:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
        print(f"\nDetailed results saved to: {args.output_path}")

        end_time = time.time()
        print(f"Total execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()