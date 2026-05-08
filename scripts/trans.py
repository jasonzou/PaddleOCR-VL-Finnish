import os
from pathlib import Path
from PIL import Image
import argparse
import json
import torch
from loguru import logger
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoTokenizer


PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
    "spotting": "Spotting:",
    "seal": "Seal Recognition:",
}


def main():
    parser = argparse.ArgumentParser(description="PaddleOCR-VL inference")
    parser.add_argument("-i", "--image", default=None, help="Input image path (single)")
    parser.add_argument("-d", "--dataset", default=None, help="Load images from dataset directory")
    parser.add_argument("-n", "--num_images", type=int, default=10, help="Number of images to test")
    parser.add_argument("-m", "--model", default="PaddlePaddle/PaddleOCR-VL-1.5", help="Model path")
    parser.add_argument("-t", "--task", default="ocr", help="Task: ocr|table|chart|formula|spotting|seal")
    parser.add_argument("-o", "--output", default="results.json", help="Output result file")
    args = parser.parse_args()

    model_path = args.model
    task = args.task
    num_images = args.num_images
    output_file = args.output

    logger.add("trans_{time}.log", rotation="10 MB")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {DEVICE}")

    logger.info(f"Loading model from {model_path}")
    model = AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(DEVICE).eval()
    logger.info("Model loaded")

    logger.info("Fixing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True)
    chat_template_path = os.path.join(model_path, "chat_template.jinja")
    if os.path.exists(chat_template_path):
        with open(chat_template_path) as f:
            tokenizer.chat_template = f.read()
        logger.info("Chat template loaded")

    processor = AutoProcessor.from_pretrained(model_path, tokenizer=tokenizer)
    logger.info("Processor loaded")

    def process_image(image_path, task):
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size
        spotting_upscale_threshold = 1500

        if task == "spotting" and orig_w < spotting_upscale_threshold and orig_h < spotting_upscale_threshold:
            process_w, process_h = orig_w * 2, orig_h * 2
            try:
                resample_filter = Image.Resampling.LANCZOS
            except AttributeError:
                resample_filter = Image.LANCZOS
            image = image.resize((process_w, process_h), resample_filter)

        max_pixels = 2048 * 28 * 28 if task == "spotting" else 1280 * 28 * 28

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPTS[task]},
                ]
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            images_kwargs={"size": {"shortest_edge": processor.image_processor.size["shortest_edge"], "longest_edge": max_pixels}},
        ).to(model.device)

        outputs = model.generate(**inputs, max_new_tokens=512, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
        result = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:-1])
        return result

    results = []

    if args.dataset:
        dataset_dir = Path(args.dataset)
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        image_files = sorted([f for f in dataset_dir.iterdir() if f.suffix.lower() in image_extensions])[:num_images]
        logger.info(f"Found {len(image_files)} images in {dataset_dir}")

        for i, img_path in enumerate(image_files):
            logger.info(f"Processing {i+1}/{len(image_files)}: {img_path.name}")
            try:
                result = process_image(img_path, task)
                results.append({"image": str(img_path), "result": result})
                logger.info(f"Result: {result}")
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                results.append({"image": str(img_path), "error": str(e)})

    elif args.image:
        logger.info(f"Processing single image: {args.image}")
        result = process_image(args.image, task)
        logger.info(f"Result: {result}")
        results.append({"image": args.image, "result": result})
    else:
        logger.error("Please specify -i for single image or -d for dataset directory")
        exit(1)

    with open(output_file, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()