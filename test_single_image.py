import argparse
import paddle
from PIL import Image

from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.generation import GenerationConfig

PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
    "region": "Recognize the text inside the red box",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Test single image with PaddleOCR-VL model")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Model path or name")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--question", type=str, default=None, help="Question about the image")
    parser.add_argument("--task", type=str, default=None, choices=list(PROMPTS.keys()), help="Predefined task (overridden by --question)")
    parser.add_argument("--max_length", type=int, default=1024, help="Max generation length")
    parser.add_argument("--device", type=str, default="gpu", help="Device: gpu / cpu / xpu / iluvatar_gpu")
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
    args = parse_args()

    model, processor = load_model_and_processor(args.model_name_or_path, args.device)

    image = Image.open(args.image_path).convert("RGB")

    if args.question:
        question = args.question
    elif args.task:
        question = PROMPTS[args.task]
    else:
        raise ValueError("Either --question or --task must be provided")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]

    print(f"\nQuestion: {question}")
    print(f"Image: {args.image_path}")
    print("\nGenerating response...")

    output = generate_response(model, processor, messages, args.max_length)

    print(f"\nAnswer: {output}")


if __name__ == "__main__":
    main()