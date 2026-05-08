CUDA_VISIBLE_DEVICES=0 \
paddleformers-cli export lora_export.yaml \
    model_name_or_path=./models/PaddleOCR-VL-1.5 \
    output_dir=/root/autodl-fs/lora01
