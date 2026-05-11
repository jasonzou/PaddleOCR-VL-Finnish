echo "lora export to /root/autodl-fs/lora/export"
echo "output_dir is a bit miss-leading; it is an input directory as well"
CUDA_VISIBLE_DEVICES=0 \
paddleformers-cli export lora_export_hf.yaml
