export FLAGS_enable_dataset_debug=false
export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0 paddleformers-cli train configs/dpo.yaml
