import paddle

paddle.utils.run_check()

print("Device:", paddle.device.get_device())
print("CUDA Version:", paddle.version.cuda())
print("Available Memory:", paddle.device.cuda.get_device_properties(paddle.device.get_device()).total_memory / 1e9, "GB")


from paddleformers.trainer import Trainer
from paddleformers import AutoModel, AutoTokenizer

# Replace with your actual VL v1.5 checkpoint path
model_name = "/root/autodl-fs/models/PaddlePaddle-VL-v1.5"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, dtype='bfloat16')

print(f"✅ Model loaded on {paddle.device.get_device()}")
print(f"✅ Parameters: {model.num_parameters()}")
print(f"✅ VRAM Usage: {paddle.device.cuda.memory_allocated()/1e9:.2f} GB")
