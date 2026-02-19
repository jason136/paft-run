import os

import numpy as np
import torch
import torchvision
from PIL import Image

import qai_hub as hub

device = hub.Device("Dragonwing RB3 Gen 2 Vision Kit")

# # 1. Load pre-trained PyTorch model from torchvision
# torch_model = torchvision.models.mobilenet_v2(weights="IMAGENET1K_V1").eval()

# # 2. Trace the model to TorchScript format
# input_shape = (1, 3, 224, 224)
# pt_model = torch.jit.trace(torch_model, torch.rand(input_shape))

# # 3. Compile the model to ONNX
# compile_onnx_job = hub.submit_compile_job(
#     model=pt_model,
#     device=device,
#     input_specs=dict(image_tensor=input_shape),
#     options="--target_runtime onnx",
# )
# assert isinstance(compile_onnx_job, hub.CompileJob)

# unquantized_onnx_model = compile_onnx_job.get_target_model()
# assert isinstance(unquantized_onnx_model, hub.Model)

# 4. Load and pre-process downloaded calibration data
# This transform is required for PyTorch imagenet classifiers
# Source: https://pytorch.org/hub/pytorch_vision_resnet/
mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
sample_inputs = []

# images_dir = "imagenette_samples/images"
# for image_path in os.listdir(images_dir):
#     image = Image.open(os.path.join(images_dir, image_path))
#     image = image.convert("RGB").resize(input_shape[2:])
#     sample_input = np.array(image).astype(np.float32) / 255.0
#     sample_input = np.expand_dims(np.transpose(sample_input, (2, 0, 1)), 0)
#     sample_inputs.append(((sample_input - mean) / std).astype(np.float32))
# calibration_data = dict(image_tensor=sample_inputs)

# 5. Quantize the model
quantize_job = hub.submit_quantize_job(
    model=unquantized_onnx_model,
    calibration_data=calibration_data,
    weights_dtype=hub.QuantizeDtype.INT8,
    activations_dtype=hub.QuantizeDtype.INT8,
)

quantized_onnx_model = quantize_job.get_target_model()
assert isinstance(quantized_onnx_model, hub.Model)
