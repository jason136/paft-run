"""Quantize all moondream2 components for Rubik Pi via Qualcomm AI Hub."""

import os

import numpy as np
import onnx
from huggingface_hub import hf_hub_download
from PIL import Image

import qai_hub as hub

device = hub.Device("Dragonwing RB3 Gen 2 Vision Kit")

images_dir = "imagenette_samples/images"
mean = np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1)).astype(np.float32)
std = np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1)).astype(np.float32)

calibration_images = []
for fname in os.listdir(images_dir):
    if not fname.lower().endswith((".jpeg", ".jpg", ".png")):
        continue
    img = Image.open(os.path.join(images_dir, fname)).convert("RGB").resize((378, 378))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr.transpose(2, 0, 1), 0)
    calibration_images.append(((arr - mean) / std).astype(np.float32))
    if len(calibration_images) >= 100:
        break

print(f"Loaded {len(calibration_images)} calibration images")

print("\n=== vision_encoder ===")
ve_path = hf_hub_download(
    repo_id="Xenova/moondream2", filename="onnx/vision_encoder.onnx"
)

ve_proto = onnx.load(ve_path, load_external_data=False)
ve_input_specs = {}
for inp in ve_proto.graph.input:
    ve_input_specs[inp.name] = tuple(
        1 if d.dim_param else d.dim_value for d in inp.type.tensor_type.shape.dim
    )
print("  Inputs:", ve_input_specs)

ve_onnx_job = hub.submit_compile_job(
    model=ve_path,
    device=device,
    input_specs=ve_input_specs,
    options="--target_runtime onnx",
)
ve_unquantized = ve_onnx_job.get_target_model()

ve_image_input = next(n for n, s in ve_input_specs.items() if len(s) == 4)
ve_calibration = {ve_image_input: calibration_images}
for n, s in ve_input_specs.items():
    if n != ve_image_input:
        ve_calibration[n] = [
            np.random.randn(*s).astype(np.float32)
            for _ in range(len(calibration_images))
        ]

ve_quantize_job = hub.submit_quantize_job(
    model=ve_unquantized,
    calibration_data=ve_calibration,
    weights_dtype=hub.QuantizeDtype.INT8,
    activations_dtype=hub.QuantizeDtype.INT8,
)
ve_quantized = ve_quantize_job.get_target_model()

ve_qnn_job = hub.submit_compile_job(
    model=ve_quantized,
    device=device,
    options="--target_runtime qnn_context_binary --quantize_io",
)
ve_compiled = ve_qnn_job.get_target_model()
ve_compiled.download("vision_encoder_int8_qnn.bin")
print("  Saved: vision_encoder_int8_qnn.bin")
print("  Profile:", hub.submit_profile_job(model=ve_compiled, device=device).url)


print("\n=== embed_tokens ===")
et_path = hf_hub_download(
    repo_id="Xenova/moondream2", filename="onnx/embed_tokens.onnx"
)

et_proto = onnx.load(et_path, load_external_data=False)
et_input_specs = {}
for inp in et_proto.graph.input:
    et_input_specs[inp.name] = tuple(
        1 if d.dim_param else d.dim_value for d in inp.type.tensor_type.shape.dim
    )
print("  Inputs:", et_input_specs)

et_onnx_job = hub.submit_compile_job(
    model=et_path,
    device=device,
    input_specs=et_input_specs,
    options="--target_runtime onnx",
)
et_unquantized = et_onnx_job.get_target_model()

et_calibration = {}
for n, s in et_input_specs.items():
    elem_type = next(
        i for i in et_proto.graph.input if i.name == n
    ).type.tensor_type.elem_type
    if elem_type in (6, 7):
        dtype = np.int64 if elem_type == 7 else np.int32
        et_calibration[n] = [
            np.random.randint(0, 1000, size=s).astype(dtype) for _ in range(100)
        ]
    else:
        et_calibration[n] = [np.random.randn(*s).astype(np.float32) for _ in range(100)]

et_quantize_job = hub.submit_quantize_job(
    model=et_unquantized,
    calibration_data=et_calibration,
    weights_dtype=hub.QuantizeDtype.INT8,
    activations_dtype=hub.QuantizeDtype.INT8,
)
et_quantized = et_quantize_job.get_target_model()

et_qnn_job = hub.submit_compile_job(
    model=et_quantized,
    device=device,
    options="--target_runtime qnn_context_binary --quantize_io",
)
et_compiled = et_qnn_job.get_target_model()
et_compiled.download("embed_tokens_int8_qnn.bin")
print("  Saved: embed_tokens_int8_qnn.bin")
print("  Profile:", hub.submit_profile_job(model=et_compiled, device=device).url)


print("\n=== decoder ===")
dec_path = hf_hub_download(
    repo_id="Xenova/moondream2", filename="onnx/decoder_model_merged.onnx"
)
hf_hub_download(
    repo_id="Xenova/moondream2", filename="onnx/decoder_model_merged.onnx_data"
)

dec_proto = onnx.load(dec_path, load_external_data=False)
dec_input_specs = {}
for inp in dec_proto.graph.input:
    dec_input_specs[inp.name] = tuple(
        1 if d.dim_param else d.dim_value for d in inp.type.tensor_type.shape.dim
    )
print("  Inputs:", dec_input_specs)

dec_onnx_job = hub.submit_compile_job(
    model=dec_path,
    device=device,
    input_specs=dec_input_specs,
    options="--target_runtime onnx",
)
dec_unquantized = dec_onnx_job.get_target_model()

dec_calibration = {}
for n, s in dec_input_specs.items():
    elem_type = next(
        i for i in dec_proto.graph.input if i.name == n
    ).type.tensor_type.elem_type
    if elem_type in (6, 7):
        dtype = np.int64 if elem_type == 7 else np.int32
        dec_calibration[n] = [
            np.random.randint(0, 1000, size=s).astype(dtype) for _ in range(100)
        ]
    else:
        dec_calibration[n] = [
            np.random.randn(*s).astype(np.float32) for _ in range(100)
        ]

dec_quantize_job = hub.submit_quantize_job(
    model=dec_unquantized,
    calibration_data=dec_calibration,
    weights_dtype=hub.QuantizeDtype.INT8,
    activations_dtype=hub.QuantizeDtype.INT8,
)
dec_quantized = dec_quantize_job.get_target_model()

dec_qnn_job = hub.submit_compile_job(
    model=dec_quantized,
    device=device,
    options="--target_runtime qnn_context_binary --quantize_io",
)
dec_compiled = dec_qnn_job.get_target_model()
dec_compiled.download("decoder_int8_qnn.bin")
print("  Saved: decoder_int8_qnn.bin")
print("  Profile:", hub.submit_profile_job(model=dec_compiled, device=device).url)

print("\nDone!")
