import torch
import segmentation_models_pytorch as smp

model = smp.Unet(
	encoder_name="mobilenet_v2",
	encoder_weights=None,
	in_channels=3,
	classes=8,
)

model_path = "outputs/models/cityscapes_mobilenet_v2_data_augmentation.pth"
state_dict = torch.load(model_path, map_location="cpu")

model.load_state_dict(state_dict)
model.eval()

dummy_input = torch.randn(1, 3, 512, 1024)

onnx_path = "outputs/models/mobilenet_v2.onnx"
torch.onnx.export(
	model,
	dummy_input, #type:ignore
	onnx_path,
	export_params=True,
	opset_version=18,
	do_constant_folding=True,
	input_names=['input'],
	output_names=['output']
)

print(f"odèle converti avec succès et sauvegardé ici : {onnx_path}")