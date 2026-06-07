import { Tensor, type TypedTensor } from "onnxruntime-web/wasm";
import { ClassificationModel } from "./scripts/RunModel";


export async function loadModel(modelPath: string, useGPU: boolean, wasmPaths: string): Promise<ClassificationModel> {
	const model = new ClassificationModel();
	model.setUseGPU(useGPU);

	await model.load(modelPath, wasmPaths);

	return model;
}


export function imageDataToTensor(image: ImageData): TypedTensor<"float32"> {
	const { data, width, height } = image; // RGBA Uint8ClampedArray

	// Convert RGBA to normalized RGB
	const floatData = new Float32Array(3 * width * height);
	for (let i = 0; i < width * height; i++) {
		const r = data[i * 4];
		const g = data[i * 4 + 1];
		const b = data[i * 4 + 2];

		// Normalize to [0, 1]
		floatData[i] = r / 255;
		floatData[i + width * height] = g / 255;
		floatData[i + 2 * width * height] = b / 255;
	}

	return new Tensor('float32', floatData, [1, 3, height, width]);
}


export async function runModelPipelineFromImageData(model: ClassificationModel, imageData: ImageData) {
	return model.runModel(imageDataToTensor(imageData)); 
}


export async function runModelPipelineFromTensor(model: ClassificationModel, tensor: TypedTensor<"float32">) {
	return model.runModel(tensor); 
}
