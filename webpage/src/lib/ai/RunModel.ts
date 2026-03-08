import * as ort from "onnxruntime-web/wasm";
import { CLASSIFICATION_CLASSES } from "./ClassificationClasses";


export class ClassificationModel {
	protected session: ort.InferenceSession | undefined;

	protected isLoaded = false;


	async load() {
		if (this.isLoaded) {
			return;
		}

		console.log("Starting model...")

		// const baseModelResponse = await fetch("assets/models/ResNet.onnx");
		// const baseModelBytes = await baseModelResponse.bytes();

		// const modelResponse = await fetch("assets/models/ResNet.onnx.data");
		// const modelBytes = await modelResponse.bytes();

		ort.env.wasm.initTimeout = 10000;
		ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@dev/dist/";

		// Load the model (can be local or remote URL)
		this.session = await ort.InferenceSession.create("/assets/models/FaceMerged.onnx", {
			// externalData: ["./assets/models/ResNet.onnx.data"],
			executionProviders: ["wasm"],
			// logSeverityLevel: 0,
			// logVerbosityLevel: 0,
		});

		this.isLoaded = true;

		console.log("Model session created correctly!");

		console.log(this.session.inputNames, this.session.outputNames)
	}


	async runModel(input: ort.Tensor): Promise<Float32Array> {
		if (!this.session) {
			return new Float32Array();
		}

		const results = await this.session.run({ x: input });
		return results["linear"].data as Float32Array;
	}
}
