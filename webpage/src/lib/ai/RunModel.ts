import * as ort from "onnxruntime-web/wasm";
import { calculateSoftmax } from "./AiHelpers";
import { CLASSIFICATION_CLASSES } from "./ClassificationClasses";


export class ClassificationModel {
	async runModel(input: ort.Tensor): Promise<Float32Array> {
		// Tell ORT where to find the .mjs/.wasm backends
		// ort.env.wasm.wasmPaths = "http://localhost:8000/assets/";
		// ort.env.wasm.wasmPaths = "/assets/models/";

		// const wasmResponse = await fetch("assets/ort-wasm-simd-threaded.wasm");
		// const bytes = await wasmResponse.bytes();
		// ort.env.wasm.wasmBinary = bytes;

		// console.log(wasmResponse.headers, bytes);

		// ort.env.wasmBinary = bytes;

		// ort.env.wasm.simd = true;
		// ort.env.wasm.numThreads = 4;

		console.log("BEFORE")

		// const baseModelResponse = await fetch("assets/models/ResNet.onnx");
		// const baseModelBytes = await baseModelResponse.bytes();

		// const modelResponse = await fetch("assets/models/ResNet.onnx.data");
		// const modelBytes = await modelResponse.bytes();

		ort.env.wasm.initTimeout = 10000;
		ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@dev/dist/";

		// Load the model (can be local or remote URL)
		const session = await ort.InferenceSession.create("/assets/models/FaceMerged.onnx", {
			// externalData: ["./assets/models/ResNet.onnx.data"],
			executionProviders: ["wasm"],
			// logSeverityLevel: 0,
			// logVerbosityLevel: 0,
		});

		console.log("Model session created correctly!");

		console.log(session.inputNames, session.outputNames)

		const results = await session.run({ x: input });
		console.log(results);

		return results["linear"].data as Float32Array;

		// Run the model
		// const results = await session.run({ image: input });

		// const probabilities = calculateSoftmax(results["predictions"].data as Float32Array);

		// return new ClassificationResults(probabilities);

		// return undefined!;
	}
}


export class ClassificationResults {
	constructor(private probabilities: number[])
		{}


	getTopResults(results: number = 3): Array<[string, number]> {
		const classes: Array<[string, number]> = Object.keys(CLASSIFICATION_CLASSES).map((className, index) => {
			return [className, this.probabilities[index]];
		});

		classes.sort((a, b) => {
			return b[1] - a[1];
		})

		return classes.splice(0, results);
	}
}

