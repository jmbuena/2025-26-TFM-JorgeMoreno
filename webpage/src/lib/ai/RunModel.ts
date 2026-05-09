import * as wasmOrt from "onnxruntime-web/wasm";
import * as gpuOrt from "onnxruntime-web/webgpu";


export class ClassificationModel {
	protected session: wasmOrt.InferenceSession | undefined;

	protected isLoaded = false;

	protected useGPU: boolean = true;

	protected ort: any;


	setUseGPU(useGPU: boolean): void {
		this.useGPU = useGPU;
	}


	async load(model: string) {
		if (this.isLoaded) {
			return;
		}

		console.log("Starting model...");

		const ort = this.useGPU
			? gpuOrt
			: wasmOrt;

		ort.env.wasm.initTimeout = 100000;
		ort.env.wasm.wasmPaths = `${window.location.origin}/assets/ort_wasm/`;

		const providers = this.useGPU
			? ["webgpu"]
			: ["wasm"];
		
		// Load the model (can be local or remote URL)
		this.session = await ort.InferenceSession.create(`/assets/models/${model}.onnx`, {
			executionProviders: providers,
		});

		this.isLoaded = true;

		console.log("Model session created correctly!");

		console.log(this.session.inputNames, this.session.outputNames)
	}


	async runModel(input: wasmOrt.Tensor): Promise<Float32Array> {
		if (!this.session) {
			return new Float32Array();
		}

		const results = await this.session.run({ x: input });
		return results["linear"].data as Float32Array;
	}
}
