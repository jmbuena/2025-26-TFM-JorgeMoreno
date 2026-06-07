// import * as wasmOrt from "onnxruntime-web/wasm";
// import * as gpuOrt from "onnxruntime-web/webgpu";

import { InferenceSession, Tensor } from "onnxruntime-web";


export class ClassificationModel {
	protected session: InferenceSession | undefined;

	protected isLoaded = false;

	protected useGPU: boolean = true;

	protected ort: any;


	setUseGPU(useGPU: boolean): void {
		this.useGPU = useGPU;
	}


	async load(modelPath: string, wasmPaths: string) {
		if (this.isLoaded) {
			return;
		}

		console.log("Starting model...");

		const ort = this.useGPU
			? await import("onnxruntime-web/webgpu")
			: await import("onnxruntime-web/wasm");

		ort.env.wasm.initTimeout = 100000;
		ort.env.wasm.wasmPaths = `${window.location.origin}/${wasmPaths}`;

		const providers = this.useGPU
			? ["webgpu"]
			: ["wasm"];

		const result = await fetch(modelPath);
		
		// Load the model (can be local or remote URL)
		this.session = await InferenceSession.create(await result.arrayBuffer(), {
			executionProviders: providers,
			enableCpuMemArena: true,
		});

		this.isLoaded = true;

		console.log("Model session created correctly!");

		console.log(this.session.inputNames, this.session.outputNames)
	}


	async runModel(input: Tensor): Promise<{ output: Float32Array, elapsed: number } | undefined> {
		if (!this.session) {
			return undefined;
		}

		const startTime = performance.now();
		const results = await this.session.run({ x: input });
		const elapsed = performance.now() - startTime;

		return {
			output: results["linear"].data as Float32Array,
			elapsed,
		};
	}
}
