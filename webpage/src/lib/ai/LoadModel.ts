import * as ort from 'onnxruntime-web';
import { type CV } from "@techstark/opencv-js";
import cvReadyPromise from "@techstark/opencv-js";


export let cv: CV;

export async function prepareModels(): Promise<void> {
	ort.env.wasm.wasmPaths = "/assets/";
	ort.env.debug = true;
	ort.env.logLevel = "verbose";

	await loadOpenCV();
}


async function loadOpenCV(): Promise<void> {
	cv = await cvReadyPromise;

	console.log("OpenCV.js is ready!");
}
