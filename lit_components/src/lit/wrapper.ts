import { LitElement, html } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { PipelineOutput, runModelPipelineFromFile } from '../scripts/Pipeline';
import { AnnotationRow } from '../scripts/Csv';
import { ClassificationModel } from '../scripts/RunModel';
import { Landmark } from '../scripts/Landmarks';
import { loadOpenCV } from '../scripts/LoadModel';
import { drawFacePoints } from '../scripts/FaceHelpers';

type ModelArgs = {
	file: File,
	annotations: Map<string, AnnotationRow> | undefined,
	model: ClassificationModel,
	inputSize: number,
	canvas: HTMLCanvasElement | undefined,
}

@customElement("face-alignment")
export class FaceAlignment extends LitElement {
	private static isOpenCvReady: boolean = false;


	public async loadOpenCV(): Promise<void> {
		if (!FaceAlignment.isOpenCvReady) {
			await loadOpenCV();

			FaceAlignment.isOpenCvReady = true;
		}
	}


	public async loadModel(modelPath: string, useGPU: boolean, wasmPaths: string): Promise<ClassificationModel> {
		const model = new ClassificationModel();
		model.setUseGPU(useGPU);

		await model.load(modelPath, wasmPaths);

		return model;
	}


	public async executeModel(args: ModelArgs): Promise<Array<Landmark> | undefined> {
		return runModelPipelineFromFile(args.model, args.inputSize, args.file, args.canvas);
	}


	protected render() {
		return html`<span style="display: none">Component loaded</span>`;
	}
}
