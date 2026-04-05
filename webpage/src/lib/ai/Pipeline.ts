import { Timings } from "./Timings";
import { detectFace, isHaarReady, loadHaarCascade } from "./HaarCascade";
import type { AnnotationRow } from "./Csv";
import { ClassificationModel } from "./RunModel";
import { Landmark, processLandmarks } from "./Landmarks";
import { Color, drawFacePoints, drawFacePointsWithOffset, drawSquare } from "./FaceHelpers";
import { copyMat, imageDataToMat, imageDataToTensor, matToImageData, resizeImage, type ImageSize } from "./AiHelpers";


type PipelineOutput = {
	error: string,
} | {
	timings: Timings,
	errors: OutputStats | undefined,
}

export async function startPipeline(
	modelName: string,
	file: File,
	annotations: Map<string, AnnotationRow> | undefined,
	model: ClassificationModel,
	drawImageFn?: (label: string, image: ImageData, size: ImageSize) => void,
	displayTable?: (outputLandmarks: Array<Landmark>, realLandmarks: Array<Landmark> | undefined, stats: OutputStats | undefined) => void,
): Promise<PipelineOutput> {
	const timings = new Timings();

	const offscreenContext = await drawFileToOffscreenCanvas(file);

	const originalSize: ImageSize = {
		width: offscreenContext.canvas.width,
		height: offscreenContext.canvas.height,
	};

	const ratiodSize: ImageSize = {
		width: 512,
		height: Math.floor(512 * originalSize.height / originalSize.width),
	};

	const initialImageData = offscreenContext.getImageData(
		0,
		0,
		offscreenContext.canvas.width,
		offscreenContext.canvas.height,
	);

	if (drawImageFn) {
		drawImageFn("Original image", initialImageData, ratiodSize);
	}
	
	const annotationLandmarks = getNormalizedLandmarks(annotations, file.name, initialImageData);
	
	const originalMat = imageDataToMat(initialImageData);

	if (!isHaarReady) {
		await loadHaarCascade();
	}

	const faceData = timings.measure("faceDetection", () => {
		return detectFace(originalMat);
	});

	if (!faceData?.mat) {
		return {
			error: "No face found in image...",
		};
	}

	const faceMat = faceData.mat;

	if (drawImageFn) {
		drawImageFn("Face", matToImageData(resizeImage(faceMat, [512, 512])), { width: 256, height: 256 });
	}

	// Run the model
	const resizedFaceMat = resizeImage(faceMat, [256, 256]);
	const resizedImageData = matToImageData(resizedFaceMat);

	// resizedFaceMat.release();
	// faceMat.delete();

	await timings.measure("loadingModel", async () => {
		return model.load(modelName);
	});

	const tensor = imageDataToTensor(resizedImageData);

	const results = await timings.measure("runModel", async () => {
		return model.runModel(tensor);
	});

	tensor.dispose();

	if (drawImageFn) {
		const resizedMat = imageDataToMat(resizedImageData);
		drawFacePoints(resizedMat, results, Color.RED, 1);
		
		drawImageFn("Landmarks in face", matToImageData(resizedMat), { width: 256, height: 256 });
		resizedMat.release();
	}

	const landmarksMat = copyMat(originalMat);

	if (annotationLandmarks) {
		drawFacePoints(
			landmarksMat,
			landmarksToFloat32Array(annotationLandmarks),
			Color.GREEN,
			5,
		);
	}
	
	if (drawImageFn) {
		drawFacePointsWithOffset(
			landmarksMat,
			faceMat,
			results,
			{ x: faceData.offset.x, y: faceData.offset.y },
			Color.RED,
			4
		);

		drawImageFn("Landmarks in image", matToImageData(landmarksMat), ratiodSize);
	}

	const imageWidth = originalSize.width;
	const headWidth = faceMat.size().width;
	const relativeHeadSize = headWidth / imageWidth;
	
	const landmarks = processLandmarks(results);
	const errors = annotationLandmarks
		? calculateLandmarksError(landmarks, annotationLandmarks, relativeHeadSize)
		: undefined;

	if (displayTable) {
		displayTable(landmarks, annotationLandmarks, errors);
	}

	return {
		timings,
		errors,
	};
}


export function getNormalizedLandmarks(
	annotations: Map<string, AnnotationRow> | undefined,
	filename: string,
	image: ImageData,
): Array<Landmark> | undefined {
	if (!annotations) {
		return undefined;
	}

	const rowData = annotations.get(filename);
	if (!rowData) {
		return undefined;
	}

	return rowData.landmarks.map((landmark) => {
		return new Landmark(
			landmark.x / image.width,
			landmark.y / image.height,
		);
	});
}


async function drawFileToOffscreenCanvas(file: File): Promise<OffscreenCanvasRenderingContext2D> {
	return new Promise((resolve, reject) => {
		const fileReader = new FileReader();

		fileReader.onload = function() {
			const image = new Image();
			
			image.onload = function() {
				const offscreen = new OffscreenCanvas(image.width, image.height);
				const offscreenContext = offscreen.getContext("2d")!;
				
				offscreenContext.drawImage(image, 0, 0);

				resolve(offscreenContext);
			}

			image.src = fileReader.result as string;
		};

		fileReader.readAsDataURL(file);
	});
}


export type OutputStats = {
	landmarkDistance: Array<number>;
	medianDistance: number;
	deviation: number;
}


function calculateLandmarksError(
	output: Array<Landmark>,
	annotations: Array<Landmark>,
	relativeHeadSize: number,
): OutputStats {
	let totalError = 0;
	let distance = new Array<number>(output.length);

	for (let i = 0; i < output.length; i++) {
		const error = output[i].euclideanDistance(annotations[i]);
		totalError += error;

		distance[i] = error;
	}

	const median = (totalError / output.length) * relativeHeadSize;

	// Calculate deviation
	const sumErrorsSquared = distance.reduce((prev, current) => {
		return Math.pow(current - median, 2) + prev;
	});

	const deviation = Math.sqrt(sumErrorsSquared / (distance.length - 1));

	return {
		deviation,
		medianDistance: median,
		landmarkDistance: distance,
	};
}


function landmarksToFloat32Array(landmarks: Array<Landmark>): Float32Array {
	return new Float32Array(landmarks.reduce((prev, curr, index) => {
		prev[index * 2] = curr.x;
		prev[index * 2 + 1] = curr.y;

		return prev;
	}, new Array<number>(landmarks.length * 2)));
}
