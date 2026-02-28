import { detectFace } from "./HaarCascade";
import type { AnnotationRow } from "./Csv";
import { ClassificationModel } from "./RunModel";
import { Landmark, processLandmarks } from "./Landmarks";
import { drawFacePoints, drawFacePointsWithOffset } from "./FaceHelpers";
import { copyMat, imageDataToMat, imageDataToTensor, matToImageData, resizeImage, type ImageSize } from "./AiHelpers";


export async function startPipeline(
	file: File,
	annotations: Map<string, AnnotationRow> | undefined,
	drawImageFn: (label: string, image: ImageData, size: ImageSize) => void,
	displayTable: (outputLandmarks: Array<Landmark>, realLandmarks: Array<Landmark> | undefined, stats: OutputStats | undefined) => void,
): Promise<{ error: string } | undefined> {
	const offscreenContext = await drawFileToOffscreenCanvas(file);

	const originalSize: ImageSize = {
		width: offscreenContext.canvas.width,
		height: offscreenContext.canvas.height,
	};

	const ratiodSize: ImageSize = {
		width: 512,
		height: Math.floor(512 * originalSize.height / originalSize.width),
	};

	console.log(ratiodSize, originalSize.width, originalSize.height);

	const initialImageData = offscreenContext.getImageData(
		0,
		0,
		offscreenContext.canvas.width,
		offscreenContext.canvas.height,
	);

	drawImageFn("Original image", initialImageData, ratiodSize);

	const originalMat = imageDataToMat(initialImageData);
	const faceData = await detectFace(originalMat)
		.catch((error) => console.error("ERROR: " + error));

	if (!faceData) {
		return {
			error: "No face found in image...",
		};
	}

	const faceMat = faceData.mat;

	drawImageFn("Face", matToImageData(resizeImage(faceMat, [512, 512])), { width: faceMat.size().width, height: faceMat.size().height });

	// Run the model
	const resizedImageData = matToImageData(resizeImage(faceMat, [256, 256]));
	const tensor = imageDataToTensor(resizedImageData);

	const model = new ClassificationModel();
	const results = await model.runModel(tensor);

	const resizedMat = imageDataToMat(resizedImageData);
	drawFacePoints(resizedMat, results);

	drawImageFn("Landmarks in face", matToImageData(resizedMat), { width: 256, height: 256});

	const landmarksMat = copyMat(originalMat);
	console.log({ x: faceData.offset.x, y: faceData.offset.y })
	drawFacePointsWithOffset(landmarksMat, faceMat, results, { x: faceData.offset.x, y: faceData.offset.y });

	drawImageFn("Landmarks in image", matToImageData(landmarksMat), ratiodSize);

	const landmarks = processLandmarks(results);

	const annotationLandmarks = getNormalizedLandmarks(annotations, file.name, initialImageData);

	const errors = annotationLandmarks ? calculateLandmarksError(landmarks, annotationLandmarks) : undefined;

	displayTable(landmarks, annotationLandmarks, errors);
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


function calculateLandmarksError(output: Array<Landmark>, annotations: Array<Landmark>): OutputStats {
	let totalError = 0;
	let distance = new Array<number>(output.length);

	for (let i = 0; i < output.length; i++) {
		const error = output[i].euclideanDistance(annotations[i]);
		totalError += error;

		distance[i] = error;
	}

	const median = totalError / output.length;

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
