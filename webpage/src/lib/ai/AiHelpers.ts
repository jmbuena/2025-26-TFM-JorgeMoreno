import * as ort from "onnxruntime-web";
import { cv } from "./LoadModel";
import type { Mat, Size } from "@techstark/opencv-js";


export async function showImageDataInCanvas(image: ImageData, canvasCtx: CanvasRenderingContext2D): Promise<void> {
	canvasCtx.clearRect(0, 0, image.width, image.height);
	canvasCtx.drawImage(await createImageBitmap(image), 0, 0, image.width, image.height);
}


export function imageDataToTensor(image: ImageData): ort.TypedTensor<"float32"> {
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

	return new ort.Tensor('float32', floatData, [1, 3, height, width]);
}


export function extractDrawing(original: ImageData): [ImageData, any] {
	let threshold = 150;

	const originalMat = cv.matFromImageData(original);

    // Canny edge detection
    let image = new cv.Mat();
    cv.Canny(originalMat, image, threshold, threshold * 2);

    // Find contours
    let contours = new cv.MatVector();
    let hierarchy = new cv.Mat();
    cv.findContours(image, contours, hierarchy, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE);

    let boundRect = [];
    let centers = [];
    let radius = [];

    for (let i = 0; i < contours.size(); ++i) {
        let c = contours.get(i);
        let approx = new cv.Mat();
        cv.approxPolyDP(c, approx, 3, true);

        let rect = cv.boundingRect(approx);
        boundRect.push(rect);
    }

    // let drawing = new cv.Mat.zeros(image.rows, image.cols, cv.CV_8UC3);

    let minX = 99999;
    let minY = 99999;
    let maxX = 0;
    let maxY = 0;

    for (let i = 0; i < boundRect.length; ++i) {
        let rect = boundRect[i];

        if (rect.x < minX) minX = rect.x;
        if (rect.x + rect.width > maxX) maxX = rect.x + rect.width;
        if (rect.y < minY) minY = rect.y;
        if (rect.y + rect.height > maxY) maxY = rect.y + rect.height;
    }

	let newWidth = maxX - minX;
	let newHeight = maxY - minY;

	let newDimension = Math.max(newWidth, newHeight);

    // Crop and resize
    let roi = originalMat.roi(new cv.Rect(minX, minY, Math.min(newDimension, image.rows - minX), Math.min(newDimension, image.cols - minY)));

	let resized = new cv.Mat();
    let size = new cv.Size(256, 256);
    cv.resize(roi, resized, size, 0, 0, cv.INTER_AREA);

    // Clean up
    image.delete();
	contours.delete();
	hierarchy.delete();
    roi.delete();

    return [matToImageData(resized), resized];
}


export function resizeImage(image: Mat, size: [number, number]): Mat {
	let resized = new cv.Mat();
    let cvSize = new cv.Size(size[0], size[1]);

    cv.resize(image, resized, cvSize, 0, 0, cv.INTER_AREA);

	return resized;
}


export function matToImageData(mat: any) {
    let img = new ImageData(mat.cols, mat.rows);

    if (mat.type() === cv.CV_8UC1) {
        // Grayscale to RGBA
        for (let i = 0; i < mat.data.length; i++) {
            let val = mat.data[i];
            img.data[i * 4] = val;
            img.data[i * 4 + 1] = val;
            img.data[i * 4 + 2] = val;
            img.data[i * 4 + 3] = 255;
        }
    } else if (mat.type() === cv.CV_8UC3) {
        // BGR to RGBA
        for (let i = 0; i < mat.rows * mat.cols; i++) {
            img.data[i * 4] = mat.data[i * 3 + 2];     // R
            img.data[i * 4 + 1] = mat.data[i * 3 + 1]; // G
            img.data[i * 4 + 2] = mat.data[i * 3];     // B
            img.data[i * 4 + 3] = 255;
        }
    } else if (mat.type() === cv.CV_8UC4) {
        // Already RGBA
        img.data.set(mat.data);
    } else {
        throw new Error('Unsupported cv.Mat type: ' + mat.type());
    }

    return img;
}


export function calculateSoftmax(network_outputs: Float32Array<ArrayBufferLike>): number[] {
	let sum = 0;
	let exponentials: number[] = [];

	for (const output of network_outputs) {
		const expValue = Math.exp(output);
		sum += expValue;
		exponentials.push(expValue);
	}

	let probabilities: number[] = [];

	for (const output of exponentials) {
		probabilities.push(output / sum);
	}

	return probabilities;
}
