import { type CV, type Mat } from "@techstark/opencv-js";
import cvReadyPromise from "@techstark/opencv-js";


export let cv: CV;

export async function loadOpenCV(): Promise<void> {
	cv = await cvReadyPromise;

	console.log("OpenCV ready...");
}


export function resizeImage(image: Mat, size: [number, number]): Mat {
	let resized = new cv.Mat();
	let cvSize = new cv.Size(size[0], size[1]);

	cv.resize(image, resized, cvSize, 0, 0, cv.INTER_AREA);

	return resized;
}


export function matToImageData(mat: Mat): ImageData {
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


export function imageDataToMat(imageData: ImageData): Mat {
	return cv.matFromImageData(imageData);
}


export function drawFacePointsWithOffset(
	image: Mat,
	faceMask: Mat,
	data: Float32Array,
	offset: { x: number, y: number },
	color: [number, number, number, number],
	size: number,
): void {
	for (let i = 0; i < data.length; i += 2) {
		const x = data[i] * faceMask.size().width + offset.x;
		const y = data[i + 1] * faceMask.size().height + offset.y;
 
		cv.circle(image, new cv.Point(x, y), size, color, -1);
	}
}
