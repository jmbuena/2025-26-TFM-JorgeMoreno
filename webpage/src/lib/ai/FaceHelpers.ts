import type { Mat } from "@techstark/opencv-js";
import { cv } from "./LoadModel";


export function drawFacePoints(image: Mat, data: Float32Array): void {
	for (let i = 0; i < data.length; i += 2) {
		const x = data[i] * image.size().width;
		const y = data[i + 1] * image.size().height;
 
		cv.circle(image, new cv.Point(x, y), 1, [255, 255, 0, 255], -1);
	}
}
