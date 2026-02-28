import type { Mat } from "@techstark/opencv-js";
import { cv } from "./LoadModel";


export function drawFacePoints(image: Mat, data: Float32Array, color: Color, size: number): void {
	for (let i = 0; i < data.length; i += 2) {
		const x = data[i] * image.size().width;
		const y = data[i + 1] * image.size().height;
 
		cv.circle(image, new cv.Point(x, y), size, color.toArray(), -1);
	}
}


export function drawFacePointsWithOffset(
	image: Mat,
	faceMask: Mat,
	data: Float32Array,
	offset: { x: number, y: number },
	color: Color,
	size: number,
): void {
	// cv.circle(image, new cv.Point(offset.x, offset.y), 10, [0, 255, 0, 255], -1);
	// cv.circle(image, new cv.Point(offset.x + faceMask.size().width, offset.y + faceMask.size().height), 10, [0, 255, 0, 255], -1);

	for (let i = 0; i < data.length; i += 2) {
		const x = data[i] * faceMask.size().width + offset.x;
		const y = data[i + 1] * faceMask.size().height + offset.y;
 
		cv.circle(image, new cv.Point(x, y), size, color.toArray(), -1);
	}
}


export class Color {
	static readonly RED = new Color(255, 0, 0);
	static readonly GREEN = new Color(0, 255, 0);
	static readonly BLUE = new Color(0, 0, 255);


	constructor(
		protected red: number,
		protected green: number,
		protected blue: number,
		protected alpha: number = 255,
	) {}


	static rgb(red: number, green: number, blue: number): Color {
		return new Color(red, green, blue);
	}


	toArray(): [number, number, number, number] {
		return [this.red, this.green, this.blue, this.alpha];
	}
}
