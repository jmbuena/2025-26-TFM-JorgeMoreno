import { cv } from "./LoadModel";


export function drawFacePoints(image: any, data: Float32Array): void {
	console.log(data.length, data.length / 2);
	for (let i = 0; i < data.length; i += 2) {
		const x = data[i] * 256;
		const y = data[i + 1] * 256;
 
		cv.circle(image, new cv.Point(x, y), 1, [255, 0, 0, 0], -1);
	}
}
