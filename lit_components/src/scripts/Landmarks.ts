
export class Landmark {
	constructor(
		readonly x: number,
		readonly y: number,
	) {}

	euclideanDistance(otherLandmark: Landmark): number {
		return Math.abs(this.x - otherLandmark.x +  this.y - otherLandmark.y);
	}

	toString(): string {
		return `(${this.x.toFixed(4)}, ${this.y.toFixed(4)})`;
	}
}


export function processLandmarks(landmarksArray: Float32Array): Array<Landmark> {
	const landmarks: Array<Landmark> = new Array(landmarksArray.length / 2);

	for (let i = 0; i < landmarksArray.length; i += 2) {
		const x = landmarksArray[i];
		const y = landmarksArray[i + 1];

		landmarks[i / 2] = new Landmark(x, y);
	}

	return landmarks;
}
