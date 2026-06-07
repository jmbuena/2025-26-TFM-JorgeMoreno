import { cv } from "./opencv";
import { type CascadeClassifier, type Rect, type Mat } from "@techstark/opencv-js";


export let isHaarReady = false;
let faceCascade: CascadeClassifier | undefined = undefined;


export async function loadHaarCascade(name: string, contents: Uint8Array<ArrayBufferLike>): Promise<void> {
	if (!isHaarReady) {
		cv.FS_createDataFile('/', name, contents, true, false, false);
		
		isHaarReady = true;
	
		faceCascade = new cv.CascadeClassifier();
		faceCascade.load(name);
	}
}


export function detectFace(src: Mat): { mat: Mat, offset: Rect } | undefined {
	let gray = new cv.Mat();
	cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
	
	let faces = new cv.RectVector();
	
	if (!faceCascade) {
		return undefined;
	}

	faceCascade.detectMultiScale(gray, faces, 1.3, 3, 0);
	gray.delete();

	let selectedRoi = undefined;
	let selectedFaceSize = 0;

	const { width: imageWidth, height: imageHeight } = src.size();
	let selectedRoiRect: Rect;

	const enlargementPercent = 0.3;

	let largestFaceIndex = 0;
	for (let i = 0; i < faces.size(); ++i) {
		const face = faces.get(i);

		if (face.width * face.height > selectedFaceSize) {
			largestFaceIndex = i;
		}
	}

	if (faces.size() === 0) {
		return undefined;
	}

	const face = faces.get(largestFaceIndex);
	faces.delete();

	if (face.width * face.height > selectedFaceSize) {
		let { x, y, width, height } = face;

		if (width > height) {
			const squareOffset = width - height;

			y = y - squareOffset;
			height = width;
		} else {
			const squareOffset = height - width;

			x = x - squareOffset;
			width = height;
		}

		x = Math.max(0, x - width * enlargementPercent);
		y = Math.max(0, y - height * enlargementPercent);

		let enlargedWidth = width + width * enlargementPercent * 2;
		let enlargedHeight = height + height * enlargementPercent * 2;

		if (enlargedWidth + x > imageWidth) {
			enlargedWidth = imageWidth - x;
		}

		if (enlargedHeight + y > imageHeight) {
			enlargedHeight = imageHeight - y;
		}

		width = enlargedWidth;
		height = enlargedHeight;

		const point = new cv.Point(x, y);
		const size = new cv.Size(width, height);

		const faceWithMargin = new cv.Rect(point, size);

		selectedRoi = src.roi(faceWithMargin);
		selectedFaceSize = face.width * face.height;
		selectedRoiRect = faceWithMargin;
	}

	return {
		mat: selectedRoi!,
		offset: selectedRoiRect!,
	};
}
