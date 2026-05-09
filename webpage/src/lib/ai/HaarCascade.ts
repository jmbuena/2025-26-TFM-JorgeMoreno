import { cv } from "../ai/LoadModel";
import { CascadeClassifier, Rect, type Mat } from "@techstark/opencv-js";
import haarPath from "/assets/detector/haarcascade_frontalface_default.xml?url";


const xml_path = "haarcascade_frontalface_default.xml";

export let isHaarReady = false;
let faceCascade: CascadeClassifier | undefined = undefined;


export async function loadHaarCascade(): Promise<void> {
	if (!isHaarReady) {
		await downloadHaarCascade();
		
		isHaarReady = true;
	
		faceCascade = new cv.CascadeClassifier();
		faceCascade.load(xml_path);
	}
}


export function detectFace(src: Mat): { mat: Mat, offset: Rect } | undefined {
	let gray = new cv.Mat();
	cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
	
	let faces = new cv.RectVector();
	
	// Detect faces
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
		
		// Uncomment to show the face ROI in the imagef
		// cv.rectangle(src, new cv.Point(face.x, face.y), new cv.Point(face.x + face.width, face.y + face.height), [255, 0, 0, 255]);
	}

	return {
		mat: selectedRoi!,
		offset: selectedRoiRect!,
	};
}


export function getFaceFromAnnotations(imageMat: Mat, face: Record<'x' | 'y' | 'w' | 'h', number>) {
	const faceWithMargin = new cv.Rect(new cv.Point(face.x, face.y), new cv.Size(face.w, face.h));
	
	return imageMat.roi(faceWithMargin);
}


function downloadHaarCascade(): Promise<void> {
	return new Promise((resolve) => {
		let request = new XMLHttpRequest();
		request.open('GET', haarPath, true);

		request.responseType = 'arraybuffer';
		request.onload = function(ev) {
			request = this;

			if (request.readyState === 4) {
				if (request.status === 200) {
					let data = new Uint8Array(request.response);
					cv.FS_createDataFile('/', xml_path, data, true, false, false);
					resolve();
				} else {
					console.error('Failed to load ' + haarPath + ' status: ' + request.status);
				}
			}
		};
		
		request.send();
	});
}
