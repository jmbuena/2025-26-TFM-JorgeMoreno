import { Point, Rect, Size, type Mat } from "@techstark/opencv-js";
import { cv } from "../ai/LoadModel";
import haarPath from "/assets/detector/haarcascade_frontalface_default.xml?url";


const xml_path = "haarcascade_frontalface_default.xml";
let haar_loaded = false;


export async function detectFace(src: Mat): Promise<{ mat: Mat, offset: Rect } | undefined> {
	const faceCascade = new cv.CascadeClassifier();

	if (!haar_loaded) {
		await loadHaarCascade();
	}
	
	faceCascade.load(xml_path);

	let gray = new cv.Mat();
	cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);

	let faces = new cv.RectVector();

	// Detect faces
	let msize = new cv.Size(0, 0);
	faceCascade.detectMultiScale(gray, faces, 1.3, 3, 0);

	let selectedRoi = undefined;
	let selectedFaceSize = 0;

	// const faceWithMargin = new cv.Rect(
	// 	0.298828125 * src.size().width,
	// 	0.30078125 * src.size().height,
	// 	0.6796875 * src.size().width - 0.298828125 * src.size().width,
	// 	0.849609375 * src.size().height - 0.30078125 * src.size().height,
	// );

	// THE GOOD ONE!
	// const faceWithMargin = new cv.Rect(
	// 	0.0502 * src.size().width,
	// 	0.1361 * src.size().height,
	// 	0.9283 * src.size().width - 0.0502 * src.size().width,
	// 	Math.min(1.0143 * src.size().height - 0.1361 * src.size().height, src.size().height - 0.1361 * src.size().height),
	// );

	const { width: imageWidth, height: imageHeight } = src.size();
	let selectedRoiRect: Rect;

	const enlargementPercent = 0.3;

	for (let i = 0; i < faces.size(); ++i) {
		const face = faces.get(i);
		// const xoffset = 100;
		// const yoffset = 150;

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
			// selectedRoiRect = new cv.Rect(
			// 	faceWithMargin.x - face.x,
			// 	faceWithMargin.y - face.y,
			// 	faceWithMargin.width - face.width,
			// 	faceWithMargin.height - face.height
			// );
			selectedRoiRect = faceWithMargin;
		}
		
		// cv.rectangle(src, new cv.Point(face.x, face.y), new cv.Point(face.x + face.width, face.y + face.height), [255, 0, 0, 255]);
	}

	return {
		mat: selectedRoi!,
		offset: selectedRoiRect!,
	};
}

async function loadHaarCascade(): Promise<void> {
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