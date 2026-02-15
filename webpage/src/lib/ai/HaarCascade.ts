import { Point, Rect, type Mat } from "@techstark/opencv-js";
import { cv } from "../ai/LoadModel";
import haarPath from "/assets/detector/haarcascade_frontalface_default.xml?url";


const xml_path = "haarcascade_frontalface_default.xml";
let haar_loaded = false;


export async function detectFace(src: Mat): Promise<Mat | undefined> {
	// const faceCascade = new cv.CascadeClassifier();

	// if (!haar_loaded) {
	// 	await loadHaarCascade();
	// }
	
	// faceCascade.load(xml_path);

	let gray = new cv.Mat();
	cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);

	let faces = new cv.RectVector();

	// Detect faces
	let msize = new cv.Size(0, 0);
	// faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0);

	let selectedRoi = undefined;
	let selectedFaceSize = 0;

	// const faceWithMargin = new cv.Rect(
	// 	0.298828125 * src.size().width,
	// 	0.30078125 * src.size().height,
	// 	0.6796875 * src.size().width - 0.298828125 * src.size().width,
	// 	0.849609375 * src.size().height - 0.30078125 * src.size().height,
	// );

	// THE GOOD ONE!
	const faceWithMargin = new cv.Rect(
		0.0502 * src.size().width,
		0.1361 * src.size().height,
		0.9283 * src.size().width - 0.0502 * src.size().width,
		Math.min(1.0143 * src.size().height - 0.1361 * src.size().height, src.size().height - 0.1361 * src.size().height),
	);

	return src.roi(faceWithMargin);

	for (let i = 0; i < faces.size(); ++i) {
		const face = faces.get(i);
		const xoffset = 0;
		const yoffset = -20;
		const faceWithMargin = new cv.Rect(
			Math.max(face.x - xoffset, 0),
			Math.max(face.y - yoffset, 0),
			Math.min(face.width + xoffset * 2, 999999),
			Math.min(face.height + yoffset * 2, 99999),
		);

		// let point1 = new cv.Point(
		// 	faces.get(i).x,
		// 	faces.get(i).y
		// );

		// let point2 = new cv.Point(
		// 	face.x + face.width,
		// 	face.y + face.height
		// );

		console.log(face.width, face.height);

		if (face.width * face.height > selectedFaceSize) {
			selectedRoi = src.roi(faceWithMargin);
			selectedFaceSize = face.width * face.height;
		}

		// cv.rectangle(src, point1, point2, [255, 0, 0, 255]);
	}

	return selectedRoi!;
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