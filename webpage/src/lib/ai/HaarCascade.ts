import { Rect, type Mat } from "@techstark/opencv-js";
import { cv } from "../ai/LoadModel";
import haarPath from "/assets/detector/haarcascade_frontalface_default.xml?url";


const xml_path = "haarcascade_frontalface_default.xml";
let haar_loaded = false;


export async function detectFace(src: Mat): Promise<Mat | undefined> {
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
	faceCascade.detectMultiScale(gray, faces, 1.5, 3, 0, msize, msize);

	let selectedRoi = undefined;
	let selectedFaceSize = 0;

	for (let i = 0; i < faces.size(); ++i) {
		const face = faces.get(i);
		const offset = 100;
		const faceWithMargin = new cv.Rect(face.x - offset, face.y - offset, face.width + offset * 2, face.height + offset * 2);

		let roiSrc = src.roi(faceWithMargin);

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
			selectedRoi = roiSrc;
			selectedFaceSize = face.width * face.height;
		} else {
			roiSrc.delete();
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