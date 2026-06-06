const faceAlignmentPromise = import("/static/js/face-alignment.mjs");

const models = {
	"ResNet50": {
		"256": "resnet50_256",
		"128": "resnet50_128",
		"64": "resnet50_64",
	},
	"MobileNet V3 Small": {
		"256": "mobilenet_v3_small_256",
		"128": "mobilenet_v3_small_128",
		"64": "mobilenet_v3_small_64",
	},
	"MobileNet V3 Large": {
		"256": "mobilenet_v3_large_256",
		"128": "mobilenet_v3_large_128",
		"64": "mobilenet_v3_large_64",
	},
	"ResNet18": {
		"256": "resnet18_256",
		"128": "resnet18_128",
		"64": "resnet18_64",
	},
	"EfficientNet-b0": {
		"256": "efficientnet-b0_256",
		"128": "efficientnet-b0_128",
		"64": "efficientnet-b0_64",
	}
};

const colors = [
	[47, 79, 79], // darkslategray
	[139, 69, 19], // saddlebrown
	[0, 100, 0], // darkgreen
	[75, 0, 130], // indigo
	[255, 0, 0], // red
	[0, 206, 209], // darkturquoise
	[255, 165, 0], // orange
	[255, 255, 0], // yelllow
	[0, 255, 0], // lime
	[0, 0, 255], // blue
	[255, 0, 255], // fuchsia
	[30, 144, 255], // dodgerblue
	[152, 251, 152], // palegreen
	[255, 218, 185], // peachpuff
	[255, 105, 180], // hotpink
];

let haarLoaded = false;

const loadedModels = {};

async function downloadAndLoadHaar(faceAlignment, haarPath) {
	if (haarLoaded) {
		return;
	}

	const data = new Promise((resolve) => {
		let request = new XMLHttpRequest();
		request.open('GET', haarPath, true);

		request.responseType = 'arraybuffer';
		request.onload = function(ev) {
			request = this;

			if (request.readyState === 4) {
				if (request.status === 200) {
					resolve(new Uint8Array(request.response));
				} else {
					console.error('Failed to load ' + haarPath + ' status: ' + request.status);
				}
			}
		};
		
		request.send();
	});

	await faceAlignment.loadHaarCascade("haarcascade_frontalface_default.xml", await data);

	haarLoaded = true;
}

function componentToHex(c) {
	c = Math.floor(c);
	const hex = c.toString(16);
	return hex.length == 1 ? "0" + hex : hex;
}

async function runModels(models, pointSize = 3) {
	const outputCanvas = document.getElementById("canvas");

	const inputFile = document.getElementById("input-file");
	const file = inputFile.files[0];

	const faceAlignment = await faceAlignmentPromise;
	await faceAlignment.loadOpenCV();
	console.log("OpenCV ready!")

	await downloadAndLoadHaar(faceAlignment, "/static/xml/haarcascade_frontalface_default.xml");
	console.log("Haar Cascade ready!")

	const labels = [];

	const imageData = await faceAlignment.getImageDataFromFile(file);
	const originalMat = faceAlignment.imageDataToMat(imageData);
	const faceData = faceAlignment.detectFace(originalMat);

	if (!faceData || !faceData.mat) {
		console.error("No face detected");

		return undefined;
	}

	let colorIndex = 0;

	for (const modelData of models) {
		const { model, size, modelPath } = modelData;

		const imageTensor = faceAlignment.imageDataToTensor(
			faceAlignment.matToImageData(
				faceAlignment.resizeImage(
					faceData.mat,
					[size, size]
				)
			)
		);

		console.log(loadedModels);

		const loadedModel = loadedModels[modelPath] !== undefined
			? loadedModels[modelPath]
			: await faceAlignment.loadModel(`/models/${modelPath}.onnx`, true, "ort/");

		loadedModels[modelPath] = loadedModel;
		console.log(loadedModels);

		const modelResults = await loadedModel.runModel(imageTensor);

		const color = [...colors[colorIndex], 255];
		colorIndex++;

		faceAlignment.drawFacePointsWithOffset(originalMat, faceData.mat, modelResults, faceData.offset, color, pointSize);

		labels.push({
			model,
			size,
			colorHex: `#${componentToHex(color[0])}${componentToHex(color[1])}${componentToHex(color[2])}`,
		});

		// results.push({
		// 	model,
		// 	size,
		// 	modelPath,
		// 	results: modelResults,
		// });
	}

	const paintedImageData = faceAlignment.matToImageData(originalMat);
	outputCanvas.width = paintedImageData.width;
	outputCanvas.height = paintedImageData.height;

	faceAlignment.showImageDataInCanvas(
		paintedImageData,
		outputCanvas,
	);

	console.log(labels);

	return labels;

	// console.log("Loaded model: ", model);

	// const results = await faceAlignment.executeModel({
	// 	file,
	// 	annotations: undefined,
	// 	model,
	// 	inputSize: modelSize,
	// 	canvas: outputCanvas,
	// });
}

document.addEventListener('alpine:init', () => {
	Alpine.data('demo', () => ({
		selectedModel: '',
		selectedSize: '',
		selectedModels: [],
		hasLoaded: false,
		labels: [],

		init() {
			faceAlignmentPromise.then(() => {
				this.hasLoaded = true;
			});
		},
		
		run() {
			this.labels = runModels(this.selectedModels);
		},

		toggleModel(model, size) {
			const modelPath = models[model][size];
			const index = this.selectedModels.indexOf(modelPath);

			if (index >= 0) {
				this.selectedModels.splice(index, 1);
			} else {
				this.selectedModels.push({
					modelPath,
					model,
					size: Number.parseInt(size),
				});
			}
		},
	}));
});
