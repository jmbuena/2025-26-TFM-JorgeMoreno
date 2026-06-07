const faceAlignmentPromise = import("/static/js/face-alignment.mjs");


const models = {
	"ResNet18": {
		"256": "resnet18_256",
		"128": "resnet18_128",
		"64": "resnet18_64",
	},
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
	"EfficientNet-b0": {
		"256": "efficientnet-b0_256",
		"128": "efficientnet-b0_128",
		"64": "efficientnet-b0_64",
	}
};

const colors = [
	[0, 0, 255], // blue
	[255, 0, 0], // red
	[0, 255, 0], // lime
	[75, 0, 130], // indigo
	[255, 165, 0], // orange
	[255, 105, 180], // hotpink
	[139, 69, 19], // saddlebrown
	[0, 100, 0], // darkgreen
	[0, 206, 209], // darkturquoise
	[255, 255, 0], // yelllow
	[255, 0, 255], // fuchsia
	[30, 144, 255], // dodgerblue
	[152, 251, 152], // palegreen
	[255, 218, 185], // peachpuff
	[47, 79, 79], // darkslategray
];

let haarLoaded = false;

const loadedModels = {};


document.addEventListener('alpine:init', () => {
	Alpine.data('live', () => ({
		hasLoaded: false,
		useGpu: false,
		isProcessing: false,
		stop: false,
		selectedModelName: undefined,
		selectedModelSize: undefined,

		async init() {
			faceAlignmentPromise.then(() => {
				this.hasLoaded = true;
			});

			if ("serviceWorker" in navigator) {
				await navigator.serviceWorker.register("/webcache.js")
					.then(() => {
						console.log("Cache ready!");
					})
					.catch(() => {
						console.error("Service worker 'webcache' could not be registered");
					});
			}
		},

		selectModel(model) {
			this.selectedModelName = model;
		},

		selectModelSize(size) {
			this.selectedModelSize = size;
		},

		stopProcessing() {
			this.stop = true;
			this.isProcessing = false;
		},

		async startRecording() {
			this.isProcessing = true;
			this.stop = false;

			await prepareLibraries();

			const canvasElement = this.$refs.canvas;
			const canvasContext = canvasElement.getContext("2d");

			const stream = await navigator.mediaDevices
				.getUserMedia({ video: true, audio: false })
				.catch((err) => {
					console.error(`An error occurred: ${err}`);
					return undefined;
				});

			if (stream === undefined) {
				console.error("No camera available!");
				return;
			}

			const track = stream.getVideoTracks()[0];

			const { width, height } = stream.getVideoTracks()[0].getSettings();
			const offscreenCanvas = new OffscreenCanvas(width, height);
			const context = offscreenCanvas.getContext("2d", {
				willReadFrequently: true,
			});

			canvasElement.width = width;
			canvasElement.height = height;

			const processor = new MediaStreamTrackProcessor(track);
			const reader = processor.readable.getReader();

			const selectedModel = {
				modelPath: models[this.selectedModelName][this.selectedModelSize],
				size: Number.parseInt(this.selectedModelSize),
				model: this.selectedModelName,
			};

			while (true) {
				const { done, value } = await reader.read();

				if (done || this.stop) {
					console.log("Camera done");

					if (value) {
						value.close();
					}

					break;
				}

				if (value) {
					context.drawImage(value, 0, 0);

					const imageData = context.getImageData(0, 0, width, height);
					const result = await runModels(selectedModel, imageData, this.useGpu, 4)
						.catch((e) => {
							console.error("Error trying to run the model...", e);

							return undefined;
						});

					if (!result) {
						canvasContext.drawImage(await createImageBitmap(imageData), 0, 0, width, height);
						value.close();

						continue;
					}

					canvasContext.drawImage(await createImageBitmap(result.imageData), 0, 0, width, height);
					value.close();
				}
			}
		}
	}));
});


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


async function prepareLibraries() {
	const faceAlignment = await faceAlignmentPromise;
	await faceAlignment.loadOpenCV();

	await downloadAndLoadHaar(faceAlignment, "/static/xml/haarcascade_frontalface_default.xml");
}


async function runModels(modelData, imageData, useGpu, pointSize = 3) {
	const faceAlignment = await faceAlignmentPromise;

	const originalMat = faceAlignment.imageDataToMat(imageData);
	const faceData = faceAlignment.detectFace(originalMat);

	if (!faceData || !faceData.mat) {
		originalMat.delete();

		return undefined;
	}

	const { size, modelPath } = modelData;

	const faceResized = faceAlignment.resizeImage(faceData.mat, [size, size]);

	const imageTensor = faceAlignment.imageDataToTensor(faceAlignment.matToImageData(faceResized));

	const loadedModel = loadedModels[modelPath] !== undefined
		? loadedModels[modelPath]
		: await faceAlignment.loadModel(`/models/${modelPath}.onnx`, useGpu, "ort/");

	loadedModels[modelPath] = loadedModel;

	const modelResults = await loadedModel.runModel(imageTensor);

	faceResized.delete();
	imageTensor.dispose();

	if (!modelResults) {
		originalMat.delete();
		faceData.mat.delete();

		return undefined;
	}

	const color = colors[0]

	const pointSizeRelative = Math.max(1, (faceData.mat.size().width / 800) * pointSize);

	faceAlignment.drawFacePointsWithOffset(originalMat, faceData.mat, modelResults.output, faceData.offset, color, pointSizeRelative);

	const paintedImageData = faceAlignment.matToImageData(originalMat);
	
	faceData.mat.delete();
	originalMat.delete();

	return {
		elapsed: modelResults.elapsed.toFixed(2),
		imageData: paintedImageData,
	}
}
