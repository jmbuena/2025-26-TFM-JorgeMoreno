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
	[0, 0, 255, 255], // blue
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
		frames: 0,
		fps: 0,
		lastFrameTime: 0,
		downloadingModel: false,
		faceAlignment: undefined,
		resolutionScaling: 1,
		resolution: undefined,
		optimizationInterpolation: false,

		async init() {
			faceAlignmentPromise.then((faceAlignment) => {
				this.hasLoaded = true;

				this.faceAlignment = faceAlignment;
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

			this.downloadingModel = true
			const loadedModel = await prepareModel(selectedModel.modelPath, this.useGpu, this.faceAlignment);
			this.downloadingModel = false;

			this.lastFrameTime = performance.now();

			const frameProcessingMethod = this.optimizationInterpolation
				? new InterpoledFrameProcessing(selectedModel, loadedModel, canvasContext, width, height)
				: new SimpleFrameProcessing(selectedModel, loadedModel, canvasContext, width, height);

			while (true) {
				const { done, value } = await reader.read();

				if (done || this.stop) {
					console.log("Camera done");

					if (value) {
						value.close();
					}

					break;
				}

				const currentResolution = [width * this.resolutionScaling, height * this.resolutionScaling];
				const imageRescale = this.resolutionScaling === 1 ? undefined : currentResolution;
				this.resolution = `${currentResolution[0]}x${currentResolution[1]}`;

				const now = performance.now();
				if (now - this.lastFrameTime > 1000) {
					this.lastFrameTime = now;
					this.fps = this.frames;
					this.frames = 0;
				}

				if (value) {
					context.drawImage(value, 0, 0);

					const imageData = context.getImageData(0, 0, width, height);

					frameProcessingMethod.process(imageData, value, this.frames, imageRescale);

					this.frames++;
				}
			}
		}
	}));
});


class SimpleFrameProcessing {
	constructor(
		selectedModel,
		loadedModel,
		canvasContext,
		width,
		height,
	) {
		this.selectedModel = selectedModel;
		this.loadedModel = loadedModel;
		this.canvasContext = canvasContext;
		this.width = width;
		this.height = height;
	}

	async process(frameImageData, frame, _frameNumber, imageRescale) {
		const result = await runModels(this.selectedModel, this.loadedModel, frameImageData, imageRescale, 4)
			.catch((e) => {
				console.error("Error trying to run the model...", e);

				return undefined;
			});
	
		if (!result) {
			this.canvasContext.drawImage(await createImageBitmap(frameImageData), 0, 0, this.width, this.height);
		} else {
			this.canvasContext.drawImage(await createImageBitmap(result.imageData), 0, 0, this.width, this.height);
		}

		frame.close();
	}
}


class InterpoledFrameProcessing {
	constructor(
		selectedModel,
		loadedModel,
		canvasContext,
		width,
		height,
	) {
		this.selectedModel = selectedModel;
		this.loadedModel = loadedModel;
		this.frameQueue = [];
		this.processesFrameQueues = [];
		this.canvasContext = canvasContext;
		this.width = width;
		this.height = height;
	}

	async process(frameImageData, frame, frameNumber, imageRescale) {
		// First frame, process it and store it
		// Second frame, store imageData
		// Third frame, process it, store it, and show the first frame
		// Fourth frame, interpolate first and third frames over the second frame, show second frame
		// Fifth frame, process frame, store it, and show the third frame

		if (frameNumber === 0) { // First frame
			const result = await runModelsLazy(this.selectedModel, this.loadedModel, frameImageData, imageRescale, 4)
				.catch((e) => {
					console.error("Error trying to run the model...", e);

					return undefined;
				});
			
			if (result === undefined) {
				this.frameQueue.push({
					ok: false,
					imageData: frameImageData,
				});
			} else {
				this.frameQueue.push({
					ok: true,
					...result,
				});
			}
		} else if (frameNumber === 1) { // Second frame
			this.frameQueue.push({
				ok: false,
				imageData: frameImageData,
			});
		} else if (frameNumber % 2 === 0) { // Even frames > 1
			const result = await runModelsLazy(this.selectedModel, this.loadedModel, frameImageData, imageRescale, 4)
				.catch((e) => {
					console.error("Error trying to run the model...", e);

					return undefined;
				});
			
			if (result === undefined) {
				this.frameQueue.push({
					ok: false,
					imageData: frameImageData,
				});
			} else {
				this.frameQueue.push({
					ok: true,
					...result,
				});
			}

			this.frameQueue.push({
				ok: true,
				...result,
			});

			const firstFrameData = this.frameQueue.at(0);
			
			this.canvasContext.drawImage(await createImageBitmap(firstFrameData.imageData), 0, 0, this.width, this.height);
		} else { // Odd frames > 2
			const firstFrameData = this.frameQueue.shift();
			const secondFrameData = this.frameQueue.shift();
			const thirdFrameData = this.frameQueue.at(0);

			// We can interpolate
			if (firstFrameData.ok && thirdFrameData.ok) {
				const secondFrameInterpolatedLandmarks = new Uint8Array(firstFrameData.landmarks.length);
				for (let i = 0; i < firstFrameData.landmarks.length; i++) {
					secondFrameInterpolatedLandmarks[i] =
						(firstFrameData.landmarks[i] + thirdFrameData.landmarks[i]) / 2;
				}

				const color = colors[0];
				const pointSizeRelative = Math.max(1, (faceData.mat.size().width / 800) * pointSize);

				faceAlignment.drawFacePointsWithOffset(originalMat, faceData.mat, modelResults.output, faceData.offset, color, pointSizeRelative);
			}

			if (firstFrameData.ok) {
				firstFrameData.cleanup();
			}

			if (secondFrameData.ok) {
				firstFrameData.cleanup();
			}
		}

		

		// if (frameNumber % 2 !== 0) {
		// 	// Los frames impares procesan los modelos
		// 	const result = await runModelsLazy(this.selectedModel, this.loadedModel, frameImageData, imageRescale, 4)
		// 		.catch((e) => {
		// 			console.error("Error trying to run the model...", e);

		// 			return undefined;
		// 		});

		// 	this.frameQueue.push(result);

		// 	if (this.frameQueue.length === 3) {
		// 		const firstFrameInQueue = this.frameQueue.at(0);

		// 		this.canvasContext.drawImage(await createImageBitmap(firstFrameInQueue), 0, 0, this.width, this.height);
		// 	}
		// } else {
		// 	// Los frames pares calculan la interpolación de los frames impares (si hubiera suficientes)
		// 	if (this.frameQueue.length === 2) {
		// 		const firstFrameInQueue = this.frameQueue.shift();
		// 		const secondFrameInQueue = this.frameQueue.shift();
		// 		const thirdFrameInQueue = this.frameQueue.at(0);

		// 		const secondFrameLandmarks = new Uint8Array(firstFrameInQueue.landmarks.length);

		// 		for (let i = 0; i < firstFrameInQueue.landmarks.length; i++) {
		// 			secondFrameLandmarks[i] = (firstFrameInQueue.landmarks[i] + thirdFrameInQueue.landmarks[i]) / 2;
		// 		}

		// 		this.canvasContext.drawImage(await createImageBitmap(secondFrameInQueue.imageData), 0, 0, this.width, this.height);

		// 		firstFrameInQueue.cleanup();
		// 		secondFrameInQueue.cleanup();
		// 	}

		// 	this.frameQueue.push(frameImageData);
		// }

		// frame.close();
	}
}


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


async function prepareModel(modelPath, useGpu, faceAlignment) {
	const loadedModel = loadedModels[modelPath] !== undefined
		? loadedModels[modelPath]
		: await faceAlignment.loadModel(`/models/${modelPath}.onnx`, useGpu, "ort/");

	loadedModels[modelPath] = loadedModel;

	return loadedModel;
}


async function runModels(modelData, loadedModel, imageData, imageRescale, pointSize = 3) {
	const faceAlignment = await faceAlignmentPromise;

	let originalMat = faceAlignment.imageDataToMat(imageData);
	if (imageRescale !== undefined) {
		const unusedOriginalMat = originalMat;
		originalMat = faceAlignment.resizeImage(originalMat, imageRescale);
		unusedOriginalMat.delete();
	}
	const faceData = faceAlignment.detectFace(originalMat);

	if (!faceData || !faceData.mat) {
		originalMat.delete();

		return undefined;
	}

	const { size } = modelData;

	const faceResized = faceAlignment.resizeImage(faceData.mat, [size, size]);

	const imageTensor = faceAlignment.imageDataToTensor(faceAlignment.matToImageData(faceResized));

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
		landmarks: modelResults.output,
	};
}


async function runModelsLazy(modelData, loadedModel, imageData, imageRescale, pointSize = 3) {
	const faceAlignment = await faceAlignmentPromise;

	let originalMat = faceAlignment.imageDataToMat(imageData);
	
	if (imageRescale !== undefined) {
		const unusedOriginalMat = originalMat;
		originalMat = faceAlignment.resizeImage(originalMat, imageRescale);
		unusedOriginalMat.delete();
	}

	const faceData = faceAlignment.detectFace(originalMat);

	if (!faceData || !faceData.mat) {
		originalMat.delete();

		return undefined;
	}

	const { size } = modelData;

	const faceResized = faceAlignment.resizeImage(faceData.mat, [size, size]);

	const imageTensor = faceAlignment.imageDataToTensor(faceAlignment.matToImageData(faceResized));

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
	
	return {
		elapsed: modelResults.elapsed.toFixed(2),
		imageData: paintedImageData,
		landmarks: modelResults.output,
		faceData,
		originalMat,
		cleanup() {
			faceData.mat.delete();
			originalMat.delete();
		},
	};
}
