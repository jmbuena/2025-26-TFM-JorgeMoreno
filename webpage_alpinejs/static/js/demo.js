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

async function runModels(models, images, useGpu, pointSize = 3) {
	const faceAlignment = await faceAlignmentPromise;
	await faceAlignment.loadOpenCV();
	console.log("OpenCV ready!");

	await downloadAndLoadHaar(faceAlignment, "/static/xml/haarcascade_frontalface_default.xml");
	console.log("Haar Cascade ready!");

	console.log("useGPU", useGpu);

	const results = [];

	for (const image of images) {
		const labels = [];
	
		const imageData = await faceAlignment.getImageDataFromFile(image);
		const originalMat = faceAlignment.imageDataToMat(imageData);
		const faceData = faceAlignment.detectFace(originalMat);
	
		if (!faceData || !faceData.mat) {
			console.error("No face detected");

			originalMat.delete();

			results.push({
				labels: [],
				imageData,
			});
	
			continue;
		}
	
		let colorIndex = 0;

		const pointSizeRelative = Math.max(1, (faceData.mat.size().width / 800) * pointSize);

		for (const modelData of models) {
			const { model, size, modelPath } = modelData;

			const faceResized = faceAlignment.resizeImage(faceData.mat, [size, size]);
	
			const imageTensor = faceAlignment.imageDataToTensor(faceAlignment.matToImageData(faceResized));

			faceResized.delete();
	
			const loadedModel = loadedModels[modelPath] !== undefined
				? loadedModels[modelPath]
				: await faceAlignment.loadModel(`/models/${modelPath}.onnx`, useGpu, "ort/");
	
			loadedModels[modelPath] = loadedModel;
	
			const modelResults = await loadedModel.runModel(imageTensor);

			imageTensor.dispose();

			if (!modelResults) {
				continue;
			}

			const color = [...colors[colorIndex], 255];
			colorIndex++;
	
			faceAlignment.drawFacePointsWithOffset(originalMat, faceData.mat, modelResults.output, faceData.offset, color, pointSizeRelative);
	
			labels.push({
				model,
				size,
				colorHex: `#${componentToHex(color[0])}${componentToHex(color[1])}${componentToHex(color[2])}`,
				time: modelResults.elapsed.toFixed(2),
				style: {
					"background-color": `#${componentToHex(color[0])}${componentToHex(color[1])}${componentToHex(color[2])}`,
				},
			});
		}
	
		const paintedImageData = faceAlignment.matToImageData(originalMat);

		originalMat.delete();
		faceData.mat.delete();
	
		results.push({
			labels,
			imageData: paintedImageData,
		});
	}

	return results;
}

async function renderResults(results) {
	const faceAlignment = await faceAlignmentPromise;

	results.forEach((result, index) => {
		const canvas = document.getElementById("canvas_" + index);

		canvas.width = result.imageData.width;
		canvas.height = result.imageData.height;
		
		faceAlignment.showImageDataInCanvas(
			result.imageData,
			canvas,
		);
	});
}

document.addEventListener('alpine:init', () => {
	Alpine.data('demo', () => ({
		selectedModels: new Map(),
		hasLoaded: false,
		results: [],
		useGpu: false,
		selectedImages: [],
		isProcessing: false,

		async init() {
			faceAlignmentPromise.then(() => {
				this.hasLoaded = true;

				this.$nextTick(() => {
					const inputFile = document.getElementById("input-file");
					inputFile.addEventListener("change", () => {
						this.selectedImages = inputFile.files;
					});
				});
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
		
		async run() {
			this.isProcessing = true;
			console.log("USE GPU PLEASE", this.useGpu);

			this.results = await runModels(Array.from(this.selectedModels.values()), this.selectedImages, this.useGpu, 4);

			this.$nextTick(async () => {
				await renderResults(this.results);

				this.isProcessing = false;
			});
		},

		toggleModel(model, size) {
			const modelPath = models[model][size];

			if (this.selectedModels.has(modelPath)) {
				this.selectedModels.delete(modelPath);
			} else {
				this.selectedModels.set(modelPath, {
					modelPath,
					model,
					size: Number.parseInt(size),
				});
			}
		},
	}));
});
