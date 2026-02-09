<script lang="ts">
	import { onMount } from "svelte";
	import { cv } from "../ai/LoadModel";
    import { detectFace } from "../ai/HaarCascade";
    import { drawFacePoints } from "../ai/FaceHelpers";
	import { ClassificationModel } from "../ai/RunModel";
	import ClassificationResult from "./ClassificationResult.svelte";
	import { imageDataToTensor, extractDrawing, showImageDataInCanvas, matToImageData, resizeImage } from "../ai/AiHelpers";

	let canvas: HTMLCanvasElement;
	let previewCanvas: HTMLCanvasElement;
	let paintingMode: "painting" | "erasing" | undefined = undefined;
	let lastPosition: { x: number; y: number } = { x: 0, y: 0 };
	let ctx: CanvasRenderingContext2D;
	let previewCtx: CanvasRenderingContext2D;
	let model: ClassificationModel;
	let processing = $state(false);
	let error: string | undefined = $state(undefined);

	let showResults: Array<[string, number]> = $state([]);

	export function getDrawing(): ImageData {
		return ctx.getImageData(0, 0, canvas.width, canvas.height);
	}

	async function run(): Promise<void> {
		processing = true;

		const [extracted, imageMat] = extractDrawing(getDrawing());
		const tensor = imageDataToTensor(extracted);

		// To preview
		const imageData = tensor.toImageData();
		showImageDataInCanvas(imageData, previewCtx);

		const results = await model.runModel(tensor);
		drawFacePoints(imageMat, results);

		showImageDataInCanvas(matToImageData(imageMat), previewCtx);

		processing = false;
	}

	function clear(): void {
		ctx.fillStyle = "#fff";
		ctx.fillRect(0, 0, canvas.width, canvas.height);

		previewCtx.fillStyle = "#fff";
		previewCtx.fillRect(0, 0, canvas.width, canvas.height);
	}

	onMount(() => {
		ctx = canvas.getContext("2d", {
			willReadFrequently: true,
		})!;

		previewCtx = previewCanvas.getContext("2d", {})!;

		clear();

		canvas.addEventListener("mousedown", (event) => {
			if (processing) {
				return;
			}

			paintingMode = "painting";

			updateMousePosition(event);
		});

		canvas.addEventListener("mousemove", (event) => {
			if (!paintingMode) {
				return;
			}

			ctx.beginPath();
			ctx.lineWidth = 5;
			ctx.lineCap = "round";
			ctx.strokeStyle = "black";

			ctx.moveTo(lastPosition.x, lastPosition.y);

			updateMousePosition(event);

			ctx.lineTo(lastPosition.x, lastPosition.y);
			ctx.stroke();
		});

		canvas.addEventListener("mouseup", (event) => {
			paintingMode = undefined;
		});

		model = new ClassificationModel();
	});

	function updateMousePosition(event: MouseEvent) {
		lastPosition.x = event.clientX - canvas.offsetLeft;
		lastPosition.y = event.clientY - canvas.offsetTop;
	}

	async function runModelFromImage(event: InputEvent): Promise<void> {
		processing = true;
		const files: Array<File> = (event.target! as any).files as Array<File>;

		const file = files[0];

		const offscreenContext = await drawFileToCanvas(file);

		const imageData = offscreenContext.getImageData(
			0,
			0,
			offscreenContext.canvas.width,
			offscreenContext.canvas.height,
		);

		const originalMat = cv.matFromImageData(imageData);
		const faceMat = await detectFace(originalMat)
			.catch((error) => console.error("ERROR: " + error));

		if (!faceMat) {
			error = 'No face found in image...';
			return;
		}

		showImageDataInCanvas(matToImageData(resizeImage(faceMat, [512, 512])), ctx);

		const resizedImageData = matToImageData(resizeImage(faceMat, [256, 256]));

		const tensor = imageDataToTensor(resizedImageData);

		const results = await model.runModel(tensor);

		const resizedMat = cv.matFromImageData(resizedImageData);
		drawFacePoints(resizedMat, results);

		showImageDataInCanvas(matToImageData(resizedMat), previewCtx);
		
		processing = false;
	}

	async function getImageArrayBuffer(file: File): Promise<ArrayBuffer> {
		return new Promise((resolve) => {
			const fileReader = new FileReader();

			fileReader.onload = function() {
				resolve(fileReader.result as ArrayBuffer);
			};

			fileReader.readAsArrayBuffer(file);
		});
	}

	async function drawFileToCanvas(file: File): Promise<OffscreenCanvasRenderingContext2D> {
		return new Promise((resolve, reject) => {
			const fileReader = new FileReader();

			fileReader.onload = function() {
				const image = new Image();
				
				image.onload = function() {
					const offscreen = new OffscreenCanvas(image.width, image.height);
					const offscreenContext = offscreen.getContext("2d")!;
					
					offscreenContext.drawImage(image, 0, 0);

					resolve(offscreenContext);
				}

				image.src = fileReader.result as string;
			};

			fileReader.readAsDataURL(file);
		});
	}

	async function getImageSize(file: File): Promise<[number, number]> {
		return new Promise((resolve, reject) => {
			const fileReader = new FileReader();

			fileReader.onload = function() {
				const image = new Image();
				
				image.onload = function() {
					resolve([image.width, image.height]);
				}

				image.src = fileReader.result as string;
			};

			fileReader.readAsDataURL(file);
		});
	}
</script>

<div class="space-y-2">
	<div class="w-full flex justify-between gap-x-72">
		<div>
			<canvas
				bind:this={canvas}
				width="512"
				height="512"
				class="bg-neutral-700 border rounded cursor-crosshair"
			></canvas>
		
			<input type="file" onchange={runModelFromImage as any}>
		</div>

		<div class="relative flex justify-center">
			<p class="absolute text-black font-semibold">PROCESSED PREVIEW</p>

			<canvas
				bind:this={previewCanvas}
				width="256"
				height="256"
				class="bg-neutral-700 border rounded cursor-crosshair"
			></canvas>
		</div>
	</div>

	<div class="flex gap-2 justify-center">
		<button onclick={run} class="px-2 py-1 border rounded disabled:bg-neutral-700" disabled={processing}>{processing ? "Processing..." : "Run AI"}</button>
		<button onclick={clear} class="px-2 py-1 border rounded disabled:bg-neutral-700" disabled={processing}>Clear</button>
	</div>

	{#if error}
		<div class="text-center text-red-500 font-bold">
			Error: {error}
		</div>
	{/if}

	<div class="flex flex-col gap-y-2">
		{#each showResults as res}
			<ClassificationResult className={res[0]} probability={res[1]} />
		{/each}
	</div>

</div>
