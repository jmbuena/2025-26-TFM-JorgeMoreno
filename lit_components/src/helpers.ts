
export async function getImageDataFromFile(file: File): Promise<ImageData> {
	const offscreenContext = await drawFileToOffscreenCanvas(file);

	return offscreenContext.getImageData(
		0,
		0,
		offscreenContext.canvas.width,
		offscreenContext.canvas.height,
	);
}


async function drawFileToOffscreenCanvas(file: File): Promise<OffscreenCanvasRenderingContext2D> {
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


export async function showImageDataInCanvas(image: ImageData, canvas: HTMLCanvasElement): Promise<void> {
	const canvasCtx = canvas.getContext("2d")!;

	canvasCtx.clearRect(0, 0, image.width, image.height);
	canvasCtx.drawImage(await createImageBitmap(image), 0, 0, image.width, image.height);
}
