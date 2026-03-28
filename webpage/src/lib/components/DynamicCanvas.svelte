<script lang="ts">
    import { onMount } from "svelte";
    import { applyImageSize, type ImageSize } from "../ai/AiHelpers";

	const {
		imageData,
		size,
	}: {
		imageData: ImageData,
		size: ImageSize,
	} = $props();

	let canvas: HTMLCanvasElement | undefined = $state(undefined);
	let displaySize: ImageSize = $state({ width: 512, height: 512 });
	let process = $state(true);

	onMount(async () => {
		const canvasContext = canvas?.getContext("2d");

		if (!canvasContext) {
			return;
		}

		displaySize = applyImageSize(size, imageData);

		canvasContext.drawImage(
			await createImageBitmap(imageData),
			0,
			0,
			displaySize.width,
			displaySize.height,
		);

		process = false;
	});
</script>

<div>
	<canvas bind:this={canvas} width={displaySize.width} height={displaySize.height}></canvas>
</div>
