<script>
	import { onMount } from "svelte";
	import { prepareModels } from "./lib/ai/LoadModel";
	import Canvas from "./lib/components/Canvas.svelte";
	import OpenCV from "./lib/components/OpenCV.svelte";
	import "/src/assets/ort-wasm-simd-threaded.jsep.mjs";
    import CanvasBox from "./lib/components/CanvasBox.svelte";

	let loading = $state(true);

	onMount(async () => {
		await prepareModels();
		loading = false;
	});
</script>

<OpenCV />

<div class="min-h-[100vh] py-10 text-white">
	{#if loading}
		Loading...
	{:else}
		<!-- <Canvas /> -->
		<CanvasBox />
	{/if}
</div>
