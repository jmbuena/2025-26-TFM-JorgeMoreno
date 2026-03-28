<script>
	import { onMount } from "svelte";
	import { prepareModels } from "./lib/ai/LoadModel";
	import OpenCV from "./lib/components/OpenCV.svelte";
	import "/src/assets/ort-wasm-simd-threaded.jsep.mjs";
    import CanvasBox from "./lib/components/CanvasBox.svelte";
    import TrainMultiple from "./lib/components/TrainMultiple.svelte";

	let loading = $state(true);
	let multiple = $state(location.pathname === "/multiple");

	onMount(async () => {
		await prepareModels();
		loading = false;
	});
</script>

<OpenCV />

<div class="min-h-[100vh] py-10 text-white">
	{#if loading}
		Loading...
	{:else if multiple}
		<TrainMultiple />
	{:else}
		<!-- <Canvas /> -->
		<CanvasBox />
	{/if}
</div>
