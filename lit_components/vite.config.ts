// vite.config.ts

import { defineConfig } from 'vite';

export default defineConfig({
	build: {
		lib: {
			entry: 'src/index.ts',
			formats: ['es'],
			fileName: 'face-alignment',
		},
		rollupOptions: {
			// external: ["onnxruntime-web", "onnxruntime-web/webgpu", "onnxruntime-web/wasm"],
		}
	}
	// build: {
	// 	outDir: "dist",
	// 	emptyOutDir: true,
	// 	rollupOptions: {
	// 		// external: ['@techstark/opencv-js'],

	// 		input: {
	// 			"index": "src/index.ts",
	// 			"core": "src/core.ts",
	// 			"haar": "src/haar.ts",
	// 			"opencv": "src/opencv.ts",
	// 			"helpers": "src/helpers.ts",
	// 			"default_haar": "src/default_haar.ts",
	// 		},

	// 		output: {
	// 			entryFileNames: "[name].js",
	// 			format: "es",
	// 			preserveModules: true,
	// 			preserveModulesRoot: "src",
	// 		},

	// 		preserveEntrySignatures: "strict",
	// 	}
	// }
});
