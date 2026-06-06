import { HandlerSignature, PalAPI } from "./palAPI.ts";
import * as path from "@std/path";
import * as fs from "@std/fs";


export const router = new PalAPI({
	"GET": html("./static/html/test.html"),
	"/static/...path": {
		"GET": staticFile("./static/"),
	},
	"/models/:path": {
		"GET": staticFile("./static/models/"),
	},
	"/ort/:path": {
		"GET": staticFile("./static/ort_wasm/"),
	},
});


function html(htmlPath: string): HandlerSignature {
	return localFile(htmlPath, "text/html");
}


function localFile(path: string, contentType: string): HandlerSignature {
	return function (request: Request): Response {
		const fileContents = Deno.readTextFileSync(path);

		return new Response(fileContents, {
			headers: {
				"Content-Type": contentType,
			},
			status: 200,
		});
	};
}

const extensionMapping: Record<string, string> = {
	".mjs": "text/javascript",
	".wasm": "application/wasm",
	".onnx": "",
	".js": "text/javascript",
};

function staticFile(folder: string): HandlerSignature {
	return function (request: Request, pathVariables: Record<string, string>): Response {
		const filePath = pathVariables["path"];

		const completeFilePath = folder + filePath;

		console.log("Reading ", completeFilePath);

		if (!fs.existsSync(completeFilePath)) {
			console.error("File does not exists");

			return new Response(null, {
				status: 404,
			});
		}

		const fileContents = Deno.readFileSync(completeFilePath);

		const fileExtension = path.extname(filePath);
		const contentType = extensionMapping[fileExtension];

		return new Response(fileContents, {
			headers: {
				"Content-Type": contentType,
			},
			status: 200,
		});
	};
}
