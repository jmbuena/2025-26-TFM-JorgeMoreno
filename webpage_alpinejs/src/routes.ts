import { handler, HandlerSignature, PalAPI } from "./palAPI.ts";
import * as path from "@std/path";
import * as fs from "@std/fs";


export const router = new PalAPI({
	"GET": html("./static/html/test.html"),
	"/static/...path": {
		"GET": staticFile("./static/"),
	},
	"/models/:path": {
		"GET": staticFile("./static/models/", true),
	},
	"/ort/:path": {
		"GET": staticFile("./static/ort_wasm/"),
	},
	"/webcache.js": {
		"GET": serviceWorkerFile,
	},
});


const extensionMapping: Record<string, string> = {
	".mjs": "text/javascript",
	".wasm": "application/wasm",
	".onnx": "application/octet-stream",
	".js": "text/javascript",
	".css": "text/css",
};


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


async function createETag(data: ArrayBuffer): Promise<string> {
	const hashBuffer = await crypto.subtle.digest("SHA-256", data);
	const hashArray = [...new Uint8Array(hashBuffer)];
	return `"${hashArray.map(b => b.toString(16).padStart(2, "0")).join("")}"`;
}

async function serviceWorkerFile(request: Request) {
	const fileContents = await Deno.readFile("./static/js/webcache.js");

	const headers: Record<string, string> = {
		"Content-Type": "text/javascript",
	};

	return new Response(fileContents, {
		headers,
		status: 200,
	});
}


function staticFile(folder: string): HandlerSignature {
	return async function (request: Request, pathVariables: Record<string, string>): Promise<Response> {
		const filePath = pathVariables["path"];

		const completeFilePath = folder + filePath;

		console.log("Reading ", completeFilePath);

		if (!await fs.exists(completeFilePath)) {
			console.error("File does not exists");

			return new Response(null, {
				status: 404,
			});
		}

		const fileContents = await Deno.readFile(completeFilePath);

		const fileExtension = path.extname(filePath);
		const contentType = extensionMapping[fileExtension];

		const headers: Record<string, string> = {
			"Content-Type": contentType,
		};

		return new Response(fileContents, {
			headers,
			status: 200,
		});
	};
}
