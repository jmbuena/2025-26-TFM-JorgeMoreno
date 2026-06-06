
/** List of HTTP methods for autocompletion  */
export type HttpMethod = "GET" | "HEAD" | "OPTIONS" | "TRACE" | "PUT" | "DELETE" | "POST" | "PATCH" | "CONNECT" | "ANY";


/** Signature for a handler function */
export type HandlerSignature = (
	context: Request,
	pathVariables: Record<string, string>
) => Response | Promise<Response>;


/** Data stored in a node of the Radix tree */
export type PathData = Partial<Record<HttpMethod, Route>>;


/** Specialization of the RadixTree for PalAPI */
export type PathRadixTree = RadixTree<PathData>;


/** Function signature of a Middleware */
export type MiddlewareSignature = (
	 request: Request,
	 queue: MiddlewareQueue,
	 pathVariables: Record<string, string>
) => Promise<Response>;


/** Route interface for a user-function and a list of middlewares */
export interface Route {
	handler: HandlerSignature,
	middlewares?: Array<MiddlewareSignature>,
}


/** Trick to make Typescript handle some autocompletion better */
export type Handlers = Partial<Record<HttpMethod, Route>>;


/** A definition of a valid path */
export type Path = `/${string}`;


/**
 * Group of routes, including the handlers
 * for each method, subroutes and middlewares
 */
export type Routes = 
	{ [K in keyof Handlers]?: Route | HandlerSignature }
	& { [path: Path]: Routes }
	& { middlewares?: Array<MiddlewareSignature> };


/** Utility function to create a handle with middlewares */
export function handler(
	handler: HandlerSignature,
	middlewares?: Array<MiddlewareSignature>
): Route {
	return {
		handler,
		middlewares,
	};
}


/** Common error type for palAPI errors */ 
export class PalAPIError extends Error {
	
}


/** Error thrown when a path and method are already used */
export class PathInUseError extends PalAPIError {
	constructor(
		readonly path: string,
		readonly method: HttpMethod,
	) {
		super("HTTP path already used with same HTTP Method", {
			cause: `Path [${path}] already in use for HTTP method [${method}]`,
		});
	}
}


/** Main class for palAPI, handles routing and initialization */
export class PalAPI {
	/** Path tree with all the available paths and methods */
	protected readonly tree: PathRadixTree = new RadixTree("");
	
	
	constructor(
		routes: Routes,
	) {
		this.searchRouteDeep(routes, [], this.tree, "");
	}
	
	
	/**
	 * Check a path exists in the route tree, and return it if it exists.
	 * If it does not exist, return undefined instead.
	 */
	look(path: string): PathRadixTree | undefined {
		if (path.endsWith("/")) {
			path = path.substring(path.length);
		}
		
		return this.tree.search(path, {});
	}
	
	
	/**
	 * Method that can be overriden to handle when
	 * the route/method does not exist.
	 */
	onNotFoundError = async (_request: Request): Promise<Response> => {
		return new Response("", {
			status: 404,
		});
	}
	
	
	/**
	 * Method that can be overriden to handle unhandled errors
	 * in the handler.
	 */
	onUnhandledError = async (_request: Request, _error: unknown): Promise<Response> => {
		return new Response("", {
			status: 500,
		});
	}
	
	
	/**
	 * Method available to be used in any standard HTTP server.
	 * 
	 * It processes a request, checks if the path is valid,
	 * and executes the middlewares and the handler.
	 */
	readonly fetch = (request: Request): Promise<Response> => {
		const path = new URL(request.url).pathname;
		
		const variables: Record<string, string> = {};
		const lastNode = this.tree.search(path, variables);

		if (!lastNode) {
			return this.onNotFoundError(request);
		}
		
		const data = lastNode["data"];
		if (!data) {
			return this.onNotFoundError(request);
		}

		let route = data[request.method as HttpMethod];
		if (!route) {
			// Check if it was declared as accepting ANY HTTP method
			if ("ANY" in data) {
				route = data["ANY"]!;
			} else {
				return this.onNotFoundError(request);
			}
		}
		
		// Remove the initial ":" or "..." from all path variable keys
		Object.keys(variables).forEach((key) => {
			const toRemove = (key.startsWith("...")) ? 3 : 1;
			delete Object.assign(variables, { [key.substring(toRemove)]: variables[key] })[key];
		});
		
		// Add route handler as last middleware (so it's executed automatically after the middleware queue)
		const pipeline: Array<MiddlewareSignature> = [
			...(route.middlewares ?? []),
			async (request, _queue, pathVariables) => {
				return route.handler(request, pathVariables);
			},
		];
		
		return new MiddlewareQueue(pipeline, request, variables)
			.next()
			.catch((error) => this.onUnhandledError(request, error));
	}
	
	
	protected searchRouteDeep(
		routes: Routes,
		middlewares: Array<MiddlewareSignature>,
		node: PathRadixTree,
		completePath: string,
	): void {
		const startMiddlewareCount = middlewares.length;
		
		if (routes.middlewares) {
			middlewares.push(...routes["middlewares"]);
		}
		
		const newNodeData: PathData = {};
		let newData = false;
		
		for (const key in routes) {
			if (key === "middlewares") {
				continue;
			}
			
			if (this.isPath(key)) {
				let lastNode = node;
				const pathParts = key.split("/");
				pathParts.shift();
				
				for (const pathPart of pathParts) {
					lastNode = lastNode.insert(pathPart);
				}
				
				this.searchRouteDeep(routes[key], middlewares, lastNode, completePath + key);
			} else {
				const method = key as HttpMethod;
				
				// Check the path and method are not already in use
				if (method in newNodeData || (node["data"] && method in node["data"])) {
					throw new PathInUseError(completePath, method);
				}
				
				if (typeof routes[method] === "function") {
					newNodeData[method] = {
						handler: routes[method],
						middlewares: middlewares.length > 0 ? [...middlewares] : undefined,
					};
				} else {
					const methodData = routes[method]!;
					const combinedMiddlewares = methodData.middlewares || middlewares.length > 0
						? [
							...middlewares,
							...(methodData.middlewares ?? []),
						]
						: undefined;
					
					methodData.middlewares = combinedMiddlewares;
					
					newNodeData[method] = methodData;
				}
				
				newData = true;
			}
		}
		
		// Remove the middlewares that were introduced inside this route path
		// but leave the old ones
		middlewares.splice(startMiddlewareCount - middlewares.length);
		
		if (newData) {
			if (node["data"]) {
				node["data"] = {
					...node["data"],
					...newNodeData,
				};
			} else {
				node["data"] = newNodeData;
			}
		}
	}
	
	
	protected isPath(value: string): value is Path {
		return value.startsWith("/");
	}
	
	
	protected createAlreadyInUseError(path: string, method: HttpMethod): never {
		throw new PalAPIError("HTTP path already used with same HTTP Method", {
			cause: `Path: ${path} :: Method ${method}`,
		});
	}
}


/** Custom generic RadixTree implementation */
export class RadixTree<T extends Record<string, unknown>> {
	/** User-data stored in this RadixTree node (if any) */
	protected data: T | undefined = undefined;
	
	protected children: Record<string, RadixTree<T>> = {};
	
	
	constructor(protected self: string)
		{ }
	
	
	/** Insert a new (single) child in this node */
	insert(path: string): RadixTree<T> {
		if (path in this.children) {
			return this.children[path];
		}
		
		const newNode = new RadixTree<T>(path);
		this.children[path] = newNode;
		
		return newNode;
	}
	
	
	/**
	 * Search the tree based on a path and returns the node found (if any).
	 *
	 * Also receives a variable object that will be filled with the path
	 * (declared with ":variableName" or "...variableName") variables
	 * found during a successful traversal of the tree.
	 */
	search(path: string, variables: Record<string, string>): RadixTree<T> | undefined {
		if (path.endsWith("/")) {
			path = path.substring(path.length);
		}
		
		return this.searchInternal(path.split("/"), 0, variables);
	}
	
	
	getData(): T | undefined {
		return this.data;
	} 
	
	
	/**
	 * Recursive implementation of a deep search inside the node (and its children)
	 */
	protected searchInternal(
		str: string[],
		index: number,
		variables: Record<string, string | undefined>
	): RadixTree<T> | undefined {
		const nextIndex = index + 1;
		
		if (this.self.startsWith("...")) {
			variables[this.self] = str.slice(index).join("/");
			
			return this;
		}
		
		if (this.self.startsWith(":")) {
			variables[this.self] = str[index];
			
			const node = this.checkChildren(str, nextIndex, variables);
			
			if (node) {
				return node;
			}
			
			variables[this.self] = undefined;
			
			return undefined;
		}
		
		if (str[index] === this.self) {
			const node = this.checkChildren(str, nextIndex, variables);
			
			if (node) {
				return node;
			}
		}
		
		return undefined;
	}
	
	
	/**
	 * Recursively checks the children of the node to search for a valid subtree.
	 */
	protected checkChildren(str: string[], nextIndex: number, variables: Record<string, string | undefined>): RadixTree<T> | undefined {
		const paths = Object.keys(this.children);
		
		if (str.length === nextIndex) {
			return this;
		}
		
		for (const path of paths) {
			const node = this.children[path].searchInternal(str, nextIndex, variables);
			if (node) {
				return node;
			}
		}
		
		return undefined;
	}
}


/**
 * Middleware queue to be executed before executing the
 * route handler.
 */
export class MiddlewareQueue {
	protected index: number = 0;
	
	
	constructor(
		protected readonly middlewares: MiddlewareSignature[],
		protected readonly request: Request,
		protected readonly variables: Record<string, string>,
	) { }
	
	
	next(): Promise<Response> {
		return this.middlewares[this.index++](this.request, this, this.variables);
	}
}
