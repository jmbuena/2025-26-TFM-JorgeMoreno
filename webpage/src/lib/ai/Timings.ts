

export class Timings {
	protected readonly timings: Record<string, { start: number, end: number }> = {};


	measure<T extends (...args: any) => any>(name: string, fn: T): ReturnType<T> {
		this.start(name);
		const result = fn();
		this.end(name);

		return result;
	}


	getTimings() {
		return this.timings;
	}


	protected start(name: string): void {
		this.timings[name] = {
			start: performance.now(),
			end: Number.NEGATIVE_INFINITY,
		};
	}


	protected end(name: string): void {
		this.timings[name].end = performance.now();
	}
}
