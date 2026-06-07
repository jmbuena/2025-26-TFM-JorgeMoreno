

export class Timings {
	protected readonly timings: Record<string, { start: number, end: number, duration: () => number }> = {};


	measure<T extends (...args: any) => any>(name: string, fn: T): ReturnType<T> {
		this.start(name);
		const result = fn();

		if (result instanceof Promise) {
			result.then(() => {
				this.end(name);
			});
		} else {
			this.end(name);
		}

		return result;
	}


	getTimings() {
		return this.timings;
	}


	protected start(name: string): void {
		this.timings[name] = {
			start: performance.now(),
			end: Number.NEGATIVE_INFINITY,
			duration: function(this) {
				return this.end - this.start;
			}
		};
	}


	protected end(name: string): void {
		this.timings[name].end = performance.now();
	}
}
