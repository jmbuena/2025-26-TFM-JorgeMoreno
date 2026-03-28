import { Landmark } from "./Landmarks";


const CSV_LANDMARKS_OFFSET_INDEX = 11;
const CSV_LANDMARKS_COUNT = 98;


export class AnnotationRow {
	constructor(
		readonly filename: string,
		readonly landmarks: Array<Landmark>,
	) {}
}


export function processCsvFile(fileContents: string): Map<string, AnnotationRow> {
	const lines: Array<string> = fileContents.split("\n");
	lines.splice(0, 1);

	const annotations: Map<string, AnnotationRow> = new Map();
	
	for (const line of lines) {
		const columns = line.split(";");

		const filename = columns[0].split("/").at(-1) ?? "unknown";

		const landmarks: Array<Landmark> = new Array(CSV_LANDMARKS_COUNT);

		for (let i = 0; i < CSV_LANDMARKS_COUNT; i++) {
			const landmarkIndex = CSV_LANDMARKS_OFFSET_INDEX + i * 2;

			const x = Number.parseFloat(columns[landmarkIndex]);
			const y = Number.parseFloat(columns[landmarkIndex + 1]);

			landmarks[i] = new Landmark(x, y);
		}

		annotations.set(filename, new AnnotationRow(
			filename,
			landmarks,
		));
	}

	return annotations;
}


export function objectToCsvString(dataRows: Array<Record<string, number | string>>): string {
	if (dataRows.length === 0) {
		return "";
	}

	const columnNames = Object.keys(dataRows[0]);

	const rowsString = dataRows.map((row) => {
		return columnNames.map((columnName) => {
			return row[columnName];
		}).join(", ");
	}).join("\n");

	const columnNamesString = columnNames.map((name) => {
		return name.replaceAll(" ", "_").toLowerCase();
	}).join(", ");

	return columnNamesString + "\n" + rowsString;
}
