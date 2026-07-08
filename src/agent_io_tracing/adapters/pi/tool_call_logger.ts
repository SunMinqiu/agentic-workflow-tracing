import { appendFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";

type PendingCall = {
	startedAt: Date;
	toolName: string;
	args: unknown;
};

const pendingById = new Map<string, PendingCall>();

function pad2(value: number): string {
	return String(value).padStart(2, "0");
}

function pad6(value: number): string {
	return String(value).padStart(6, "0");
}

function formatTime(value: Date): string {
	const micros = value.getMilliseconds() * 1000;
	return `${pad2(value.getHours())}:${pad2(value.getMinutes())}:${pad2(value.getSeconds())}.${pad6(micros)}`;
}

function toPythonLiteral(value: unknown): string {
	if (value === null || value === undefined) return "None";
	if (typeof value === "boolean") return value ? "True" : "False";
	if (typeof value === "number") return Number.isFinite(value) ? String(value) : "None";
	if (typeof value === "string") {
		const escaped = value
			.replace(/\\/g, "\\\\")
			.replace(/'/g, "\\'")
			.replace(/\r/g, "\\r")
			.replace(/\n/g, "\\n")
			.replace(/\t/g, "\\t");
		return `'${escaped}'`;
	}
	if (Array.isArray(value)) return `[${value.map((item) => toPythonLiteral(item)).join(", ")}]`;

	if (typeof value === "object") {
		const entries = Object.entries(value as Record<string, unknown>);
		const serialized = entries
			.map(([key, item]) => `${toPythonLiteral(key)}: ${toPythonLiteral(item)}`)
			.join(", ");
		return `{${serialized}}`;
	}

	return toPythonLiteral(String(value));
}

function normalizeToolName(toolName: string): string {
	if (!toolName) return "Tool";
	if (toolName.toLowerCase() === "bash") return "Bash";
	return toolName[0].toUpperCase() + toolName.slice(1);
}

function formatLogLine(startedAt: Date, endedAt: Date, toolName: string, toolId: string, toolInput: unknown): string {
	const durationMs = endedAt.getTime() - startedAt.getTime();
	return `[${formatTime(startedAt)} -> ${formatTime(endedAt)}] (${durationMs.toFixed(1)}ms) ${toolName} (id=${toolId}) input=${toPythonLiteral(toolInput)}\n`;
}

function formatSystemPromptEntry(capturedAt: Date, prompt: string): string {
	return [
		`[${capturedAt.toISOString()}] length=${prompt.length}`,
		"--- SYSTEM PROMPT START ---",
		prompt,
		"--- SYSTEM PROMPT END ---",
		"",
	].join("\n");
}

export default function registerToolCallLogger(pi: ExtensionAPI): void {
	const logPath = process.env.PI_TOOL_LOG?.trim();
	if (!logPath) return;
	const systemPromptLogPath = `${logPath}.system_prompt`;

	mkdirSync(dirname(logPath), { recursive: true });
	mkdirSync(dirname(systemPromptLogPath), { recursive: true });

	pi.on("before_agent_start", async (_event, ctx) => {
		const prompt = ctx.getSystemPrompt();
		const entry = formatSystemPromptEntry(new Date(), prompt);
		appendFileSync(systemPromptLogPath, entry, { encoding: "utf-8" });
	});

	pi.on("tool_execution_start", async (event) => {
		pendingById.set(event.toolCallId, {
			startedAt: new Date(),
			toolName: normalizeToolName(event.toolName),
			args: event.args,
		});
	});

	pi.on("tool_execution_end", async (event) => {
		const endedAt = new Date();
		const pending = pendingById.get(event.toolCallId);
		pendingById.delete(event.toolCallId);

		const startedAt = pending?.startedAt ?? endedAt;
		const toolName = pending?.toolName ?? normalizeToolName(event.toolName);
		const toolInput = pending?.args ?? {};

		const line = formatLogLine(startedAt, endedAt, toolName, event.toolCallId, toolInput);
		appendFileSync(logPath, line, { encoding: "utf-8" });
	});

	pi.on("session_shutdown", async () => {
		const now = new Date();
		for (const [toolId, pending] of pendingById.entries()) {
			const line = formatLogLine(pending.startedAt, now, pending.toolName, toolId, pending.args);
			appendFileSync(logPath, line, { encoding: "utf-8" });
			pendingById.delete(toolId);
		}
	});
}
