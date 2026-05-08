import type { ExtensionAPI, ExtensionContext } from "@mariozechner/pi-coding-agent";
import { Box, Text } from "@mariozechner/pi-tui";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";

const MESSAGE_TYPE = "burl-microscope";
const BASE_URL = process.env.BURL_MICROSCOPE_URL ?? "http://127.0.0.1:8765";
const DEFAULT_HARVEST = "harvest_batched_20260425_072910";

let currentSession: string | undefined;
let burlMode = false;
let serverProc: ChildProcessWithoutNullStreams | undefined;

export default function (pi: ExtensionAPI) {
	pi.registerMessageRenderer(MESSAGE_TYPE, (message, _options, theme) => {
		const box = new Box(1, 1, (t) => theme.bg("customMessageBg", t));
		box.addChild(new Text(`${theme.fg("success", "[Burl]")} ${message.content ?? ""}`, 0, 0));
		return box;
	});

	pi.registerCommand("burl", {
		description: "Burl microscope: /burl open|step|say|auto|tools|prompt|mode|recipes|health",
		handler: async (args, ctx) => {
			const tokens = splitArgs(args);
			const cmd = tokens.shift() ?? "help";
			try {
				await handleCommand(pi, ctx, cmd, tokens);
			} catch (error) {
				const msg = error instanceof Error ? error.message : String(error);
				emit(pi, `ERROR: ${msg}`);
				ctx.ui.notify(`Burl microscope error: ${msg}`, "error");
			}
		},
	});

	pi.on("input", async (event, ctx) => {
		if (!burlMode) return { action: "continue" as const };
		if (event.source === "extension") return { action: "continue" as const };
		if (event.text.trim().startsWith("/")) return { action: "continue" as const };
		if (!currentSession) {
			emit(pi, "No Burl session. Run `/burl open 1` first, or `/burl mode off`.");
			return { action: "handled" as const };
		}
		await ensureServer(ctx);
		const result = await postJson(`/api/sessions/${currentSession}/step`, { text: event.text });
		emit(pi, formatStep(result));
		return { action: "handled" as const };
	});

	pi.on("session_shutdown", async (event) => {
		// Reload tears down the extension runtime but should not tear down the
		// microscope server. The replacement extension will reconnect to the
		// already-running process, avoiding noisy Python resource_tracker warnings
		// from MLX/oracle shutdown on every /reload.
		if (event.reason === "reload") return;
		if (serverProc) {
			try {
				await postJson("/api/shutdown", {}, 2000);
			} catch {
				serverProc.kill();
			}
			serverProc = undefined;
		}
	});
}

async function handleCommand(
	pi: ExtensionAPI,
	ctx: ExtensionContext,
	cmd: string,
	tokens: string[],
) {
	switch (cmd) {
		case "help":
		case "":
			emit(pi, helpText());
			return;
		case "start":
			await ensureServer(ctx);
			emit(pi, `Microscope server is up at ${BASE_URL}`);
			return;
		case "health": {
			await ensureServer(ctx);
			const health = await getJson("/api/health");
			emit(pi, `health: ${JSON.stringify(health, null, 2)}`);
			return;
		}
		case "recipes": {
			await ensureServer(ctx);
			const data = await getJson("/api/recipes");
			emit(pi, `recipes: ${(data.recipes ?? []).join(", ")}`);
			return;
		}
		case "open": {
			await ensureServer(ctx);
			const spec = parseOpenArgs(tokens);
			const session = await postJson("/api/sessions", spec);
			currentSession = session.session_id;
			emit(pi, formatSession(session) + "\n\nBurl mode is on. Type normal messages to chat with Burl, or use `/burl step`.");
			burlMode = true;
			return;
		}
		case "show": {
			const sid = requireSession();
			await ensureServer(ctx);
			const session = await getJson(`/api/sessions/${sid}`);
			emit(pi, formatSession(session));
			return;
		}
		case "prompt": {
			const sid = requireSession();
			await ensureServer(ctx);
			const data = await getJson(`/api/sessions/${sid}/prompt`);
			emit(pi, `SYSTEM\n------\n${data.prompt.system}\n\nUSER\n----\n${data.prompt.user}`);
			return;
		}
		case "tools": {
			await ensureServer(ctx);
			if (tokens.length === 0) {
				const data = await getJson("/api/tools");
				emit(pi, formatTools(data.tools ?? []));
				return;
			}
			const sid = requireSession();
			const data = await postJson(`/api/sessions/${sid}/tools`, { names: tokens });
			emit(pi, `active tools now: ${(data.recipe.tools ?? []).join(", ")}`);
			return;
		}
		case "step": {
			const sid = requireSession();
			await ensureServer(ctx);
			const result = await postJson(`/api/sessions/${sid}/step`, { text: tokens.join(" ") });
			emit(pi, formatStep(result));
			return;
		}
		case "say": {
			const sid = requireSession();
			await ensureServer(ctx);
			const result = await postJson(`/api/sessions/${sid}/step`, { text: tokens.join(" ") });
			emit(pi, formatStep(result));
			return;
		}
		case "auto": {
			const sid = requireSession();
			await ensureServer(ctx);
			const result = await postJson(`/api/sessions/${sid}/auto`, {
				text: tokens.join(" "),
				max_steps: 8,
			});
			emit(pi, formatAuto(result));
			return;
		}
		case "mode": {
			const value = tokens[0] ?? (burlMode ? "off" : "on");
			burlMode = value !== "off";
			emit(pi, `Burl mode ${burlMode ? "on" : "off"}.`);
			return;
		}
		case "stop":
			try {
				await postJson("/api/shutdown", {}, 2000);
				serverProc = undefined;
				emit(pi, "Stopped Burl microscope server.");
			} catch (error) {
				if (serverProc) {
					serverProc.kill();
					serverProc = undefined;
					emit(pi, "Stopped spawned Burl microscope server.");
				} else {
					const msg = error instanceof Error ? error.message : String(error);
					emit(pi, `No spawned server to stop (${msg}).`);
				}
			}
			return;
		default:
			emit(pi, `Unknown /burl command ${JSON.stringify(cmd)}.\n\n${helpText()}`);
	}
}

function writeServerStderr(chunk: Buffer | string) {
	const text = String(chunk);
	for (const line of text.split(/\r?\n/)) {
		if (!line) continue;
		if (line.includes("resource_tracker: There appear to be") || line.includes("warnings.warn('resource_tracker")) {
			continue;
		}
		process.stderr.write(`[burl-microscope] ${line}\n`);
	}
}

function parseOpenArgs(tokens: string[]) {
	const spec: Record<string, unknown> = { harvest: DEFAULT_HARVEST, idx: 1, recipe: "baseline" };
	for (let i = 0; i < tokens.length; i++) {
		const t = tokens[i];
		if (t === "--harvest") spec.harvest = tokens[++i];
		else if (t === "--idx") {
			spec.idx = Number(tokens[++i]);
			delete spec.seed;
		} else if (t === "--seed") {
			spec.seed = Number(tokens[++i]);
			delete spec.idx;
		} else if (t === "--recipe") spec.recipe = tokens[++i];
		else if (/^\d+$/.test(t)) spec.idx = Number(t);
		else spec.recipe = t;
	}
	return spec;
}

async function ensureServer(ctx: ExtensionContext) {
	try {
		await getJson("/api/health", 1000);
		return;
	} catch {
		// fall through and spawn
	}
	if (!serverProc) {
		ctx.ui.notify("Starting Burl microscope server...", "info");
		serverProc = spawn(process.env.PYTHON ?? "python", ["-m", "burl.microscope.server"], {
			cwd: ctx.cwd,
			env: { ...process.env },
		});
		serverProc.stdout.on("data", (chunk) => process.stderr.write(`[burl-microscope] ${chunk}`));
		serverProc.stderr.on("data", writeServerStderr);
		serverProc.on("exit", () => {
			serverProc = undefined;
		});
	}
	const deadline = Date.now() + 30_000;
	while (Date.now() < deadline) {
		try {
			await getJson("/api/health", 1000);
			return;
		} catch {
			await new Promise((resolve) => setTimeout(resolve, 500));
		}
	}
	throw new Error(`Burl microscope server did not start at ${BASE_URL}`);
}

async function getJson(path: string, timeoutMs = 10_000) {
	return requestJson(path, { method: "GET" }, timeoutMs);
}

async function postJson(path: string, body: unknown, timeoutMs = 600_000) {
	return requestJson(path, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(body),
	}, timeoutMs);
}

async function requestJson(path: string, init: RequestInit, timeoutMs: number) {
	const controller = new AbortController();
	const timer = setTimeout(() => controller.abort(), timeoutMs);
	try {
		const response = await fetch(`${BASE_URL}${path}`, { ...init, signal: controller.signal });
		const text = await response.text();
		let data: any = undefined;
		try {
			data = text ? JSON.parse(text) : {};
		} catch {
			data = { text };
		}
		if (!response.ok) {
			throw new Error(data?.detail ? String(data.detail) : `HTTP ${response.status}: ${text}`);
		}
		return data;
	} finally {
		clearTimeout(timer);
	}
}

function requireSession() {
	if (!currentSession) throw new Error("No Burl session. Run `/burl open 1` first.");
	return currentSession;
}

function emit(pi: ExtensionAPI, content: string) {
	pi.sendMessage({ customType: MESSAGE_TYPE, content, display: true });
}

function formatSession(session: any) {
	const c = session.case ?? {};
	const r = session.recipe ?? {};
	return [
		`session: ${session.session_id}`,
		`case: ${c.harvest} ${c.lookup_key}=${c.lookup_value} bucket=${c.bucket}`,
		`refs: oracle=${c.oracle_play} burl=${c.burl_play} pi=${c.pi_play} qmean=${c.qmean_play}`,
		`tools: ${(r.tools ?? []).join(", ")}`,
		session.outcome ? `outcome: ${JSON.stringify(session.outcome)}` : "",
	].filter(Boolean).join("\n");
}

function formatStep(result: any) {
	const parts: string[] = [];
	if (result.assistant_text?.trim()) parts.push(`assistant:\n${result.assistant_text.trim()}`);
	if (result.tool_call) parts.push(`tool call: ${result.tool_call.name}(${JSON.stringify(result.tool_call.args)})`);
	if (result.tool_result) {
		const response = String(result.tool_result.response ?? "");
		parts.push(`tool result [${result.tool_result.name}]:\n${truncate(response, 2400)}`);
	}
	if (result.committed) parts.push(`COMMITTED: ${JSON.stringify(result.outcome, null, 2)}`);
	return parts.join("\n\n") || "No assistant text/tool call this step.";
}

function formatAuto(result: any) {
	const blocks = (result.results ?? []).map((r: any, i: number) => `# step ${i + 1}\n${formatStep(r)}`);
	if (result.committed) blocks.push(`FINAL OUTCOME:\n${JSON.stringify(result.outcome, null, 2)}`);
	return blocks.join("\n\n---\n\n");
}

function formatTools(tools: any[]) {
	return tools.map((tool) => `- ${tool.name} [${tool.protocol_role}] ${tool.protocol_phrase}`).join("\n");
}

function truncate(text: string, max: number) {
	return text.length <= max ? text : `${text.slice(0, max)}\n... <truncated ${text.length - max} chars>`;
}

function splitArgs(args: string) {
	// Good enough for microscope commands; prompt text after /burl say is kept
	// by joining tokens back with spaces.
	return args.trim() ? args.trim().split(/\s+/) : [];
}

function helpText() {
	return `Burl microscope commands:
/burl start                         start/check the local server
/burl open [idx] [--recipe name]     load a harvest decision, turn Burl mode on
/burl step [text]                    one model turn; dispatch at most one tool
/burl say <text>                     send a user message to Burl, then one turn
/burl auto [text]                    run turns until commit or pause
/burl tools                          list available tools
/burl tools name name ...            set active tools for current session
/burl prompt                         show rendered system + user prompt
/burl show                           show current case/outcome
/burl mode on|off                    route normal typed input to Burl
/burl recipes                        list recipes
/burl health                         show backend health

Recipe files live under burl/microscope/recipes/. Edit system.md, play.md,
tools.json, or tool_responses/<tool>.md, then /burl open the same case again.`;
}
