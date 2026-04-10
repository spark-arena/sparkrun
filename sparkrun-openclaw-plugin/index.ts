// noinspection TypeScriptCheckImport

// @ts-ignore
import {definePluginEntry} from "openclaw/plugin-sdk/plugin-entry";
// @ts-ignore
import {Type} from "@sinclair/typebox";
import {execSync} from "node:child_process";

export default definePluginEntry({
    // @ts-ignore
    register(api) {
        api.logger.info("sparkrun plugin loaded");

        api.registerTool({
            name: "sparkrun_exec",
            description:
                "Execute a sparkrun CLI command. Use this tool for all sparkrun operations: " +
                "launching inference workloads, checking status, stopping jobs, browsing recipes, " +
                "benchmarking, cluster management, and more. The command should start with 'sparkrun'.",
            parameters: Type.Object({
                command: Type.String({
                    description:
                        "The full sparkrun CLI command to execute (e.g. 'sparkrun list', " +
                        "'sparkrun run qwen3-1.7b-vllm --tp 1 --no-follow', 'sparkrun cluster status')",
                }),
                timeout: Type.Optional(
                    Type.Number({
                        description: "Command timeout in milliseconds (default: 120000)",
                    }),
                ),
            }),
            // @ts-ignore
            execute: async ({command, timeout}) => {
                // Ensure the command starts with sparkrun for safety
                const trimmed = command.trim();
                if (!trimmed.startsWith("sparkrun")) {
                    return {
                        content: [
                            {
                                type: "text" as const,
                                text: "Error: Command must start with 'sparkrun'. Use this tool only for sparkrun CLI commands.",
                            },
                        ],
                    };
                }

                try {
                    const result = execSync(trimmed, {
                        encoding: "utf-8",
                        timeout: timeout ?? 120_000,
                        stdio: ["pipe", "pipe", "pipe"],
                    });

                    return {
                        content: [{type: "text" as const, text: result}],
                    };
                } catch (err: unknown) {
                    const error = err as {
                        stdout?: string;
                        stderr?: string;
                        status?: number;
                        message?: string;
                    };
                    const output = [error.stdout, error.stderr].filter(Boolean).join("\n");

                    return {
                        content: [
                            {
                                type: "text" as const,
                                text: output || error.message || "Command failed with no output",
                            },
                        ],
                    };
                }
            },
        });
    },
});
