import {defineConfig} from 'astro/config';
import starlight from '@astrojs/starlight';
import react from '@astrojs/react';

export default defineConfig({
    site: 'https://sparkrun.dev',
    server: {
        port: 4324,
        host: true, // Equivalent to --host flag
    },
    output: 'static',
    integrations: [
        starlight({
            title: 'sparkrun',
            logo: {
                src: './src/assets/sparkrun-logo.svg',
            },
            components: {
                Footer: './src/components/overrides/Footer.astro',
            },
            social: {
                github: 'https://github.com/spark-arena/sparkrun',
            },
            customCss: [
                '@fontsource/inter/400.css',
                '@fontsource/inter/500.css',
                '@fontsource/inter/600.css',
                '@fontsource/inter/700.css',
                '@fontsource/jetbrains-mono/400.css',
                '@fontsource/jetbrains-mono/500.css',
                './src/styles/custom.css',
                './src/styles/landing.css',
            ],
            sidebar: [
                {
                    label: 'Getting Started',
                    items: [
                        {label: 'Quick Start', slug: 'getting-started/quick-start'},
                        {label: 'Installation', slug: 'getting-started/installation'},
                        {label: 'Setup Wizard Walkthrough', slug: 'getting-started/setup-wizard'},
                    ],
                },
                {
                    label: 'Tutorials',
                    items: [
                        {label: 'Your First Model', slug: 'tutorials/first-model'},
                        {label: 'Multi-Node Tensor Parallelism', slug: 'tutorials/multi-node'},
                        {label: 'Benchmarking Models', slug: 'tutorials/benchmarking'},
                        {label: 'Proxy Gateway', slug: 'tutorials/proxy-gateway'},
                    ],
                },
                {
                    label: 'Recipes',
                    items: [
                        {label: 'Recipe Format', slug: 'recipes/format'},
                        {label: 'Writing Recipes', slug: 'recipes/writing-recipes'},
                        {label: 'GGUF Recipes', slug: 'recipes/gguf-recipes'},
                        {label: 'Registries', slug: 'recipes/registries'},
                    ],
                },
                {
                    label: 'Clusters',
                    items: [
                        {label: 'Creating Clusters', slug: 'clusters/creating'},
                        {label: 'Managing Clusters', slug: 'clusters/managing'},
                        {label: 'Cluster Status', slug: 'clusters/status'},
                    ],
                },
                {
                    label: 'Runtimes',
                    items: [
                        {label: 'Overview', slug: 'runtimes/overview'},
                        {label: 'vLLM', slug: 'runtimes/vllm'},
                        {label: 'SGLang', slug: 'runtimes/sglang'},
                        {label: 'llama.cpp', slug: 'runtimes/llama-cpp'},
                        {label: 'TensorRT-LLM', slug: 'runtimes/trtllm'},
                    ],
                },
                {
                    label: 'Advanced Topics',
                    items: [
                        {label: 'Networking', slug: 'getting-started/networking'},
                        {label: 'SSH Setup', slug: 'getting-started/ssh-setup'},
                        {label: 'Tips & Troubleshooting', slug: 'getting-started/troubleshooting'},
                        {label: 'Diagnostics', slug: 'getting-started/diagnostics'},
                    ],
                },
                {
                    label: 'Claude Code Plugin',
                    items: [
                        {label: 'Overview', slug: 'claude-code-plugin/overview'},
                        {label: 'Commands', slug: 'claude-code-plugin/commands'},
                    ],
                },
                {
                    label: 'CLI Reference',
                    items: [
                        {label: 'Overview', slug: 'cli/overview'},
                        {label: 'run', slug: 'cli/run'},
                        {label: 'stop', slug: 'cli/stop'},
                        {label: 'logs', slug: 'cli/logs'},
                        {label: 'status', slug: 'cli/status'},
                        {label: 'benchmark', slug: 'cli/benchmark'},
                        {label: 'Recipe Commands', slug: 'cli/recipe-commands'},
                        {label: 'Cluster Commands', slug: 'cli/cluster-commands'},
                        {label: 'Setup Commands', slug: 'cli/setup-commands'},
                        {label: 'export', slug: 'cli/export'},
                        {label: 'tune', slug: 'cli/tune'},
                        {label: 'proxy', slug: 'cli/proxy'},
                    ],
                },
                {
                    label: 'Developer Reference',
                    items: [
                        {label: 'CLI Automation', slug: 'developer-reference/cli-automation'},
                        {label: 'Execution Flow', slug: 'developer-reference/execution-flow'},
                        {label: 'Architecture', slug: 'developer-reference/architecture'},
                        {label: 'Contributing', slug: 'developer-reference/contributing'},
                    ],
                },
            ],
        }),
        react(),
    ],
    vite: {
        server: {
            allowedHosts: ["spark-2918", "localhost"],
        }
    }
});
