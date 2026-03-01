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
                github: 'https://github.com/scitrera/sparkrun',
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
                        {label: 'Installation', slug: 'getting-started/installation'},
                        {label: 'Quick Start', slug: 'getting-started/quick-start'},
                        {label: 'Networking', slug: 'getting-started/networking'},
                        {label: 'SSH Setup', slug: 'getting-started/ssh-setup'},
                        {label: 'Tips & Troubleshooting', slug: 'getting-started/troubleshooting'},
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
                        {label: 'tune', slug: 'cli/tune'},
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
                    label: 'Runtimes',
                    items: [
                        {label: 'Overview', slug: 'runtimes/overview'},
                        {label: 'vLLM', slug: 'runtimes/vllm'},
                        {label: 'SGLang', slug: 'runtimes/sglang'},
                        {label: 'llama.cpp', slug: 'runtimes/llama-cpp'},
                        {label: 'eugr-vllm', slug: 'runtimes/eugr-vllm'},
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
                    label: 'Claude Code Plugin',
                    items: [
                        {label: 'Overview', slug: 'claude-code-plugin/overview'},
                        {label: 'Commands', slug: 'claude-code-plugin/commands'},
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
