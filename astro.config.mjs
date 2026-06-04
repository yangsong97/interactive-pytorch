// @ts-check
import { defineConfig } from 'astro/config';
import svelte from '@astrojs/svelte';
import mdx from '@astrojs/mdx';

// https://astro.build/config
export default defineConfig({
  site: 'https://yangsong97.github.io',
  base: '/interactive-pytorch',
  integrations: [svelte(), mdx()],
});
