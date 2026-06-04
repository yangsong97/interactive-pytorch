import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const lessons = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/lessons' }),
  schema: z.object({
    title: z.string(),
    module: z.string(),            // e.g. "M1 · Tensors"
    order: z.number(),             // global ordering for prev/next + sidebar
    summary: z.string(),
    colab: z.string().optional(),  // notebook filename under notebooks/
    draft: z.boolean().default(false),
  }),
});

export const collections = { lessons };
