/** Join Astro's BASE_URL with a root-relative path, collapsing duplicate slashes. */
export function withBase(path: string, base = import.meta.env.BASE_URL): string {
  const b = base.replace(/\/$/, '');
  const p = path.startsWith('/') ? path : `/${path}`;
  return `${b}${p}`;
}
