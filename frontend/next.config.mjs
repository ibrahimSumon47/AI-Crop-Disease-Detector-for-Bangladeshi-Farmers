/**
 * Next.js config.
 * No API rewrites needed: the client calls the FastAPI backend directly
 * (CORS is configured on the backend via ALLOWED_ORIGINS).
 */
const nextConfig = {
  reactStrictMode: true,
};

export default nextConfig;
