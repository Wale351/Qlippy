import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  allowedDevOrigins: ["192.168.0.102"],
  // Pin the workspace root to this app. Otherwise Next infers it from the nearest
  // lockfile and picked up a stray ~/package-lock.json, mis-resolving node_modules
  // (which left the native SWC binary unfound → slow WASM compile fallback).
  turbopack: { root: __dirname },
  // Turbopack's persistent dev cache (default-on since Next 16) grew to ~900MB
  // in .next/dev/cache/turbopack; restoring it on every `next dev` start held
  // 4-6.6GB RSS and pinned the CPU, OOMing the machine. Compile-on-demand is
  // cheaper than that restore ever was.
  experimental: { turbopackFileSystemCacheForDev: false },
  images: {
    remotePatterns: [
      { protocol: "https", hostname: "img.youtube.com" },
      { protocol: "https", hostname: "i.ytimg.com" },
      // Add your R2 public domain here when configured
      // { protocol: "https", hostname: "cdn.yourapp.com" },
    ],
  },
  async headers() {
    // Belt-and-braces with robots.ts: even pages a crawler reaches are
    // marked non-indexable at the header level.
    return [
      {
        source: "/:path*",
        headers: [{ key: "X-Robots-Tag", value: "noindex, nofollow, noarchive" }],
      },
    ];
  },
};

export default nextConfig;
