import type { NextConfig } from "next"

const nextConfig: NextConfig = {
  output: "export",
  images: { unoptimized: true },
  turbopack: {},
  serverExternalPackages: ["onnxruntime-web"],
}

export default nextConfig
