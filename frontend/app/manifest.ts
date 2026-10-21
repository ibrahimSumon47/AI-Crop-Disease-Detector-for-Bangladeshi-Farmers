import type { MetadataRoute } from "next";

/**
 * PWA manifest (Next Metadata API) — makes the app installable on Android
 * (Add to Home Screen -> standalone app) and desktop browsers.
 * Icons live in frontend/public/ (icon-192.png, icon-512.png).
 */
export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "ফসল রোগ নির্ণয় — AI Crop Disease Detector",
    short_name: "ফসল রোগ",
    description:
      "ছবি তুলে ধান, আলু বা টমেটোর পাতার রোগ জানুন — বাংলায় চিকিৎসা পরামর্শ সহ।",
    start_url: "/",
    display: "standalone",
    background_color: "#f0fdf4",
    theme_color: "#16a34a",
    lang: "bn",
    icons: [
      { src: "/icon-192.png", sizes: "192x192", type: "image/png" },
      { src: "/icon-512.png", sizes: "512x512", type: "image/png" },
    ],
  };
}
