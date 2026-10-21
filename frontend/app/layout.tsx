import type { Metadata, Viewport } from "next";
import { Noto_Sans_Bengali } from "next/font/google";
import "./globals.css";

/**
 * Bangla-first: the whole UI defaults to Bengali script. Noto Sans Bengali
 * renders cleanly at small sizes, which matters for farmers' phones.
 */
const bengali = Noto_Sans_Bengali({
  subsets: ["bengali", "latin"],
  variable: "--font-noto-bengali",
  display: "swap",
});

export const metadata: Metadata = {
  title: "ফসল রোগ নির্ণয় — AI Crop Disease Detector",
  description:
    "ছবি তুলে ধান, আলু বা টমেটোর পাতার রোগ জানুন — বাংলায় চিকিৎসা পরামর্শ সহ।",
  // /favicon.ico 404 fix + mobile icon: app/icon.png is auto-served by Next
  // (favicon + apple-touch); public/icon-*.png serve the PWA manifest.
  icons: { icon: "/icon-192.png", apple: "/icon-192.png" },
  manifest: "/manifest.webmanifest",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1, // mobile-friendly, no auto zoom
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="bn">
      <body className={`${bengali.variable} font-bangla antialiased`}>
        <header className="border-b border-leaf-100 bg-white">
          <div className="mx-auto flex max-w-3xl items-center justify-between px-4 py-3">
            <div className="flex items-center justify-between w-full">
              <span className="text-lg font-bold text-leaf-700">🌱 ফসল রোগ নির্ণয়</span>
              <a href="/history" className="text-sm text-stone-600 underline">📅 ইতিহাস</a>
            </div>
          </div>
        </header>
        <main className="mx-auto max-w-3xl px-4 py-6">{children}</main>
        <footer className="px-4 pb-6 pt-2 text-center text-sm text-stone-500">
          এটি একটি সহায়ক অ্যাপ — চিকিৎসা সিদ্ধান্তে কৃষি অফিসারের পরামর্শ নিন।
        </footer>
      </body>
    </html>
  );
}
