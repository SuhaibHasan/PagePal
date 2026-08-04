import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ProdSupportBuddy",
  description: "RAG-powered assistant for resolving production issues",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
