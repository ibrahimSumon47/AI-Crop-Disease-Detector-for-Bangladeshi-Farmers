/**
 * Tailwind CSS v4 PostCSS wiring — single plugin handles everything;
 * theme tokens live in app/globals.css via @theme.
 */
const config = {
  plugins: {
    "@tailwindcss/postcss": {},
  },
};

export default config;
