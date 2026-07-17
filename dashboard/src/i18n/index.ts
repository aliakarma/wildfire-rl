import i18n from "i18next";
import { initReactI18next } from "react-i18next";

import ar from "./locales/ar.json";
import en from "./locales/en.json";

export type Locale = "en" | "ar";

function detect(): Locale {
  // Shared URLs may carry the query inside the hash (#/x?lang=ar) or before it.
  const url = new URLSearchParams(
    `${window.location.hash.split("?")[1] ?? ""}&${window.location.search.slice(1)}`,
  );
  const fromUrl = url.get("lang");
  if (fromUrl === "ar" || fromUrl === "en") return fromUrl;
  const stored = localStorage.getItem("locale");
  if (stored === "ar" || stored === "en") return stored;
  return "en";
}

i18n.use(initReactI18next).init({
  resources: { en: { translation: en }, ar: { translation: ar } },
  lng: detect(),
  fallbackLng: "en",
  interpolation: { escapeValue: false },
});

export function setLocale(lng: Locale) {
  i18n.changeLanguage(lng);
  document.documentElement.lang = lng;
  document.documentElement.dir = lng === "ar" ? "rtl" : "ltr";
  localStorage.setItem("locale", lng);
}

// Apply direction for the detected locale on first load.
setLocale(i18n.language as Locale);

export default i18n;
