import { Suspense, lazy } from "react";
import { useTranslation } from "react-i18next";
import { HashRouter, Route, Routes } from "react-router-dom";

import { Footer } from "./components/Footer";
import { Skeleton } from "./components/Loading";
import { TopBar } from "./components/TopBar";
import { ThemeContext, useThemeState } from "./theme/useTheme";

// Route-level code splitting (§14).
const Overview = lazy(() => import("./pages/Overview"));
const Replays = lazy(() => import("./pages/Replays"));
const Results = lazy(() => import("./pages/Results"));
const Ablation = lazy(() => import("./pages/Ablation"));
const Robustness = lazy(() => import("./pages/Robustness"));
const Generalization = lazy(() => import("./pages/Generalization"));
const Benchmark = lazy(() => import("./pages/Benchmark"));
const Method = lazy(() => import("./pages/Method"));
const Reproducibility = lazy(() => import("./pages/Reproducibility"));
const About = lazy(() => import("./pages/About"));

function Shell() {
  const { t } = useTranslation();
  return (
    <div className="app-shell">
      <a href="#main" className="skip-link">
        {t("app.skipToContent")}
      </a>
      <TopBar />
      <main id="main" style={{ flex: 1, display: "flex", flexDirection: "column" }}>
        <Suspense
          fallback={
            <div className="page">
              <Skeleton height={400} />
            </div>
          }
        >
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/replays" element={<Replays />} />
            <Route path="/results" element={<Results />} />
            <Route path="/ablation" element={<Ablation />} />
            <Route path="/robustness" element={<Robustness />} />
            <Route path="/generalization" element={<Generalization />} />
            <Route path="/benchmark" element={<Benchmark />} />
            <Route path="/method" element={<Method />} />
            <Route path="/reproducibility" element={<Reproducibility />} />
            <Route path="/about" element={<About />} />
            <Route path="*" element={<Overview />} />
          </Routes>
        </Suspense>
      </main>
      <Footer />
    </div>
  );
}

export default function App() {
  const themeState = useThemeState();
  return (
    <ThemeContext.Provider value={themeState}>
      <HashRouter>
        <Shell />
      </HashRouter>
    </ThemeContext.Provider>
  );
}
