import { useTranslation } from "react-i18next";

import { Callout } from "../components/Callout";
import { dataUrl } from "../lib/data";

/** The paper abstract, verbatim (EN) with a reviewed Arabic rendering. */
const ABSTRACT_EN = `Coordinating multi-agent teams to defend critical infrastructure during wildfires couples strategic asset prioritization with local suppression. We contribute (i) an infrastructure-aware wildfire-coordination benchmark built on the validated Cell2Fire physics engine and two documented, GIS-derived landscapes — Saudi Arabian petroleum infrastructure and California Wildland–Urban Interface (WUI) zones — under a suppression-relevant regime with a uniform 5-seed × 15-episode protocol; and (ii) HierComm, a value-aware hierarchical multi-agent policy that composes a rule-based value-aware asset assignment with a learned tactical policy trained by imitation of a value-aware expert and reinforcement-learning fine-tuning. Because the strongest baseline differs by region, consistent performance across both landscapes is non-trivial. HierComm attains the best average Weighted Economic Loss and Infrastructure Survival Rate in both regions; these improvements are statistically significant against the communicating baseline and the heuristic teacher and are largest on the scattered-asset (California) landscape, while on the concentrated-asset (Saudi) landscape the margin over the strongest flat baseline is within seed variance. A component ablation attributes the gains to the value-aware hierarchy, the learned tactical policy, and RL fine-tuning, and reports that explicit inter-agent communication is not a statistically significant driver at the three-agent scale studied. HierComm is also robust — strongest across an easy/medium/hard difficulty sweep and retaining a protective margin over inaction under held-out ignitions, rotated wind and asset layouts, and, with degradation, cross-region transfer. We make no claim of novelty for the underlying reinforcement-learning or communication primitives; our contributions are the benchmark, a simple and strong integrated system, an honest component analysis, and a fully reproducible release (code, checkpoints, GIS pipeline, hashed manifest).`;

const ABSTRACT_AR = `يجمع تنسيقُ فرقٍ متعددة الوكلاء للدفاع عن البنية التحتية الحرجة أثناء حرائق الغابات بين ترتيب أولويات الأصول استراتيجيًا والإخماد الموضعي. نقدّم (1) منصّة قياس لتنسيق مكافحة حرائق الغابات واعية بالبنية التحتية، مبنية على محرك فيزياء Cell2Fire الموثّق ومنظرين طبيعيين موثّقين مشتقين من نظم المعلومات الجغرافية — البنية التحتية النفطية في المملكة العربية السعودية ومناطق التماس العمراني في كاليفورنيا — ضمن نظام ملائم لمهام الإخماد وبروتوكول موحّد (5 بذور × 15 حلقة)؛ و(2) طريقة HierComm، وهي سياسة هرمية واعية بالقيمة متعددة الوكلاء تجمع بين إسناد أصول قائم على قاعدة واعية بالقيمة وسياسة تكتيكية متعلَّمة دُرّبت بمحاكاة خبير واعٍ بالقيمة ثم بالضبط الدقيق بالتعلم المعزَّز. ولأن أقوى سياسة مرجعية تختلف باختلاف المنطقة، فإن الأداء المتسق عبر كلا المنظرين ليس أمرًا بديهيًا. تحقق HierComm أفضل متوسط للخسارة الاقتصادية الموزونة ولمعدل بقاء البنية التحتية في كلتا المنطقتين؛ وهذه التحسينات ذات دلالة إحصائية مقابل السياسة المتواصلة والمعلّم الاسترشادي، وهي الأكبر في منظر الأصول المتناثرة (كاليفورنيا)، بينما يبقى الفارق عن أقوى سياسة مسطّحة في منظر الأصول المركّزة (السعودية) ضمن حدود تباين البذور. ويُرجع تحليلُ إزالة المكوّنات المكاسبَ إلى الهرمية الواعية بالقيمة والسياسة التكتيكية المتعلَّمة والضبط الدقيق بالتعلم المعزَّز، ويُبيّن أن التواصل الصريح بين الوكلاء ليس عاملًا ذا دلالة إحصائية عند نطاق الوكلاء الثلاثة المدروس. كما أن HierComm متينة — فهي الأقوى عبر مسح صعوبة بثلاثة مستويات، وتحافظ على هامش حماية مقارنةً بعدم التدخل تحت مواقع اشتعال محجوبة ورياح وأصول مُدارة، ومع تدهورٍ عند النقل بين المنطقتين. ولا ندّعي جِدّةً في مكوّنات التعلم المعزَّز أو التواصل الأساسية؛ فإسهاماتنا هي المنصّة، ونظام متكامل بسيط وقوي، وتحليل صريح للمكوّنات، وإصدار قابل لإعادة الإنتاج بالكامل (شيفرة ونقاط حفظ وخط معالجة جغرافي وقائمة موثّقة بالبصمات).`;

const BIBTEX = `@misc{anonymous2027wildfire,
  title  = {Infrastructure-Aware Wildfire Coordination: A Physics-Grounded
            Benchmark and a Value-Aware Hierarchical Multi-Agent
            Reinforcement Learning Method},
  author = {Anonymous},
  note   = {Under double-blind review},
  year   = {2026}
}`;

export default function About() {
  const { t, i18n } = useTranslation();
  const ar = i18n.language === "ar";

  return (
    <div className="page">
      <h1>{t("about.title")}</h1>
      <p className="page-summary">{t("about.summary")}</p>

      <Callout>{t("about.anonymity")}</Callout>

      <section className="section" style={{ marginBlockStart: "var(--sp-8)" }}>
        <h2>{t("about.abstract")}</h2>
        <div className="card" style={{ maxWidth: "88ch" }}>
          <p
            lang={ar ? "ar" : "en"}
            dir={ar ? "rtl" : "ltr"}
            style={{ margin: 0, fontSize: 15, color: "var(--text-secondary)" }}
          >
            {ar ? ABSTRACT_AR : ABSTRACT_EN}
          </p>
        </div>
        <p className="card-footnote">
          <a href={dataUrl("media/paper/paper.pdf")}>{t("actions.readPaper")} (PDF)</a>
        </p>
      </section>

      <section className="section">
        <h2>{t("about.citation")}</h2>
        <p className="page-summary">{t("about.citationNote")}</p>
        <div className="code-block" style={{ maxWidth: "88ch" }}>
          <pre>
            <code>{BIBTEX}</code>
          </pre>
        </div>
      </section>

      <section className="section">
        <h2>{t("about.license")}</h2>
        <p className="page-summary">{t("about.licenseBody")}</p>
      </section>
    </div>
  );
}
