# Arabic copy review deck — Wildfire-MARL dashboard

*Generated 2026-07-17 from `dashboard/src/i18n/locales/*.json` by
`scripts/make_ar_review_deck.py`. Regenerate after any copy change.*

## Instructions for the reviewer

You are reviewing the **Arabic locale of a public dashboard accompanying a peer-reviewed
AAAI submission**. Please check every row for:

1. **Faithfulness** — the Arabic must not make a claim stronger or weaker than the
   English (e.g., "best on average, statistically tied on Saudi" must survive exactly).
2. **Terminology** — statistical and ML terms should read naturally to an Arabic-speaking
   ML researcher; method names (HierComm, CommNet, MAPPO) stay in Latin script.
3. **Register** — Modern Standard Arabic, scientific tone, no colloquialisms.
4. **Placeholders** — `{{x}}` markers and `<bdi>` tags must stay exactly where they are
   (they inject live numbers and protect them from right-to-left reordering).

Mark corrections directly in this file (or in `ar.json`), then complete the sign-off
block at the end.


## `ablation.*`

| Key | English | Arabic | OK? |
|---|---|---|---|
| `ablation.contribution` | Component contributions — WEL vs Full HierComm | <div dir="rtl">إسهامات المكوّنات — WEL مقارنةً بالطريقة الكاملة</div> |  |
| `ablation.contributionSubtitle` | Bars show each ablated variant's WEL relative to the full method; sorted by effect | <div dir="rtl">تُظهر الأعمدة WEL لكل متغيّر منزوع المكوّن نسبةً إلى الطريقة الكاملة؛ مرتّبة حسب حجم الأثر</div> |  |
| `ablation.deterministicNote` | deterministic — magnitude only, no seed-variance test (n = 1) | <div dir="rtl">حتمي — المقدار فقط، دون اختبار تباين بين البذور (n = 1)</div> |  |
| `ablation.negativeBody` | Removing explicit communication produces no statistically significant change (Saudi <bdi>{{pSaudi}}</bdi>, California <bdi>{{pCalifornia}}</bdi>). We report this plainly rather than claiming communication as a driver of the results. | <div dir="rtl">إزالة التواصل الصريح لا تُحدث تغييرًا ذا دلالة إحصائية (السعودية <bdi>{{pSaudi}}</bdi>، كاليفورنيا <bdi>{{pCalifornia}}</bdi>). نُقرّ بذلك بوضوح بدلًا من ادّعاء أن التواصل عامل محرّك للنتائج.</div> |  |
| `ablation.negativeTitle` | The communication negative result | <div dir="rtl">النتيجة السلبية للتواصل</div> |  |
| `ablation.summary` | Single-factor component ablation of HierComm: each variant removes one part. Removing the hierarchy, the learned tactical policy, or RL fine-tuning significantly hurts; removing communication or shaping does not — the paper reports that negative result plainly. | <div dir="rtl">تحليل أحادي العامل لمكوّنات HierComm: كل متغيّر يزيل جزءًا واحدًا. إزالة الهرمية أو السياسة التكتيكية المتعلَّمة أو الضبط الدقيق بالتعلم المعزَّز تضر بالنتائج بدلالة إحصائية؛ أما إزالة التواصل أو تشكيل المكافأة فلا — وتُقرّ الورقة بهذه النتيجة السلبية بوضوح.</div> |  |
| `ablation.tableTitle` | Full ablation table | <div dir="rtl">جدول إزالة المكوّنات الكامل</div> |  |
| `ablation.title` | Ablation Analysis | <div dir="rtl">تحليل إزالة المكوّنات</div> |  |
| `ablation.variantCol` | Variant | <div dir="rtl">المتغيّر</div> |  |

## `about.*`

| Key | English | Arabic | OK? |
|---|---|---|---|
| `about.abstract` | Abstract | <div dir="rtl">الملخّص</div> |  |
| `about.anonymity` | This site is anonymized for AAAI double-blind review: no author names, institutions, or repository links appear anywhere. | <div dir="rtl">هذا الموقع مجهول الهوية لمراجعة AAAI المزدوجة التعمية: لا تظهر فيه أسماء مؤلفين ولا مؤسسات ولا روابط مستودعات.</div> |  |
| `about.citation` | Citation | <div dir="rtl">الاستشهاد</div> |  |
| `about.citationNote` | Anonymized during double-blind review; the full citation is added upon acceptance. | <div dir="rtl">مجهول الهوية أثناء المراجعة المزدوجة التعمية؛ يُضاف الاستشهاد الكامل عند القبول.</div> |  |
| `about.license` | License | <div dir="rtl">الترخيص</div> |  |
| `about.licenseBody` | Repository code: MIT. Cell2Fire (fire physics): GPL-3.0. Data sources are credited in the data card. | <div dir="rtl">شيفرة المستودع: MIT. ‏Cell2Fire (فيزياء الحرائق): GPL-3.0. مصادر البيانات موثّقة في بطاقة البيانات.</div> |  |
| `about.summary` | The paper, in brief. | <div dir="rtl">الورقة باختصار.</div> |  |
| `about.title` | About & Paper | <div dir="rtl">حول المشروع والورقة البحثية</div> |  |

## `actions.*`

| Key | English | Arabic | OK? |
|---|---|---|---|
| `actions.copied` | Copied | <div dir="rtl">تم النسخ</div> |  |
| `actions.copy` | Copy | <div dir="rtl">نسخ</div> |  |
| `actions.downloadCsv` | Download CSV | <div dir="rtl">تنزيل CSV</div> |  |
| `actions.downloadPng` | Download chart (PNG) | <div dir="rtl">تنزيل الرسم (PNG)</div> |  |
| `actions.exploreResults` | Explore results | <div dir="rtl">استكشاف النتائج</div> |  |
| `actions.play` | Play | <div dir="rtl">تشغيل</div> |  |
| `actions.readPaper` | Read the paper | <div dir="rtl">قراءة الورقة البحثية</div> |  |
| `actions.retry` | Retry | <div dir="rtl">إعادة المحاولة</div> |  |
| `actions.viewChart` | View as chart | <div dir="rtl">عرض كرسم بياني</div> |  |
| `actions.viewTable` | View as table | <div dir="rtl">عرض كجدول</div> |  |
| `actions.watchReplays` | Watch replays | <div dir="rtl">مشاهدة إعادة العرض</div> |  |

## `app.*`

| Key | English | Arabic | OK? |
|---|---|---|---|
| `app.skipToContent` | Skip to content | <div dir="rtl">الانتقال إلى المحتوى</div> |  |
| `app.tagline` | Infrastructure-aware wildfire coordination — a physics-grounded MARL benchmark and HierComm, a value-aware hierarchical method. | <div dir="rtl">تنسيق مكافحة حرائق الغابات مع مراعاة البنية التحتية — منصّة قياس متعددة الوكلاء قائمة على فيزياء محاكاة موثّقة، وطريقة HierComm الهرمية الواعية بالقيمة.</div> |  |
| `app.title` | Wildfire-MARL Benchmark | <div dir="rtl">منصّة قياس التنسيق متعدد الوكلاء لحرائق الغابات</div> |  |

## `benchmark.*`

| Key | English | Arabic | OK? |
|---|---|---|---|
| `benchmark.boundary` | Cell2Fire vs wrapper boundary | <div dir="rtl">الحد الفاصل بين Cell2Fire والمغلّف البرمجي</div> |  |
| `benchmark.boundaryPhysics` | Validated physics (never modified): fire growth, spread rates, fuel consumption — Cell2Fire C++. | <div dir="rtl">فيزياء موثّقة (لا تُعدَّل أبدًا): نمو الحريق ومعدلات الانتشار واستهلاك الوقود — Cell2Fire بلغة C++‎.</div> |  |
| `benchmark.boundaryWrapper` | Wrapper (this benchmark): suppression treatment, asset cascade, rewards, observations, PettingZoo multi-agent layer. | <div dir="rtl">المغلّف البرمجي (هذه المنصّة): معالجة الإخماد، وتتابع الأصول، والمكافآت، والمشاهدات، وطبقة PettingZoo متعددة الوكلاء.</div> |  |
| `benchmark.californiaDesc` | Forest-WUI regime: scattered critical facilities across the wildland–urban interface. If no one acts, WEL = {{wel}}. | <div dir="rtl">نظام غابات-تماس عمراني: منشآت حرجة متناثرة على واجهة التماس بين البرّية والعمران. دون أي تدخل، WEL = ‏{{wel}}.</div> |  |
| `benchmark.glossary` | Metrics glossary | <div dir="rtl">مسرد المقاييس</div> |  |
| `benchmark.pipeline` | GIS pipeline | <div dir="rtl">خط معالجة نظم المعلومات الجغرافية</div> |  |
| `benchmark.pipelineSteps.crit` | Stylized Gaussian criticality (declared as stylized) | <div dir="rtl">أهمية حرجة غاوسية مبسّطة (مصرَّح بأنها مبسّطة)</div> |  |
| `benchmark.pipelineSteps.era5` | ERA5 → hourly weather + CFFDRS FWI | <div dir="rtl">‏ERA5 ← طقس ساعي + مؤشر FWI الكندي</div> |  |
| `benchmark.pipelineSteps.firms` | FIRMS → ignition sampling | <div dir="rtl">‏FIRMS ← عيّنات مواقع الاشتعال</div> |  |
| `benchmark.pipelineSteps.ndvi` | MODIS NDVI → FBP fuel model | <div dir="rtl">‏MODIS NDVI ← نموذج وقود FBP</div> |  |
| `benchmark.pipelineSteps.srtm` | SRTM → slope | <div dir="rtl">‏SRTM ← الانحدار</div> |  |
| `benchmark.regime` | Suppression-relevant regime | <div dir="rtl">النظام الملائم لمهام الإخماد</div> |  |
| `benchmark.regimeCellNote` | Cells show Saudi / California where the two regions differ | <div dir="rtl">تعرض الخلايا قيمة السعودية / كاليفورنيا حيثما تختلف المنطقتان</div> |  |
| `benchmark.regimeIntro` | The stock Cell2Fire setup is unsuppressible for a 3-agent team; the benchmark derives a regime in which suppression is possible but not trivial: scaled wind, fixed fire-weather (FFMC), one threat-biased upwind ignition, a crew-scale treatment footprint (radius 1), and 30 simulated minutes per agent step. | <div dir="rtl">إعداد Cell2Fire الأصلي غير قابل للإخماد بفريق من ثلاثة وكلاء؛ لذلك تشتق المنصّة نظامًا يكون فيه الإخماد ممكنًا دون أن يكون سهلًا: رياح مُحجَّمة، ومؤشر طقس حرائق ثابت (FFMC)، واشتعال واحد مع الريح منحاز نحو التهديد، وبصمة معالجة بحجم طاقم (نصف قطر 1)، و30 دقيقة محاكاة لكل خطوة وكيل.</div> |  |
| `benchmark.regimeParam` | Parameter | <div dir="rtl">المعامل</div> |  |
| `benchmark.regionCards` | The two regions | <div dir="rtl">المنطقتان</div> |  |
| `benchmark.saudiDesc` | Desert-petroleum regime: concentrated, high-value petroleum infrastructure. If no one acts, WEL = {{wel}}. | <div dir="rtl">نظام صحراوي-نفطي: بنية تحتية نفطية مركّزة عالية القيمة. دون أي تدخل، WEL = ‏{{wel}}.</div> |  |
| `benchmark.summary` | Two GIS-derived 32×32 landscapes with real, provenance-documented critical assets, run on unmodified Cell2Fire fire physics under a suppression-relevant regime. | <div dir="rtl">منظران طبيعيان 32×32 مشتقان من نظم المعلومات الجغرافية مع أصول حرجة حقيقية موثّقة المصدر، يعملان على فيزياء Cell2Fire دون أي تعديل ضمن نظام ملائم لمهام الإخماد.</div> |  |
| `benchmark.title` | Benchmark & Data | <div dir="rtl">المعيار والبيانات</div> |  |

## `common.*`

| Key | English | Arabic | OK? |
|---|---|---|---|
| `common.higherBetter` | higher is better | <div dir="rtl">الأعلى أفضل</div> |  |
| `common.loadError` | Could not load this panel's data. | <div dir="rtl">تعذّر تحميل بيانات هذه اللوحة.</div> |  |
| `common.loading` | Loading… | <div dir="rtl">جارٍ التحميل…</div> |  |
| `common.lowerBetter` | lower is better | <div dir="rtl">الأقل أفضل</div> |  |
| `common.noData` | No data for this filter. | <div dir="rtl">لا توجد بيانات لهذا المرشّح.</div> |  |
| `common.ours` | ours | <div dir="rtl">طريقتنا</div> |  |
| `common.protocolNote` | 5 training seeds × 15 held-out episodes; unit of analysis = per-training-seed mean (n = 5) | <div dir="rtl">5 بذور تدريب × 15 حلقة تقييم محجوبة؛ وحدة التحليل = متوسط كل بذرة تدريب (n = 5)</div> |  |
| `common.sourceArtifact` | Source: {{path}} | <div dir="rtl">المصدر: {{path}}</div> |  |
| `common.vsNoop` | vs No-Op {{value}} | <div dir="rtl">مقابل عدم التدخل {{value}}</div> |  |

## `condition.*`

| Key | English | Arabic | OK? |
|---|---|---|---|
| `condition.assets_rot90` | Rotated asset layout | <div dir="rtl">تخطيط أصول مُدار بمقدار 90°</div> |  |
| `condition.cross_region` | Cross-region transfer | <div dir="rtl">النقل بين المنطقتين</div> |  |
| `condition.heldout_ignition` | Held-out ignitions | <div dir="rtl">مواقع اشتعال محجوبة عن التدريب</div> |  |
| `condition.standard` | Standard | <div dir="rtl">الوضع القياسي</div> |  |
| `condition.wind_plus90` | Wind rotated +90° | <div dir="rtl">رياح مُدارة بمقدار 90°</div> |  |

## `difficulty.*`

| Key | English | Arabic | OK? |
|---|---|---|---|
| `difficulty.easy` | Easy | <div dir="rtl">سهل</div> |  |
| `difficulty.hard` | Hard | <div dir="rtl">صعب</div> |  |
| `difficulty.medium` | Medium | <div dir="rtl">متوسط</div> |  |

## `filters.*`

| Key | English | Arabic | OK? |
|---|---|---|---|
| `filters.condition` | Condition | <div dir="rtl">الحالة</div> |  |
| `filters.difficulty` | Difficulty | <div dir="rtl">مستوى الصعوبة</div> |  |
| `filters.kind` | Type | <div dir="rtl">النوع</div> |  |
| `filters.metric` | Metric | <div dir="rtl">المقياس</div> |  |
| `filters.reset` | Reset filters | <div dir="rtl">إعادة ضبط المرشّحات</div> |  |
| `filters.seed` | Training seed | <div dir="rtl">بذرة التدريب العشوائية</div> |  |

## `footer.*`

| Key | English | Arabic | OK? |
|---|---|---|---|
| `footer.built` | built | <div dir="rtl">تاريخ البناء</div> |  |
| `footer.commit` | commit | <div dir="rtl">الإيداع</div> |  |
| `footer.downloads` | Data downloads | <div dir="rtl">تنزيل البيانات</div> |  |
| `footer.fingerprint` | Freeze fingerprint | <div dir="rtl">بصمة التجميد</div> |  |
| `footer.license` | Code: MIT · Cell2Fire: GPL-3.0 | <div dir="rtl">الشيفرة: MIT · ‏Cell2Fire: ‏GPL-3.0</div> |  |
| `footer.provenance` | All values are generated from frozen artifacts | <div dir="rtl">جميع القيم مولَّدة من نتائج مجمَّدة</div> |  |
| `footer.verified` | SHA-256 verified | <div dir="rtl">تم التحقق عبر SHA-256</div> |  |

## `generalization.*`

| Key | English | Arabic | OK? |
|---|---|---|---|
| `generalization.asymmetry` | Asymmetric collapse: CommNet transfers well in one direction (TRS {{good}}) and collapses in the other (TRS {{bad}}). HierComm degrades under transfer but never collapses. | <div dir="rtl">انهيار غير متماثل: تنتقل CommNet جيدًا في اتجاه واحد (TRS ‏{{good}}) وتنهار في الاتجاه الآخر (TRS ‏{{bad}}). أما HierComm فتتدهور مع النقل لكنها لا تنهار أبدًا.</div> |  |
| `generalization.deltaSubtitle` | Higher = more loss prevented within that condition; zero = no added value | <div dir="rtl">الأعلى = خسارة أكبر جرى تفاديها ضمن الحالة؛ الصفر = لا قيمة مضافة</div> |  |
| `generalization.deltaTitle` | ΔWEL vs No-Op by condition | <div dir="rtl">‏ΔWEL مقابل عدم التدخل حسب الحالة</div> |  |
| `generalization.matrixSubtitle` | Rows = trained on; columns = evaluated on. Every cell is annotated — color is never the only encoding. | <div dir="rtl">الصفوف = منطقة التدريب؛ الأعمدة = منطقة التقييم. كل خلية موسومة بقيمتها — اللون ليس الترميز الوحيد أبدًا.</div> |  |
| `generalization.summary` | Held-out ignitions, rotated wind, rotated assets, and cross-region transfer. ΔWEL vs No-Op is the falsifiability axis: a policy adding no value sits at zero. The transfer matrix shows where learned coordination survives a region swap — and where it collapses. | <div dir="rtl">مواقع اشتعال محجوبة، ورياح مُدارة، وأصول مُدارة، ونقل بين المنطقتين. ‏ΔWEL مقابل عدم التدخل هو محور قابلية الدحض: السياسة التي لا تضيف قيمة تقف عند الصفر. وتُظهر مصفوفة النقل أين يصمد التنسيق المتعلَّم بعد تبديل المنطقة — وأين ينهار.</div> |  |
| `generalization.title` | Generalization & Transfer | <div dir="rtl">التعميم والنقل</div> |  |
| `generalization.transferReplays` | Watch the native-vs-transferred rollouts on the Replays page. | <div dir="rtl">شاهد جولات «أصلي مقابل منقول» في صفحة إعادة العرض.</div> |  |

## `lang.*`

| Key | English | Arabic | OK? |
|---|---|---|---|
| `lang.label` | Switch language | <div dir="rtl">تغيير اللغة</div> |  |
| `lang.switch` | العربية | <div dir="rtl">English</div> |  |

## `method.*`

| Key | English | Arabic | OK? |
|---|---|---|---|
| `method.archCommander` | Strategic commander (value-aware rule) | <div dir="rtl">القائد الاستراتيجي (قاعدة واعية بالقيمة)</div> |  |
| `method.archCommanderDesc` | Every {{n}} steps, scores each asset by threat τ = v/(1+d) — value over distance-to-fire — and assigns agents to the top assets. | <div dir="rtl">كل {{n}} خطوات، يُقيَّم كل أصل بدرجة تهديد τ = v/(1+d) — القيمة على المسافة إلى الحريق — وتُسند الأصول الأعلى تهديدًا إلى الوكلاء.</div> |  |
| `method.archEnv` | Cell2Fire environment | <div dir="rtl">بيئة Cell2Fire</div> |  |
| `method.archEnvDesc` | Validated fire physics; the wrapper applies treatments and computes WEL/ISR rewards. | <div dir="rtl">فيزياء حرائق موثّقة؛ يطبّق المغلّف المعالجات ويحسب مكافآت WEL/ISR.</div> |  |
| `method.archTactical` | Communicating tactical policy (learned) | <div dir="rtl">السياسة التكتيكية المتواصلة (متعلَّمة)</div> |  |
| `method.archTacticalDesc` | CNN over a 9×9 egocentric crop + target encoding; {{r}} communication rounds; outputs move/treat actions. | <div dir="rtl">شبكة CNN على مقطع ذاتي المركز 9×9 مع ترميز الهدف؛ {{r}} جولتا تواصل؛ وتُخرج أفعال الحركة/المعالجة.</div> |  |
| `method.architecture` | Architecture | <div dir="rtl">البنية</div> |  |
| `method.hyperparams` | Hyperparameters | <div dir="rtl">المعاملات الفائقة</div> |  |
| `method.scope` | No algorithmic-novelty claim: the primitives (CNN policies, communication rounds, PPO, BC) are standard. A fully learned commander variant exists and underperforms the value-aware rule — see the ablation page. The contribution is the benchmark, the integrated system, and the honest component analysis. | <div dir="rtl">لا ادّعاء بجِدّة خوارزمية: فالمكوّنات الأولية (سياسات CNN وجولات التواصل وPPO والاستنساخ السلوكي) قياسية. ويوجد متغيّر بقائد متعلَّم بالكامل لكنه أدنى أداءً من القاعدة الواعية بالقيمة — انظر صفحة إزالة المكوّنات. الإسهام هو المنصّة، والنظام المتكامل، والتحليل الصريح للمكوّنات.</div> |  |
| `method.scopeTitle` | Honest scope | <div dir="rtl">نطاق صريح</div> |  |
| `method.storySteps.s1` | Threat scores τ = v/(1+d) are computed for every asset. | <div dir="rtl">تُحسب درجات التهديد τ = v/(1+d) لكل أصل.</div> |  |
| `method.storySteps.s2` | Agents are assigned to the top-threat assets. | <div dir="rtl">يُعيَّن الوكلاء على الأصول الأعلى تهديدًا.</div> |  |
| `method.storySteps.s3` | The tactical policy builds a firebreak at the fire frontier nearest its asset. | <div dir="rtl">تبني السياسة التكتيكية حاجزًا ناريًا عند جبهة الحريق الأقرب إلى الأصل.</div> |  |
| `method.storySteps.s4` | At the next macro-step the commander reassigns as threats shift. | <div dir="rtl">عند الخطوة الكلية التالية يعيد القائد التعيين مع تغيّر التهديدات.</div> |  |
| `method.storyboard` | How it decides, step by step | <div dir="rtl">كيف تقرّر، خطوة بخطوة</div> |  |
| `method.summary` | A value-aware hierarchical policy: a rule-based strategic commander assigns agents to threatened assets; a learned communicating tactical policy builds firebreaks. Trained by behavior cloning of a value-aware expert, then PPO fine-tuning. | <div dir="rtl">سياسة هرمية واعية بالقيمة: قائد استراتيجي قائم على قاعدة يعيّن الوكلاء على الأصول المهددة؛ وسياسة تكتيكية متعلَّمة متواصلة تبني حواجز نارية. تُدرَّب بالاستنساخ السلوكي عن خبير واعٍ بالقيمة ثم بالضبط الدقيق عبر PPO.</div> |  |
| `method.title` | Method: HierComm | <div dir="rtl">الطريقة: HierComm</div> |  |
| `method.training` | Training recipe | <div dir="rtl">وصفة التدريب</div> |  |
| `method.trainingBC` | Behavior-cloning warm start: 30 demonstration episodes from the value-aware asset-defense expert. | <div dir="rtl">بدء بالاستنساخ السلوكي: 30 حلقة عرض توضيحي من الخبير الواعي بالقيمة للدفاع عن الأصول.</div> |  |
| `method.trainingPPO` | PPO fine-tuning: 100k environment steps on the WEL/ISR delta reward with firebreak shaping. | <div dir="rtl">ضبط دقيق عبر PPO: ‏100 ألف خطوة بيئة على مكافأة فرق WEL/ISR مع تشكيل مكافأة الحاجز الناري.</div> |  |

## `metric.*`

| Key | English | Arabic | OK? |
|---|---|---|---|
| `metric.ce.hint` | How efficiently agents divide the work | <div dir="rtl">مدى كفاءة توزيع العمل بين الوكلاء</div> |  |
| `metric.ce.name` | Coordination Efficiency (CE) | <div dir="rtl">كفاءة التنسيق (CE)</div> |  |
| `metric.dwel.hint` | Loss prevented relative to no intervention — higher is better; 0 means no added value | <div dir="rtl">الخسارة التي جرى تفاديها مقارنةً بعدم التدخل — الأعلى أفضل؛ الصفر يعني عدم وجود قيمة مضافة</div> |  |
| `metric.dwel.name` | ΔWEL vs No-Op (within condition) | <div dir="rtl">‏ΔWEL مقارنةً بعدم التدخل (ضمن الحالة نفسها)</div> |  |
| `metric.isr.hint` | Fraction of assets unburned — higher is better | <div dir="rtl">نسبة الأصول غير المحترقة — الأعلى أفضل</div> |  |
| `metric.isr.name` | Infrastructure Survival Rate (ISR) | <div dir="rtl">معدل بقاء البنية التحتية (ISR)</div> |  |
| `metric.trs.hint` | Fraction of native ISR retained after cross-region transfer | <div dir="rtl">نسبة ما يُحتفظ به من معدل البقاء الأصلي بعد النقل بين المنطقتين</div> |  |
| `metric.trs.name` | Transfer Retention Score (TRS) | <div dir="rtl">درجة الاحتفاظ بعد النقل (TRS)</div> |  |
| `metric.wel.hint` | Total value of burned infrastructure — lower is better | <div dir="rtl">إجمالي قيمة البنية التحتية المحترقة — الأقل أفضل</div> |  |
| `metric.wel.name` | Weighted Economic Loss (WEL) | <div dir="rtl">الخسارة الاقتصادية الموزونة (WEL)</div> |  |

## `nav.*`

| Key | English | Arabic | OK? |
|---|---|---|---|
| `nav.ablation` | Ablation Analysis | <div dir="rtl">تحليل إزالة المكوّنات</div> |  |
| `nav.about` | About & Paper | <div dir="rtl">حول المشروع والورقة البحثية</div> |  |
| `nav.benchmark` | Benchmark & Data | <div dir="rtl">المعيار والبيانات</div> |  |
| `nav.generalization` | Generalization & Transfer | <div dir="rtl">التعميم والنقل</div> |  |
| `nav.method` | Method: HierComm | <div dir="rtl">الطريقة: HierComm</div> |  |
| `nav.more` | More | <div dir="rtl">المزيد</div> |  |
| `nav.overview` | Overview | <div dir="rtl">نظرة عامة</div> |  |
| `nav.replays` | Simulation Replays | <div dir="rtl">إعادة عرض المحاكاة</div> |  |
| `nav.reproducibility` | Reproducibility | <div dir="rtl">قابلية إعادة الإنتاج</div> |  |
| `nav.results` | Main Results | <div dir="rtl">النتائج الرئيسية</div> |  |
| `nav.robustness` | Robustness | <div dir="rtl">المتانة</div> |  |

## `overlay.*`

| Key | English | Arabic | OK? |
|---|---|---|---|
| `overlay.agents` | Agents | <div dir="rtl">الوكلاء</div> |  |
| `overlay.assets` | Assets | <div dir="rtl">الأصول</div> |  |
| `overlay.burnedOut` | Burned-out | <div dir="rtl">محترق بالكامل</div> |  |
| `overlay.fire` | Active fire | <div dir="rtl">الحريق النشط</div> |  |
| `overlay.targets` | Assigned targets | <div dir="rtl">الأهداف المعيّنة</div> |  |
| `overlay.treated` | Treated cells (firebreak) | <div dir="rtl">الخلايا المعالَجة (حاجز ناري)</div> |  |

## `overview.*`

| Key | English | Arabic | OK? |
|---|---|---|---|
| `overview.comparisonCaption` | HierComm versus baselines on the same episode ({{region}}). Full replays on the Replays page. | <div dir="rtl">HierComm مقابل السياسات المرجعية في الحلقة نفسها ({{region}}). إعادة العرض الكاملة في صفحة إعادة العرض.</div> |  |
| `overview.honesty` | Honest caveat: on Saudi Arabia the margin over the strongest flat baseline (MAPPO) is within seed variance ({{p}}). We report this plainly. | <div dir="rtl">توضيح صريح: في السعودية يبقى الفارق عن أقوى سياسة مسطّحة (MAPPO) ضمن حدود تباين البذور ({{p}}). نُقرّ بذلك بوضوح.</div> |  |
| `overview.subtitle` | A physics-grounded MARL benchmark on validated Cell2Fire fire physics, plus HierComm — a value-aware hierarchical multi-agent method. 2 regions · 7 policies · 5 seeds · frozen results. | <div dir="rtl">منصّة قياس متعددة الوكلاء قائمة على فيزياء Cell2Fire الموثّقة، مع HierComm — طريقة هرمية واعية بالقيمة متعددة الوكلاء. منطقتان · 7 سياسات · 5 بذور · نتائج مجمَّدة.</div> |  |
| `overview.tiles.isrBoth` | ISR (Saudi / California) | <div dir="rtl">‏ISR (السعودية / كاليفورنيا)</div> |  |
| `overview.tiles.isrContext` | best in both regions | <div dir="rtl">الأفضل في كلتا المنطقتين</div> |  |
| `overview.tiles.protocol` | Seeds × episodes | <div dir="rtl">البذور × الحلقات</div> |  |
| `overview.tiles.protocolContext` | held-out evaluation stream | <div dir="rtl">تقييم على بذور محجوبة عن التدريب</div> |  |
| `overview.tiles.protocolValue` | 5 × 15 | <div dir="rtl">5 × 15</div> |  |
| `overview.tiles.welCalifornia` | HierComm WEL — California | <div dir="rtl">‏WEL لطريقة HierComm — كاليفورنيا</div> |  |
| `overview.tiles.welSaudi` | HierComm WEL — Saudi | <div dir="rtl">‏WEL لطريقة HierComm — السعودية</div> |  |
| `overview.title` | Infrastructure-Aware Wildfire Coordination | <div dir="rtl">تنسيق مكافحة حرائق الغابات مع مراعاة البنية التحتية</div> |  |
| `overview.whatWorks.comms` | Explicit communication | <div dir="rtl">التواصل الصريح بين الوكلاء</div> |  |
| `overview.whatWorks.commsNote` | not a statistically significant driver at the 3-agent scale studied — see the ablation | <div dir="rtl">ليس عاملًا ذا دلالة إحصائية عند نطاق الوكلاء الثلاثة المدروس — انظر تحليل إزالة المكوّنات</div> |  |
| `overview.whatWorks.finetune` | RL fine-tuning | <div dir="rtl">الضبط الدقيق بالتعلم المعزَّز</div> |  |
| `overview.whatWorks.hierarchy` | Value-aware hierarchy | <div dir="rtl">الهرمية الواعية بالقيمة</div> |  |
| `overview.whatWorks.significant` | significant contribution | <div dir="rtl">إسهام ذو دلالة إحصائية</div> |  |
| `overview.whatWorks.tactical` | Learned tactical policy | <div dir="rtl">السياسة التكتيكية المتعلَّمة</div> |  |
| `overview.whatWorks.title` | What makes it work? | <div dir="rtl">ما الذي يجعلها تنجح؟</div> |  |

## `policy.*`

| Key | English | Arabic | OK? |
|---|---|---|---|
| `policy.all` | All policies | <div dir="rtl">جميع السياسات</div> |  |
| `policy.commnet` | CommNet | <div dir="rtl">CommNet</div> |  |
| `policy.greedy_risk` | Greedy-Risk | <div dir="rtl">الأشد خطورةً أولًا</div> |  |
| `policy.hiercomm_heur` | HierComm (ours) | <div dir="rtl">HierComm (طريقتنا)</div> |  |
| `policy.local_reactive` | Local Reactive | <div dir="rtl">التفاعلي المحلي</div> |  |
| `policy.mappo` | Flat MARL (MAPPO) | <div dir="rtl">التعلم المسطّح متعدد الوكلاء (MAPPO)</div> |  |
| `policy.noop` | No-Op (no intervention) | <div dir="rtl">بدون تدخل</div> |  |
| `policy.value_first` | Value-First (teacher) | <div dir="rtl">الأولوية للقيمة (المعلّم)</div> |  |

## `region.*`

| Key | English | Arabic | OK? |
|---|---|---|---|
| `region.both` | Both regions | <div dir="rtl">كلتا المنطقتين</div> |  |
| `region.california` | Northern California (WUI) | <div dir="rtl">شمال كاليفورنيا (مناطق التماس العمراني)</div> |  |
| `region.californiaShort` | California | <div dir="rtl">كاليفورنيا</div> |  |
| `region.label` | Region | <div dir="rtl">المنطقة</div> |  |
| `region.saudi` | Saudi Arabia — Eastern Province | <div dir="rtl">المملكة العربية السعودية — المنطقة الشرقية</div> |  |
| `region.saudiShort` | Saudi Arabia | <div dir="rtl">السعودية</div> |  |

## `replay.*`

| Key | English | Arabic | OK? |
|---|---|---|---|
| `replay.burned` | Burned cells | <div dir="rtl">الخلايا المحترقة</div> |  |
| `replay.compare` | Compare with… | <div dir="rtl">قارن مع…</div> |  |
| `replay.compareNone` | No comparison | <div dir="rtl">بدون مقارنة</div> |  |
| `replay.interactiveTitle` | Interactive replay | <div dir="rtl">إعادة عرض تفاعلية</div> |  |
| `replay.overlays` | Overlays | <div dir="rtl">الطبقات</div> |  |
| `replay.pause` | Pause | <div dir="rtl">إيقاف مؤقت</div> |  |
| `replay.play` | Play | <div dir="rtl">تشغيل</div> |  |
| `replay.policy` | Policy | <div dir="rtl">السياسة</div> |  |
| `replay.reset` | Restart | <div dir="rtl">إعادة التشغيل</div> |  |
| `replay.speed` | Speed | <div dir="rtl">السرعة</div> |  |
| `replay.step` | Step | <div dir="rtl">الخطوة</div> |  |
| `replay.trails` | Agent trails | <div dir="rtl">مسارات الوكلاء</div> |  |
| `replay.transferFrom` | transferred from {{region}} | <div dir="rtl">منقول من {{region}}</div> |  |
| `replay.verifiedCaption` | Recorded from the frozen checkpoints on the held-out evaluation stream (episode seed {{seed}}); the endpoint is verified against the frozen per-episode record. | <div dir="rtl">مسجَّلة من نقاط الحفظ المجمَّدة على تيار التقييم المحجوب (بذرة الحلقة {{seed}})؛ وتم التحقق من النتيجة النهائية مقابل السجل المجمَّد لكل حلقة.</div> |  |

## `replays.*`

| Key | English | Arabic | OK? |
|---|---|---|---|
| `replays.clickToPlay` | Click to load and play ({{size}}) | <div dir="rtl">انقر للتحميل والتشغيل ({{size}})</div> |  |
| `replays.gallery` | Replay gallery | <div dir="rtl">معرض إعادة العرض</div> |  |
| `replays.honesty` | Replays are recorded from the frozen released checkpoints on the held-out evaluation stream — the same rollouts behind the reported numbers. Single episodes are illustrative; aggregate results are in Main Results. | <div dir="rtl">إعادات العرض مسجَّلة من نقاط الحفظ المجمَّدة المنشورة على تيار التقييم المحجوب — وهي الجولات نفسها وراء الأرقام المنشورة. الحلقات المفردة للتوضيح؛ والنتائج المجمَّعة في صفحة النتائج الرئيسية.</div> |  |
| `replays.kind.comparison` | Side-by-side comparison | <div dir="rtl">مقارنة جنبًا إلى جنب</div> |  |
| `replays.kind.rollout` | Single-policy rollout | <div dir="rtl">جولة سياسة واحدة</div> |  |
| `replays.kind.transfer_cross` | Transfer — cross-region | <div dir="rtl">النقل — بين المنطقتين</div> |  |
| `replays.kind.transfer_native` | Transfer — native | <div dir="rtl">النقل — أصلي</div> |  |
| `replays.mediaUnavailable` | Media not included in this build — regenerate with build_dashboard_data.py --copy-media. | <div dir="rtl">الوسائط غير مضمّنة في هذا البناء — أعد توليدها عبر build_dashboard_data.py --copy-media.</div> |  |
| `replays.noCommnetNote` | CommNet rollouts are shown from the transfer study's native runs; the main sweep's GIF set does not include one. | <div dir="rtl">تُعرض جولات CommNet من التشغيلات الأصلية لدراسة النقل؛ فمجموعة صور المسح الرئيسي لا تتضمن واحدة.</div> |  |
| `replays.summary` | Pre-rendered rollouts of the frozen policies on both regions, including native-vs-transferred pairs. Every replay is recorded from the released checkpoints on the held-out evaluation stream — the same rollouts behind the reported numbers. | <div dir="rtl">جولات معروضة مسبقًا للسياسات المجمَّدة على كلتا المنطقتين، بما في ذلك أزواج «أصلي مقابل منقول». كل إعادة عرض مسجَّلة من نقاط الحفظ المنشورة على تيار التقييم المحجوب — وهي الجولات نفسها وراء الأرقام المنشورة.</div> |  |
| `replays.title` | Simulation Replays | <div dir="rtl">إعادة عرض المحاكاة</div> |  |

## `repro.*`

| Key | English | Arabic | OK? |
|---|---|---|---|
| `repro.artifact` | Artifact | <div dir="rtl">الأثر</div> |  |
| `repro.compute` | Compute disclosure | <div dir="rtl">الإفصاح عن الموارد الحاسوبية</div> |  |
| `repro.downloads` | Downloads | <div dir="rtl">التنزيلات</div> |  |
| `repro.downloadsNote` | Checkpoints are released upon acceptance (anonymity protocol). | <div dir="rtl">تُنشر نقاط الحفظ عند القبول (بروتوكول إخفاء الهوية).</div> |  |
| `repro.file` | File | <div dir="rtl">الملف</div> |  |
| `repro.hashes` | Checkpoint hashes (seed 42) | <div dir="rtl">بصمات نقاط الحفظ (البذرة 42)</div> |  |
| `repro.hashesSubtitle` | SHA-256 over the released checkpoints; the full {{n}}-checkpoint manifest is downloadable below | <div dir="rtl">‏SHA-256 لنقاط الحفظ المنشورة؛ يمكن تنزيل قائمة نقاط الحفظ الكاملة ({{n}}) أدناه</div> |  |
| `repro.protocol` | Protocol | <div dir="rtl">البروتوكول</div> |  |
| `repro.protocolItems.determinism` | Bit-identical CPU determinism (verified) | <div dir="rtl">حتمية مطابقة بِتًّا-بِبِتّ على المعالج (متحقَّق منها)</div> |  |
| `repro.protocolItems.eval` | Evaluation: held-out seed groups × 15 episodes; disjoint train/eval seed streams (no leakage) | <div dir="rtl">التقييم: مجموعات بذور محجوبة × 15 حلقة؛ تيارات بذور منفصلة للتدريب والتقييم (لا تسرّب)</div> |  |
| `repro.protocolItems.seeds` | 5 training seeds: {42, 1042, 2042, 3042, 4042} | <div dir="rtl">5 بذور تدريب: {42, 1042, 2042, 3042, 4042}</div> |  |
| `repro.protocolItems.unit` | Unit of analysis: per-training-seed mean (n = 5) | <div dir="rtl">وحدة التحليل: متوسط كل بذرة تدريب (n = 5)</div> |  |
| `repro.reproduce` | Reproduce the main table | <div dir="rtl">إعادة إنتاج الجدول الرئيسي</div> |  |
| `repro.summary` | Everything needed to verify or re-run the main results: the protocol, the frozen fingerprint, per-checkpoint hashes, and one-command reproduction. | <div dir="rtl">كل ما يلزم للتحقق من النتائج الرئيسية أو إعادة تشغيلها: البروتوكول، والبصمة المجمَّدة، وبصمات نقاط الحفظ، وأمر إعادة الإنتاج بسطر واحد.</div> |  |
| `repro.title` | Reproducibility | <div dir="rtl">قابلية إعادة الإنتاج</div> |  |
| `repro.verify` | Verify this dashboard | <div dir="rtl">التحقق من هذه المنصّة</div> |  |
| `repro.verifySteps` | Re-run the freeze script against the results directory; the fingerprint must match the value in the footer. | <div dir="rtl">أعد تشغيل نص التجميد على مجلد النتائج؛ يجب أن تطابق البصمة القيمة الظاهرة في التذييل.</div> |  |

## `results.*`

| Key | English | Arabic | OK? |
|---|---|---|---|
| `results.baseline` | Baseline | <div dir="rtl">السياسة المرجعية</div> |  |
| `results.comparison` | Comparison | <div dir="rtl">المقارنة</div> |  |
| `results.curvesSubtitle` | {{window}}-episode rolling mean · median across 5 seeds, aligned by episode (each method spans the ≥ {{n}} episodes common to its seeds; ≈100k env steps) | <div dir="rtl">متوسط متحرك على {{window}} حلقة · الوسيط عبر 5 بذور بمحاذاة رقم الحلقة (كل طريقة تغطي ≥ {{n}} حلقة مشتركة بين بذورها؛ ≈100 ألف خطوة بيئة)</div> |  |
| `results.envSteps` | ≈ env steps | <div dir="rtl">≈ خطوات البيئة</div> |  |
| `results.episodeAxis` | episode | <div dir="rtl">الحلقة</div> |  |
| `results.isrByPolicy` | ISR by policy | <div dir="rtl">‏ISR حسب السياسة</div> |  |
| `results.perSeedToggle` | Show per-seed curves | <div dir="rtl">إظهار منحنيات كل بذرة</div> |  |
| `results.statsPanel` | Significance — HierComm vs baselines | <div dir="rtl">الدلالة الإحصائية — HierComm مقابل السياسات المرجعية</div> |  |
| `results.statsSubtitle` | Welch t-test for learned baselines; one-sample t-test against deterministic heuristics | <div dir="rtl">اختبار Welch مع السياسات المتعلَّمة؛ واختبار t لعينة واحدة مع السياسات الحتمية</div> |  |
| `results.summary` | The interactive twin of the paper's Table 1: Weighted Economic Loss and Infrastructure Survival Rate for all 7 policies, with bootstrap 95% CIs and every per-seed value visible. Learned methods aggregate 5 training seeds × 15 held-out episodes; heuristics are deterministic. | <div dir="rtl">النسخة التفاعلية من الجدول 1 في الورقة: الخسارة الاقتصادية الموزونة ومعدل بقاء البنية التحتية للسياسات السبع، مع فترات ثقة 95٪ وقيم كل بذرة ظاهرة. الطرق المتعلَّمة تجمع 5 بذور تدريب × 15 حلقة محجوبة؛ والسياسات الاسترشادية حتمية.</div> |  |
| `results.title` | Main Results | <div dir="rtl">النتائج الرئيسية</div> |  |
| `results.trainingCurves` | Training curves — WEL over training | <div dir="rtl">منحنيات التدريب — WEL خلال التدريب</div> |  |
| `results.welByPolicy` | WEL by policy | <div dir="rtl">‏WEL حسب السياسة</div> |  |

## `robustness.*`

| Key | English | Arabic | OK? |
|---|---|---|---|
| `robustness.anchor` | No-Op worsens monotonically with difficulty — the sweep is doing what it should. | <div dir="rtl">«بدون تدخل» يسوء باطّراد مع الصعوبة — المسح يعمل كما ينبغي.</div> |  |
| `robustness.callout` | HierComm is the lowest-WEL policy in {{hc}} / {{total}} region × difficulty cells. | <div dir="rtl">HierComm هي السياسة الأدنى في WEL في {{hc}} / {{total}} من خلايا المنطقة × الصعوبة.</div> |  |
| `robustness.summary` | An easy/medium/hard difficulty sweep (wind scaling and fire-weather index). No-Op's monotone worsening anchors the axis; HierComm stays lowest-WEL in every region × difficulty cell. | <div dir="rtl">مسح صعوبة بثلاثة مستويات (سهل/متوسط/صعب) عبر تحجيم الرياح ومؤشر طقس الحرائق. تدهور «بدون تدخل» المطّرد يثبّت المحور؛ وتبقى HierComm صاحبة أدنى WEL في كل خلايا المنطقة × الصعوبة.</div> |  |
| `robustness.sweep` | WEL across difficulty | <div dir="rtl">‏WEL عبر مستويات الصعوبة</div> |  |
| `robustness.title` | Robustness | <div dir="rtl">المتانة</div> |  |

## `stats.*`

| Key | English | Arabic | OK? |
|---|---|---|---|
| `stats.ci95` | 95% confidence interval | <div dir="rtl">فترة ثقة 95٪</div> |  |
| `stats.deterministic` | Deterministic policy — no seed variance | <div dir="rtl">سياسة حتمية — لا تباين بين البذور</div> |  |
| `stats.effect` | Effect size (Cohen's d) | <div dir="rtl">حجم الأثر (d)</div> |  |
| `stats.mean` | mean | <div dir="rtl">المتوسط</div> |  |
| `stats.nSeeds` | {{n}} seeds | <div dir="rtl">{{n}} بذور</div> |  |
| `stats.ns` | Not significant (n.s.) | <div dir="rtl">غير دالّ إحصائيًا</div> |  |
| `stats.oneSample` | one-sample t-test | <div dir="rtl">اختبار t لعينة واحدة</div> |  |
| `stats.perSeed` | per-seed values | <div dir="rtl">القيم لكل بذرة</div> |  |
| `stats.pvalue` | p-value | <div dir="rtl">القيمة الاحتمالية (p)</div> |  |
| `stats.sig` | Statistically significant | <div dir="rtl">ذو دلالة إحصائية</div> |  |
| `stats.test` | Test | <div dir="rtl">الاختبار</div> |  |
| `stats.welch` | Welch t-test | <div dir="rtl">اختبار Welch</div> |  |

## `theme.*`

| Key | English | Arabic | OK? |
|---|---|---|---|
| `theme.dark` | Dark mode | <div dir="rtl">الوضع الداكن</div> |  |
| `theme.light` | Light mode | <div dir="rtl">الوضع الفاتح</div> |  |

## `transfer.*`

| Key | English | Arabic | OK? |
|---|---|---|---|
| `transfer.evaluatedOn` | Evaluated on | <div dir="rtl">قُيّم على</div> |  |
| `transfer.matrixTitle` | Transfer matrix — WEL | <div dir="rtl">مصفوفة النقل — WEL</div> |  |
| `transfer.native` | Native | <div dir="rtl">أصلي</div> |  |
| `transfer.trainedOn` | Trained on | <div dir="rtl">دُرّب على</div> |  |
| `transfer.transferred` | Transferred | <div dir="rtl">منقول</div> |  |
| `transfer.trsTitle` | Transfer retention (TRS, ISR-based) | <div dir="rtl">الاحتفاظ بعد النقل (TRS بدلالة ISR)</div> |  |


## Sign-off (release gate — Dashboard_Guide.md §10.3/§16)

- [ ] Every string reviewed against its English source
- [ ] No claim is stronger or weaker in Arabic than in English
- [ ] Statistical terminology verified by a reviewer with ML familiarity
- [ ] Method names remain in Latin script; hashes/seeds/paths render LTR
- [ ] Corrections applied to `dashboard/src/i18n/locales/ar.json` and `npm test` passes

Reviewer name: ______________________  Date: ____________

