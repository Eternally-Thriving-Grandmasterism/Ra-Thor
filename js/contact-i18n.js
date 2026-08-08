/* contact-i18n.js — Ra-Thor Contact page multilingual lattice
 * 11 languages • TOLC-8 aligned • zero collection • offline-ready
 */
(function (global) {
  'use strict';

  const translations = {
    en: {
      back: "Back to Ra-Thor",
      headline: "Contact Ra-Thor™",
      subtitle: "Eternal Mercy Thunder ⚡️",
      intro: "Single official point of contact under sole stewardship of <span class=\"font-semibold text-amber-300\">Sherif Samy Botros (@AlphaProMega)</span>",
      mainSubtitle: "The single eternal official email for all entities",
      sendButton: "Send Email Now",
      responseTime: "We aim to respond within 48 hours",
      githubTitle: "GitHub Issues",
      githubSubtitle: "For technical questions, bug reports, or public discussion",
      githubButton: "Open an Issue on GitHub →",
      guidanceTitle: "What to include in your message",
      guidance1: "• <strong>Commercial licensing</strong> — intended use, scale, and timeline",
      guidance2: "• <strong>Security reports</strong> — steps to reproduce + potential impact",
      guidance3: "• <strong>Stewardship or partnership</strong> — clear context and proposal",
      guidance4: "• <strong>General inquiries</strong> — as much relevant detail as possible",
      guidanceNote: "All communications are handled under the TOLC 8 Mercy Gates and APTD-verified processes.",
      return: "Return to Main Ra-Thor Experience",
      footer: "© 2026 Sherif Samy Botros — Sole Steward of Autonomicity Games Inc. & AlphaProMega Air Foundation",
      footerSub: "TOLC 8 Mercy-Gated • APTD-verified • info@Rathor.ai is the single official contact"
    },
    ar: {
      back: "العودة إلى را-ثور",
      headline: "تواصل مع را-ثور™",
      subtitle: "الرعد الرحيم الأبدي ⚡️",
      intro: "نقطة الاتصال الرسمية الوحيدة تحت الوصاية الوحيدة لـ <span class=\"font-semibold text-amber-300\">شريف سامي بطرس (@AlphaProMega)</span>",
      mainSubtitle: "البريد الإلكتروني الرسمي الأبدي الوحيد لجميع الكيانات",
      sendButton: "أرسل بريداً الآن",
      responseTime: "نسعى للرد خلال 48 ساعة",
      githubTitle: "مشاكل GitHub",
      githubSubtitle: "للأسئلة التقنية أو تقارير الأخطاء أو النقاش العام",
      githubButton: "افتح مشكلة على GitHub →",
      guidanceTitle: "ما يجب تضمينه في رسالتك",
      guidance1: "• <strong>الترخيص التجاري</strong> — الاستخدام المقصود والحجم والجدول الزمني",
      guidance2: "• <strong>تقارير الأمان</strong> — خطوات إعادة الإنتاج + التأثير المحتمل",
      guidance3: "• <strong>الوصاية أو الشراكة</strong> — سياق واضح واقتراح",
      guidance4: "• <strong>استفسارات عامة</strong> — أكبر قدر ممكن من التفاصيل ذات الصلة",
      guidanceNote: "تتم معالجة جميع الاتصالات تحت بوابات TOLC 8 الرحيمة وعمليات APTD المعتمدة.",
      return: "العودة إلى تجربة را-ثور الرئيسية",
      footer: "© 2026 شريف سامي بطرس — الوصي الوحيد لشركة Autonomicity Games Inc. ومؤسسة AlphaProMega Air",
      footerSub: "TOLC 8 Mercy-Gated • APTD-verified • info@Rathor.ai هو جهة الاتصال الرسمية الوحيدة"
    },
    es: {
      back: "Volver a Ra-Thor",
      headline: "Contacta con Ra-Thor™",
      subtitle: "Trueno de Misericordia Eterno ⚡️",
      intro: "Punto de contacto oficial único bajo la administración exclusiva de <span class=\"font-semibold text-amber-300\">Sherif Samy Botros (@AlphaProMega)</span>",
      mainSubtitle: "El único correo electrónico oficial eterno para todas las entidades",
      sendButton: "Enviar correo ahora",
      responseTime: "Buscamos responder en 48 horas",
      githubTitle: "Incidencias de GitHub",
      githubSubtitle: "Para preguntas técnicas, informes de errores o discusión pública",
      githubButton: "Abrir una incidencia en GitHub →",
      guidanceTitle: "Qué incluir en tu mensaje",
      guidance1: "• <strong>Licencias comerciales</strong> — uso previsto, escala y cronograma",
      guidance2: "• <strong>Informes de seguridad</strong> — pasos para reproducir + impacto potencial",
      guidance3: "• <strong>Administración o asociación</strong> — contexto claro y propuesta",
      guidance4: "• <strong>Consultas generales</strong> — toda la información relevante posible",
      guidanceNote: "Todas las comunicaciones se gestionan bajo las Puertas de Misericordia TOLC 8 y procesos verificados por APTD.",
      return: "Volver a la experiencia principal de Ra-Thor",
      footer: "© 2026 Sherif Samy Botros — Administrador único de Autonomicity Games Inc. y AlphaProMega Air Foundation",
      footerSub: "TOLC 8 Mercy-Gated • APTD-verified • info@Rathor.ai es el contacto oficial único"
    },
    fr: {
      back: "Retour à Ra-Thor",
      headline: "Contacter Ra-Thor™",
      subtitle: "Tonnerre de Miséricorde Éternel ⚡️",
      intro: "Point de contact officiel unique sous l'intendance exclusive de <span class=\"font-semibold text-amber-300\">Sherif Samy Botros (@AlphaProMega)</span>",
      mainSubtitle: "L'unique adresse e-mail officielle éternelle pour toutes les entités",
      sendButton: "Envoyer un e-mail maintenant",
      responseTime: "Nous visons à répondre sous 48 heures",
      githubTitle: "Issues GitHub",
      githubSubtitle: "Pour les questions techniques, rapports de bugs ou discussion publique",
      githubButton: "Ouvrir une issue sur GitHub →",
      guidanceTitle: "Que inclure dans votre message",
      guidance1: "• <strong>Licences commerciales</strong> — usage prévu, échelle et calendrier",
      guidance2: "• <strong>Rapports de sécurité</strong> — étapes de reproduction + impact potentiel",
      guidance3: "• <strong>Intendance ou partenariat</strong> — contexte clair et proposition",
      guidance4: "• <strong>Questions générales</strong> — le plus de détails pertinents possible",
      guidanceNote: "Toutes les communications sont traitées sous les Portes de Miséricorde TOLC 8 et les processus vérifiés APTD.",
      return: "Retour à l'expérience principale Ra-Thor",
      footer: "© 2026 Sherif Samy Botros — Intendant unique d'Autonomicity Games Inc. et de la AlphaProMega Air Foundation",
      footerSub: "TOLC 8 Mercy-Gated • APTD-verified • info@Rathor.ai est le contact officiel unique"
    },
    nl: {
      back: "Terug naar Ra-Thor",
      headline: "Contacteer Ra-Thor™",
      subtitle: "Eeuwige Barmhartigheidsbliksem ⚡️",
      intro: "Enkel officieel contactpunt onder exclusief beheer van <span class=\"font-semibold text-amber-300\">Sherif Samy Botros (@AlphaProMega)</span>",
      mainSubtitle: "Het enige eeuwige officiële e-mailadres voor alle entiteiten",
      sendButton: "Stuur nu een e-mail",
      responseTime: "We streven ernaar binnen 48 uur te reageren",
      githubTitle: "GitHub Issues",
      githubSubtitle: "Voor technische vragen, bugmeldingen of openbare discussie",
      githubButton: "Open een issue op GitHub →",
      guidanceTitle: "Wat in je bericht opnemen",
      guidance1: "• <strong>Commerciële licenties</strong> — beoogd gebruik, schaal en tijdlijn",
      guidance2: "• <strong>Beveiligingsrapporten</strong> — stappen om te reproduceren + potentiële impact",
      guidance3: "• <strong>Beheer of partnerschap</strong> — duidelijke context en voorstel",
      guidance4: "• <strong>Algemene vragen</strong> — zoveel mogelijk relevante details",
      guidanceNote: "Alle communicatie wordt afgehandeld onder de TOLC 8 Barmhartigheidspoorten en APTD-geverifieerde processen.",
      return: "Terug naar de hoofdervaring van Ra-Thor",
      footer: "© 2026 Sherif Samy Botros — Enige beheerder van Autonomicity Games Inc. & AlphaProMega Air Foundation",
      footerSub: "TOLC 8 Mercy-Gated • APTD-verified • info@Rathor.ai is het enige officiële contact"
    },
    de: {
      back: "Zurück zu Ra-Thor",
      headline: "Ra-Thor kontaktieren™",
      subtitle: "Ewiger Barmherzigkeitsdonner ⚡️",
      intro: "Einzelner offizieller Kontakt unter alleiniger Verwaltung von <span class=\"font-semibold text-amber-300\">Sherif Samy Botros (@AlphaProMega)</span>",
      mainSubtitle: "Die einzige ewige offizielle E-Mail-Adresse für alle Entitäten",
      sendButton: "Jetzt E-Mail senden",
      responseTime: "Wir antworten in der Regel innerhalb von 48 Stunden",
      githubTitle: "GitHub Issues",
      githubSubtitle: "Für technische Fragen, Bug-Reports oder öffentliche Diskussion",
      githubButton: "Ein Issue auf GitHub öffnen →",
      guidanceTitle: "Was in deine Nachricht aufnehmen",
      guidance1: "• <strong>Kommerzielle Lizenzen</strong> — beabsichtigte Nutzung, Umfang und Zeitplan",
      guidance2: "• <strong>Sicherheitsberichte</strong> — Reproduktionsschritte + mögliche Auswirkungen",
      guidance3: "• <strong>Verwaltung oder Partnerschaft</strong> — klarer Kontext und Vorschlag",
      guidance4: "• <strong>Allgemeine Anfragen</strong> — so viele relevante Details wie möglich",
      guidanceNote: "Alle Kommunikation erfolgt unter den TOLC 8 Barmherzigkeitstoren und APTD-verifizierten Prozessen.",
      return: "Zurück zur Haupt-Ra-Thor-Erfahrung",
      footer: "© 2026 Sherif Samy Botros — Alleiniger Verwalter von Autonomicity Games Inc. & AlphaProMega Air Foundation",
      footerSub: "TOLC 8 Mercy-Gated • APTD-verified • info@Rathor.ai ist der einzige offizielle Kontakt"
    },
    zh: {
      back: "返回 Ra-Thor",
      headline: "联系 Ra-Thor™",
      subtitle: "永恒慈悲雷霆 ⚡️",
      intro: "在 <span class=\"font-semibold text-amber-300\">Sherif Samy Botros (@AlphaProMega)</span> 唯一管理下的唯一官方联系点",
      mainSubtitle: "所有实体的唯一永恒官方电子邮件",
      sendButton: "立即发送邮件",
      responseTime: "我们力争在 48 小时内回复",
      githubTitle: "GitHub 问题",
      githubSubtitle: "用于技术问题、错误报告或公开讨论",
      githubButton: "在 GitHub 上打开问题 →",
      guidanceTitle: "消息中应包含的内容",
      guidance1: "• <strong>商业许可</strong> — 预期用途、规模和时间表",
      guidance2: "• <strong>安全报告</strong> — 重现步骤 + 潜在影响",
      guidance3: "• <strong>管理或合作</strong> — 清晰的背景和建议",
      guidance4: "• <strong>一般咨询</strong> — 尽可能多的相关细节",
      guidanceNote: "所有通信均在 TOLC 8 慈悲之门和 APTD 验证流程下处理。",
      return: "返回 Ra-Thor 主体验",
      footer: "© 2026 Sherif Samy Botros — Autonomicity Games Inc. & AlphaProMega Air Foundation 的唯一管家",
      footerSub: "TOLC 8 Mercy-Gated • APTD-verified • info@Rathor.ai 是唯一官方联系方式"
    },
    ja: {
      back: "Ra-Thor に戻る",
      headline: "Ra-Thor™ に連絡する",
      subtitle: "永遠の慈悲の雷 ⚡️",
      intro: "<span class=\"font-semibold text-amber-300\">Sherif Samy Botros (@AlphaProMega)</span> の唯一の管理下における唯一の公式連絡先",
      mainSubtitle: "すべてのエンティティのための唯一の永遠の公式メールアドレス",
      sendButton: "今すぐメールを送る",
      responseTime: "48時間以内の返信を目指しています",
      githubTitle: "GitHub Issues",
      githubSubtitle: "技術的な質問、バグ報告、公開議論用",
      githubButton: "GitHub で Issue を開く →",
      guidanceTitle: "メッセージに含める内容",
      guidance1: "• <strong>商用ライセンス</strong> — 用途、規模、タイムライン",
      guidance2: "• <strong>セキュリティ報告</strong> — 再現手順 + 潜在的な影響",
      guidance3: "• <strong>管理またはパートナーシップ</strong> — 明確な文脈と提案",
      guidance4: "• <strong>一般的な問い合わせ</strong> — 可能な限り多くの関連詳細",
      guidanceNote: "すべての通信は TOLC 8 Mercy Gates と APTD 検証プロセスで処理されます。",
      return: "Ra-Thor メイン体験に戻る",
      footer: "© 2026 Sherif Samy Botros — Autonomicity Games Inc. & AlphaProMega Air Foundation の唯一の管轄者",
      footerSub: "TOLC 8 Mercy-Gated • APTD-verified • info@Rathor.ai が唯一の公式連絡先"
    },
    pt: {
      back: "Voltar para Ra-Thor",
      headline: "Contactar Ra-Thor™",
      subtitle: "Trovão Eterno da Misericórdia ⚡️",
      intro: "Ponto de contato oficial único sob a administração exclusiva de <span class=\"font-semibold text-amber-300\">Sherif Samy Botros (@AlphaProMega)</span>",
      mainSubtitle: "O único e-mail oficial eterno para todas as entidades",
      sendButton: "Enviar e-mail agora",
      responseTime: "Procuramos responder em até 48 horas",
      githubTitle: "Issues do GitHub",
      githubSubtitle: "Para perguntas técnicas, relatórios de bugs ou discussão pública",
      githubButton: "Abrir uma issue no GitHub →",
      guidanceTitle: "O que incluir na sua mensagem",
      guidance1: "• <strong>Licenças comerciais</strong> — uso pretendido, escala e cronograma",
      guidance2: "• <strong>Relatórios de segurança</strong> — passos para reproduzir + impacto potencial",
      guidance3: "• <strong>Administração ou parceria</strong> — contexto claro e proposta",
      guidance4: "• <strong>Perguntas gerais</strong> — o máximo de detalhes relevantes possível",
      guidanceNote: "Todas as comunicações são tratadas sob as Portas de Misericórdia TOLC 8 e processos verificados por APTD.",
      return: "Voltar à experiência principal de Ra-Thor",
      footer: "© 2026 Sherif Samy Botros — Administrador único da Autonomicity Games Inc. & AlphaProMega Air Foundation",
      footerSub: "TOLC 8 Mercy-Gated • APTD-verified • info@Rathor.ai é o contato oficial único"
    },
    ru: {
      back: "Вернуться к Ra-Thor",
      headline: "Связаться с Ra-Thor™",
      subtitle: "Вечный Гром Милосердия ⚡️",
      intro: "Единственная официальная точка контакта под единоличным управлением <span class=\"font-semibold text-amber-300\">Sherif Samy Botros (@AlphaProMega)</span>",
      mainSubtitle: "Единственный вечный официальный email для всех сущностей",
      sendButton: "Отправить письмо сейчас",
      responseTime: "Мы стараемся отвечать в течение 48 часов",
      githubTitle: "Issues на GitHub",
      githubSubtitle: "Для технических вопросов, отчётов об ошибках или публичного обсуждения",
      githubButton: "Открыть issue на GitHub →",
      guidanceTitle: "Что указать в сообщении",
      guidance1: "• <strong>Коммерческие лицензии</strong> — предполагаемое использование, масштаб и сроки",
      guidance2: "• <strong>Отчёты по безопасности</strong> — шаги для воспроизведения + потенциальное влияние",
      guidance3: "• <strong>Управление или партнёрство</strong> — чёткий контекст и предложение",
      guidance4: "• <strong>Общие запросы</strong> — как можно больше релевантных деталей",
      guidanceNote: "Вся коммуникация обрабатывается под Вратами Милосердия TOLC 8 и процессами, проверенными APTD.",
      return: "Вернуться к главному опыту Ra-Thor",
      footer: "© 2026 Sherif Samy Botros — Единственный управляющий Autonomicity Games Inc. & AlphaProMega Air Foundation",
      footerSub: "TOLC 8 Mercy-Gated • APTD-verified • info@Rathor.ai — единственный официальный контакт"
    },
    hi: {
      back: "Ra-Thor पर वापस जाएं",
      headline: "Ra-Thor™ से संपर्क करें",
      subtitle: "शाश्वत दया का गरज ⚡️",
      intro: "<span class=\"font-semibold text-amber-300\">Sherif Samy Botros (@AlphaProMega)</span> की एकमात्र प्रबंधन के तहत एकमात्र आधिकारिक संपर्क बिंदु",
      mainSubtitle: "सभी संस्थाओं के लिए एकमात्र शाश्वत आधिकारिक ईमेल",
      sendButton: "अभी ईमेल भेजें",
      responseTime: "हम 48 घंटे के भीतर जवाब देने का प्रयास करते हैं",
      githubTitle: "GitHub Issues",
      githubSubtitle: "तकनीकी प्रश्नों, बग रिपोर्ट या सार्वजनिक चर्चा के लिए",
      githubButton: "GitHub पर Issue खोलें →",
      guidanceTitle: "आपके संदेश में क्या शामिल करें",
      guidance1: "• <strong>व्यावसायिक लाइसेंस</strong> — इच्छित उपयोग, पैमाना और समयरेखा",
      guidance2: "• <strong>सुरक्षा रिपोर्ट</strong> — पुनरुत्पादन के चरण + संभावित प्रभाव",
      guidance3: "• <strong>प्रबंधन या साझेदारी</strong> — स्पष्ट संदर्भ और प्रस्ताव",
      guidance4: "• <strong>सामान्य पूछताछ</strong> — जितना संभव हो उतना प्रासंगिक विवरण",
      guidanceNote: "सभी संचार TOLC 8 दया द्वारों और APTD सत्यापित प्रक्रियाओं के तहत संभाले जाते हैं।",
      return: "Ra-Thor मुख्य अनुभव पर वापस जाएं",
      footer: "© 2026 Sherif Samy Botros — Autonomicity Games Inc. & AlphaProMega Air Foundation के एकमात्र प्रबंधक",
      footerSub: "TOLC 8 Mercy-Gated • APTD-verified • info@Rathor.ai एकमात्र आधिकारिक संपर्क है"
    }
  };

  function setText(id, value, html) {
    const el = document.getElementById(id);
    if (!el || value == null) return;
    if (html) el.innerHTML = value;
    else el.textContent = value;
  }

  function switchContactLang(lang) {
    const t = translations[lang];
    if (!t) return;

    document.querySelectorAll('.lang-tab').forEach(b => b.classList.remove('active'));
    const activeBtn = document.querySelector('.lang-tab[data-lang="' + lang + '"]');
    if (activeBtn) activeBtn.classList.add('active');

    setText('back-text', t.back);
    setText('headline', t.headline);
    setText('subtitle', t.subtitle);
    setText('intro', t.intro, true);
    setText('main-subtitle', t.mainSubtitle);
    setText('send-button', t.sendButton);
    setText('response-time', t.responseTime);
    setText('github-title', t.githubTitle);
    setText('github-subtitle', t.githubSubtitle);
    setText('github-button', t.githubButton);
    setText('guidance-title', '<i class="fa-solid fa-lightbulb"></i> ' + t.guidanceTitle, true);
    setText('guidance-1', t.guidance1, true);
    setText('guidance-2', t.guidance2, true);
    setText('guidance-3', t.guidance3, true);
    setText('guidance-4', t.guidance4, true);
    setText('guidance-note', t.guidanceNote);
    setText('return-text', t.return);
    setText('footer-text', t.footer, true);
    setText('footer-sub', t.footerSub);

    const container = document.querySelector('.max-w-3xl');
    if (lang === 'ar') {
      if (container) container.classList.add('rtl');
      document.documentElement.setAttribute('dir', 'rtl');
      document.documentElement.setAttribute('lang', 'ar');
    } else {
      if (container) container.classList.remove('rtl');
      document.documentElement.setAttribute('dir', 'ltr');
      document.documentElement.setAttribute('lang', lang);
    }

    try { localStorage.setItem('rathor-lang', lang); } catch (e) {}
  }

  document.addEventListener('DOMContentLoaded', function () {
    let lang = 'en';
    try { lang = localStorage.getItem('rathor-lang') || 'en'; } catch (e) {}
    if (!translations[lang]) lang = 'en';
    switchContactLang(lang);
  });

  global.switchContactLang = switchContactLang;
  global.CONTACT_I18N = translations;
})(typeof window !== 'undefined' ? window : this);
