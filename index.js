require('dotenv').config();
const express   = require('express');
const cors      = require('cors');
const rateLimit = require('express-rate-limit');
const Anthropic = require('@anthropic-ai/sdk');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const Groq      = require('groq-sdk');
const Together  = require('together-ai');
const jwt       = require('jsonwebtoken');
const https     = require('https');

// fetch HTTP avec https natif — contourne les problèmes SSL/proxy de fetch()
function fetchHttps(url, options = {}) {
  return new Promise((resolve, reject) => {
    const urlObj = new URL(url);
    const body   = options.body ? Buffer.from(options.body) : null;

    const reqOptions = {
      hostname: urlObj.hostname,
      path:     urlObj.pathname + urlObj.search,
      method:   options.method || 'GET',
      headers:  {
        ...options.headers,
        ...(body ? { 'Content-Length': body.length } : {}),
      },
      rejectUnauthorized: false,   // accepte les certificats auto-signés
    };

    const req = https.request(reqOptions, (res) => {
      const chunks = [];
      res.on('data', (chunk) => chunks.push(chunk));
      res.on('end', () => {
        const buffer = Buffer.concat(chunks);
        resolve({
          ok: res.statusCode >= 200 && res.statusCode < 300,
          status: res.statusCode,
          headers: { get: (h) => res.headers[h.toLowerCase()] ?? null },
          arrayBuffer: () => Promise.resolve(buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength)),
          text: () => Promise.resolve(buffer.toString('utf-8')),
          json: () => Promise.resolve(JSON.parse(buffer.toString('utf-8'))),
        });
      });
    });

    req.on('error', reject);
    req.setTimeout(60000, () => { req.destroy(new Error('timeout')); });
    if (body) req.write(body);
    req.end();
  });
}

// ─── Configuration ─────────────────────────────────────────────────────────────

const PORT                = process.env.PORT || 3000;
const AI_PROVIDER         = process.env.AI_PROVIDER || 'groq'; // 'groq' | 'gemini' | 'claude'
const ANTHROPIC_KEY       = process.env.ANTHROPIC_API_KEY;
const GEMINI_KEY          = process.env.GEMINI_API_KEY;
const GROQ_KEY            = process.env.GROQ_API_KEY;
const SUPABASE_JWT_SECRET = process.env.SUPABASE_JWT_SECRET;
const ALLOWED_ORIGINS     = (process.env.ALLOWED_ORIGINS || '*').split(',');
const HF_TOKEN            = process.env.HF_TOKEN;
const TOGETHER_KEY        = process.env.TOGETHER_API_KEY;
const together            = TOGETHER_KEY && TOGETHER_KEY !== 'your_together_key_here'
  ? new Together({ apiKey: TOGETHER_KEY })
  : null;

// Validations
if (AI_PROVIDER === 'groq'   && !GROQ_KEY)      console.warn('⚠️  GROQ_API_KEY manquante');
if (AI_PROVIDER === 'gemini' && !GEMINI_KEY)    console.warn('⚠️  GEMINI_API_KEY manquante');
if (AI_PROVIDER === 'claude' && !ANTHROPIC_KEY) console.warn('⚠️  ANTHROPIC_API_KEY manquante');

// Clients IA
const anthropic   = ANTHROPIC_KEY ? new Anthropic.default({ apiKey: ANTHROPIC_KEY }) : null;
const geminiAI    = GEMINI_KEY    ? new GoogleGenerativeAI(GEMINI_KEY)               : null;
const geminiModel = geminiAI?.getGenerativeModel({ model: 'gemini-1.5-flash-8b' }, { apiVersion: 'v1beta' });
const groq        = GROQ_KEY      ? new Groq({ apiKey: GROQ_KEY })                   : null;

// ─── Abstraction IA ────────────────────────────────────────────────────────────

async function analyserDessinIA(imageBase64) {
  const prompt = `C'est le dessin fait par un enfant. Analyse-le et réponds UNIQUEMENT en JSON valide :
{
  "personnages": ["personnage 1", "personnage 2"],
  "couleurs": ["couleur 1", "couleur 2"],
  "decor": "description du décor en une phrase",
  "ambiance": "ambiance en 2-3 mots",
  "descriptionComplete": "description complète en 2-3 phrases"
}
Réponds UNIQUEMENT avec le JSON, rien d'autre.`;

  if (AI_PROVIDER === 'groq') {
    // Modèles vision Groq avec fallback
    const GROQ_VISION = [
      'meta-llama/llama-4-scout-17b-16e-instruct',
      'meta-llama/llama-4-maverick-17b-128e-instruct',
      'llama-3.2-11b-vision-preview',
    ];
    for (const model of GROQ_VISION) {
      try {
        const response = await groq.chat.completions.create({
          model,
          max_tokens: 500,
          messages: [{
            role: 'user',
            content: [
              { type: 'image_url', image_url: { url: `data:image/jpeg;base64,${imageBase64}` } },
              { type: 'text', text: prompt },
            ],
          }],
        });
        return response.choices[0].message.content;
      } catch (err) {
        const s = err?.status;
        if (s === 429 || s === 400 || err?.message?.includes('decommissioned')) {
          console.warn(`[groq vision] ${model} indisponible (${s}), essai du suivant…`);
          continue;
        }
        throw err;
      }
    }
    throw new Error('Tous les modèles vision Groq sont en limite.');
  }

  if (AI_PROVIDER === 'gemini') {
    const result = await geminiModel.generateContent([
      { inlineData: { data: imageBase64, mimeType: 'image/jpeg' } },
      prompt,
    ]);
    return result.response.text();
  }

  // Claude
  const response = await anthropic.messages.create({
    model: 'claude-opus-4-7',
    max_tokens: 600,
    messages: [{
      role: 'user',
      content: [
        { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: imageBase64 } },
        { type: 'text', text: prompt },
      ],
    }],
  });
  return response.content[0].text;
}

// Modèles Groq actifs (mis à jour juin 2026)
// https://console.groq.com/docs/models
const GROQ_MODELES = [
  { model: 'llama-3.3-70b-versatile',  max_tokens: 3000 }, // meilleure qualité
  { model: 'gemma2-9b-it',             max_tokens: 2500 }, // 500k TPD — bon fallback
  { model: 'llama-3.1-8b-instant',     max_tokens: 2500 }, // 500k TPD — très léger
  { model: 'llama-3.2-3b-preview',     max_tokens: 2000 }, // dernier recours
];

async function genererHistoireIA(promptTexte) {
  if (AI_PROVIDER === 'groq') {
    // Essaie les modèles dans l'ordre, passe au suivant si rate limit (429)
    for (const { model, max_tokens } of GROQ_MODELES) {
      try {
        const response = await groq.chat.completions.create({
          model,
          max_tokens,
          messages: [{ role: 'user', content: promptTexte }],
        });
        console.log(`[groq] modèle utilisé: ${model}`);
        return response.choices[0].message.content;
      } catch (err) {
        const status = err?.status ?? err?.response?.status;
        const isSkippable = status === 429          // rate limit
          || status === 400                          // modèle décommissionné ou invalide
          || err?.message?.includes('decommissioned')
          || err?.message?.includes('not found');
        if (isSkippable) {
          console.warn(`[groq] modèle ${model} indisponible (${status}), essai du suivant…`);
          continue;
        }
        throw err; // autre erreur → propage
      }
    }
    throw new Error('Tous les modèles Groq sont en limite. Réessaie dans 1h.');
  }

  if (AI_PROVIDER === 'gemini') {
    const result = await geminiModel.generateContent(promptTexte);
    return result.response.text();
  }

  // Claude
  const response = await anthropic.messages.create({
    model: 'claude-opus-4-7',
    max_tokens: 1600,
    messages: [{ role: 'user', content: promptTexte }],
  });
  return response.content[0].text;
}

// ─── Privacy headers ───────────────────────────────────────────────────────────

const app = express();

app.use((req, res, next) => {
  res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate');
  res.setHeader('Pragma', 'no-cache');
  res.setHeader('X-Data-Retention', 'none');
  res.setHeader('X-Image-Storage', 'memory-only-ephemeral');
  next();
});

// ─── Middlewares ───────────────────────────────────────────────────────────────

app.use(cors({
  origin: true,
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  credentials: true,
}));
app.use(express.json({ limit: '10mb' }));

app.use((req, _res, next) => {
  if (req.body && req.body.imageBase64) {
    req._imageSizeKb = Math.round((req.body.imageBase64.length * 3) / 4 / 1024);
  }
  next();
});

app.use(rateLimit({ windowMs: 60 * 1000, max: 30, standardHeaders: true }));
const aiLimiter = rateLimit({ windowMs: 60 * 1000, max: 10 });

// ─── JWT ───────────────────────────────────────────────────────────────────────

function verifierJWT(req, res, next) {
  if (!SUPABASE_JWT_SECRET) { req.userId = 'anonymous'; return next(); }
  const auth = req.headers.authorization;
  if (!auth?.startsWith('Bearer ')) return res.status(401).json({ error: 'Token manquant.' });
  try {
    req.userId = jwt.verify(auth.slice(7), SUPABASE_JWT_SECRET).sub;
    next();
  } catch {
    return res.status(401).json({ error: 'Token invalide.' });
  }
}

function validerBase64(str) {
  return str && typeof str === 'string' && (str.length * 3) / 4 / 1024 / 1024 < 5;
}
function validerTexte(str, max = 2000) {
  return typeof str === 'string' && str.trim().length > 0 && str.length <= max;
}
function nettoyerImage(req) {
  if (req.body) { req.body.imageBase64 = null; delete req.body.imageBase64; }
}

// ─── Routes ────────────────────────────────────────────────────────────────────

app.get('/health', (_req, res) => {
  res.json({
    status: 'ok',
    version: '1.0.0',
    timestamp: new Date().toISOString(),
    ai: {
      provider: AI_PROVIDER,
      configured: AI_PROVIDER === 'groq'   ? Boolean(GROQ_KEY)
                : AI_PROVIDER === 'gemini' ? Boolean(GEMINI_KEY)
                : Boolean(ANTHROPIC_KEY),
      illustrations: Boolean(together) ? 'Together AI FLUX (gratuit)' : 'non configuré',
    },
  });
});

app.get('/test-ia', async (_req, res) => {
  try {
    const texte = await genererHistoireIA('Réponds uniquement avec ce JSON: {"ok":true}');
    res.json({ status: 'ok', provider: AI_PROVIDER, response: texte.slice(0, 100) });
  } catch (err) {
    res.status(500).json({ status: 'error', provider: AI_PROVIDER, message: err.message });
  }
});


// ── POST /analyser-dessin ──────────────────────────────────────────────────────

app.post('/analyser-dessin', verifierJWT, aiLimiter, async (req, res) => {
  const { imageBase64 } = req.body;
  if (!validerBase64(imageBase64)) {
    nettoyerImage(req);
    return res.status(400).json({ error: 'Image manquante ou trop volumineuse.' });
  }
  console.log(`[analyser-dessin] user=${req.userId} size=${req._imageSizeKb}kb provider=${AI_PROVIDER}`);
  let result = null;
  try {
    const texte = await analyserDessinIA(imageBase64);
    nettoyerImage(req);
    const match = texte.match(/\{[\s\S]*\}/);
    if (!match) return res.status(502).json({ error: 'Réponse IA invalide.' });
    const data = JSON.parse(match[0]);
    result = {
      personnages:         Array.isArray(data.personnages) ? data.personnages : [],
      couleurs:            Array.isArray(data.couleurs)    ? data.couleurs    : [],
      decor:               data.decor               ?? '',
      ambiance:            data.ambiance             ?? '',
      descriptionComplete: data.descriptionComplete  ?? '',
    };
    res.json(result);
  } catch (err) {
    nettoyerImage(req);
    console.error(`[analyser-dessin] erreur user=${req.userId}:`, err.message);
    res.status(500).json({ error: 'Erreur lors de l\'analyse. ' + err.message });
  } finally {
    result = null;
  }
});

// ── POST /generer-histoire ─────────────────────────────────────────────────────

const GENRES = {
  aventure: 'Aventure / Adventure / مغامرة / Aventura — rythme rapide, quête courageuse, tension et action.',
  conte:    'Conte magique / Magical tale / حكاية سحرية / Cuento mágico — fées, sortilèges, poésie, merveilleux.',
  drole:    'Humour / Funny / مضحك / Gracioso — situations comiques, quiproquos, ton léger et amusant.',
  sf:       'Science-fiction / Sci-Fi / خيال علمي / Ciencia ficción — robots, espace, gadgets futuristes.',
  animaux:  'Animaux / Animals / حيوانات / Animales — animaux parlants, nature, amitié entre espèces.',
};

const AGES = {
  '3-4':  'très simple (phrases courtes de 5-8 mots, vocabulaire de maternelle, beaucoup de répétitions, rythme lent et rassurant)',
  '5-6':  'simple (phrases de 8-12 mots, vocabulaire courant, quelques mots nouveaux expliqués par le contexte)',
  '7-8':  'intermédiaire (phrases variées de 10-15 mots, vocabulaire enrichi, aventure avec péripéties)',
  '9-10': 'élaboré (phrases complexes, vocabulaire riche, intrigue avec rebondissements, personnages nuancés)',
};

const LANGUES = {
  fr: 'Écris TOUT en français. Titre en français.',
  en: 'Write EVERYTHING in English. Title in English.',
  ar: 'اكتب كل شيء باللغة العربية الفصحى البسيطة. العنوان بالعربية.',
  es: 'Escribe TODO en español. Título en español.',
};
const VARIANTES = [
  '', 'Défi INÉDIT.', 'Change le lieu et le problème.',
  'Noms originaux, début inattendu.', 'Résolution créative.', 'Rebondissement surprenant.',
];

app.post('/generer-histoire', verifierJWT, aiLimiter, async (req, res) => {
  const { analyse, prenomEnfant, age, genre = 'aventure', langue = 'fr', variante = 1 } = req.body;
  if (!validerTexte(prenomEnfant, 50)) return res.status(400).json({ error: 'Prénom invalide.' });
  if (!analyse) return res.status(400).json({ error: 'Analyse manquante.' });

  console.log(`[generer-histoire] user=${req.userId} prenom=${prenomEnfant} langue=${langue} provider=${AI_PROVIDER}`);

  const contexte = [
    `Personnages: ${(analyse.personnages || []).join(', ')}`,
    `Couleurs: ${(analyse.couleurs || []).join(', ')}`,
    `Décor: ${analyse.decor || ''}`,
    `Description: ${analyse.descriptionComplete || ''}`,
  ].join('\n');

  const contrainteVariante = VARIANTES[variante % VARIANTES.length]
    ? `\nVARIANTE #${variante}: ${VARIANTES[variante % VARIANTES.length]}` : '';

  const niveauAge = AGES[age] || AGES['5-6'];

  const prompt = `Tu es un auteur de livres illustrés pour enfants. Écris une histoire captivante (600-800 mots) pour ${prenomEnfant}, enfant de ${age} ans.

=== PARAMÈTRES OBLIGATOIRES ===
- HÉROS : ${prenomEnfant} (utilise son prénom à chaque paragraphe, il/elle est le personnage principal)
- ÂGE : ${age} ans → niveau de langage ${niveauAge}
- TYPE D'HISTOIRE : ${GENRES[genre] ?? GENRES.aventure}
- ${LANGUES[langue] ?? LANGUES.fr}

=== DESSIN DE L'ENFANT ===
${contexte}

=== STRUCTURE OBLIGATOIRE : 10 à 12 paragraphes ===
- Paragraphe 1-2 : Introduction du décor vivant et des personnages du dessin
- Paragraphe 3 : Moment magique où ${prenomEnfant} entre dans le dessin
- Paragraphe 4-5 : ${prenomEnfant} découvre le monde du dessin et rencontre les personnages
- Paragraphe 6-7 : Un problème ou défi se présente, tension dramatique
- Paragraphe 8-9 : ${prenomEnfant} affronte le défi avec courage et imagination
- Paragraphe 10-11 : Résolution heureuse, victoire de ${prenomEnfant}
- Paragraphe 12 : Retour et morale

Chaque paragraphe doit faire 4 à 6 phrases. L'histoire doit être riche, détaillée et captivante.

=== FORMAT JSON OBLIGATOIRE ===
Réponds UNIQUEMENT avec ce JSON:
{
  "titre": "titre poétique et accrocheur (max 8 mots)",
  "entree": "phrase magique décrivant comment ${prenomEnfant} entre dans le dessin",
  "paragraphes": [
    "paragraphe 1 long (4-6 phrases)...",
    "paragraphe 2 long (4-6 phrases)...",
    "paragraphe 3 long (4-6 phrases)...",
    "paragraphe 4 long (4-6 phrases)...",
    "paragraphe 5 long (4-6 phrases)...",
    "paragraphe 6 long (4-6 phrases)...",
    "paragraphe 7 long (4-6 phrases)...",
    "paragraphe 8 long (4-6 phrases)...",
    "paragraphe 9 long (4-6 phrases)...",
    "paragraphe 10 long (4-6 phrases)...",
    "paragraphe 11 long (4-6 phrases)...",
    "paragraphe 12 long (4-6 phrases)..."
  ],
  "morale": "une phrase de morale inspirante"
}${contrainteVariante}`;

  try {
    const texte = await genererHistoireIA(prompt);
    const match = texte.match(/\{[\s\S]*\}/);
    if (!match) return res.status(502).json({ error: 'Réponse IA invalide.' });
    const data = JSON.parse(match[0]);
    const paragraphes = Array.isArray(data.paragraphes)
      ? data.paragraphes.filter(p => typeof p === 'string' && p.trim()) : [];
    res.json({
      titre: data.titre ?? prenomEnfant, entree: data.entree ?? '',
      paragraphes, morale: data.morale ?? '',
      tempsLectureMinutes: Math.max(1, Math.round(paragraphes.join(' ').split(/\s+/).length / 100)),
      langue,
    });
  } catch (err) {
    console.error(`[generer-histoire] erreur user=${req.userId}:`, err.message);
    res.status(500).json({ error: 'Erreur lors de la génération. ' + err.message });
  }
});

// ── POST /traduire-histoire ────────────────────────────────────────────────────
// Corps  : { histoire, langueSource, langueCible, prenomEnfant }
// Retour : { histoire } — même structure, traduite

const NOMS_LANGUES = { fr: 'français', en: 'English', ar: 'Arabic', es: 'Spanish' };

app.post('/traduire-histoire', verifierJWT, aiLimiter, async (req, res) => {
  const { histoire, langueSource, langueCible, prenomEnfant } = req.body;

  if (!histoire || !langueCible || !prenomEnfant) {
    return res.status(400).json({ error: 'Paramètres manquants.' });
  }
  if (langueSource === langueCible) {
    return res.status(400).json({ error: 'Langue source et cible identiques.' });
  }

  const nomSource = NOMS_LANGUES[langueSource] ?? langueSource;
  const nomCible  = NOMS_LANGUES[langueCible]  ?? langueCible;
  const rtlCible  = langueCible === 'ar';

  const prompt = `You are a professional children's story translator.

Translate the following story FROM ${nomSource} TO ${nomCible}.

Rules:
- Keep the EXACT same narrative tone and style (magical, child-friendly)
- Keep the child's name "${prenomEnfant}" as-is (do not translate names)
- ${rtlCible ? 'Write in Arabic script, right-to-left.' : ''}
- Keep all emojis as-is
- Translate naturally — not word-for-word, but meaning-for-meaning
- Keep each paragraph roughly the same length

Original story in ${nomSource}:
titre: ${histoire.titre}
entree: ${histoire.entree}
paragraphes: ${JSON.stringify(histoire.paragraphes)}
morale: ${histoire.morale}

Reply ONLY with valid JSON (same structure):
{
  "titre": "translated title",
  "entree": "translated entree",
  "paragraphes": ["p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12"],
  "morale": "translated moral"
}`;

  console.log(`[traduire] ${langueSource} → ${langueCible} pour "${prenomEnfant}"`);

  try {
    const texte = await genererHistoireIA(prompt);
    const match = texte.match(/\{[\s\S]*\}/);
    if (!match) return res.status(502).json({ error: 'Réponse IA invalide.' });

    const data = JSON.parse(match[0]);
    const paragraphes = Array.isArray(data.paragraphes)
      ? data.paragraphes.filter(p => typeof p === 'string' && p.trim())
      : histoire.paragraphes;

    const histoireTraduite = {
      ...histoire,
      titre:      typeof data.titre   === 'string' ? data.titre   : histoire.titre,
      entree:     typeof data.entree  === 'string' ? data.entree  : histoire.entree,
      paragraphes,
      morale:     typeof data.morale  === 'string' ? data.morale  : histoire.morale,
      langue:     langueCible,
    };

    console.log(`[traduire] ✅ ${langueSource} → ${langueCible}`);
    return res.json({ histoire: histoireTraduite });

  } catch (err) {
    console.error('[traduire] erreur:', err.message);
    return res.status(500).json({ error: 'Erreur de traduction: ' + err.message });
  }
});

// ─── Démarrage ─────────────────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log(`✅ DoodleStory backend — port ${PORT}`);
  console.log(`   🤖 Provider : ${AI_PROVIDER.toUpperCase()}`);
  console.log(`   ${AI_PROVIDER === 'groq' ? (GROQ_KEY ? '✓ Groq prêt' : '✗ GROQ_API_KEY manquante') : ''}`);
  console.log(`   Supabase JWT : ${SUPABASE_JWT_SECRET ? '✓' : '⚠ absent'}`);
});
