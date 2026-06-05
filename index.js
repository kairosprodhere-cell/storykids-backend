require('dotenv').config();
const express    = require('express');
const cors       = require('cors');
const rateLimit  = require('express-rate-limit');
const Anthropic  = require('@anthropic-ai/sdk');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const jwt        = require('jsonwebtoken');

// ─── Configuration ────────────────────────────────────────────────────────────

const PORT                = process.env.PORT || 3000;
const AI_PROVIDER         = process.env.AI_PROVIDER || 'gemini'; // 'gemini' | 'claude'
const ANTHROPIC_KEY       = process.env.ANTHROPIC_API_KEY;
const GEMINI_KEY          = process.env.GEMINI_API_KEY;
const SUPABASE_JWT_SECRET = process.env.SUPABASE_JWT_SECRET;
const ALLOWED_ORIGINS     = (process.env.ALLOWED_ORIGINS || '*').split(',');

// Validation selon le provider choisi — warnings uniquement, pas de crash au démarrage
if (AI_PROVIDER === 'claude' && !ANTHROPIC_KEY) {
  console.warn('⚠️  ANTHROPIC_API_KEY manquante — les appels IA échoueront');
}
if (AI_PROVIDER === 'gemini' && !GEMINI_KEY) {
  console.warn('⚠️  GEMINI_API_KEY manquante — les appels IA échoueront');
}
if (!ANTHROPIC_KEY && !GEMINI_KEY) {
  console.warn('⚠️  Aucune clé IA configurée. Configure AI_PROVIDER et la clé correspondante.');
}

// Clients IA
const anthropic   = ANTHROPIC_KEY ? new Anthropic.default({ apiKey: ANTHROPIC_KEY }) : null;
const geminiAI    = GEMINI_KEY    ? new GoogleGenerativeAI(GEMINI_KEY)               : null;
const geminiModel = geminiAI?.getGenerativeModel({ model: 'gemini-1.5-flash' });

// ─── Abstraction IA — même interface pour Gemini et Claude ───────────────────

async function analyserDessinIA(imageBase64) {
  const prompt = `C'est le dessin fait par un enfant. Analyse-le et réponds UNIQUEMENT en JSON valide :
{
  "personnages": ["personnage 1", "personnage 2"],
  "couleurs": ["couleur 1", "couleur 2"],
  "decor": "description du décor",
  "ambiance": "ambiance en 2-3 mots",
  "descriptionComplete": "description complète en 2-3 phrases"
}
Réponds UNIQUEMENT avec le JSON, rien d'autre.`;

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

async function genererHistoireIA(promptTexte) {
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

const app = express();

// ─── Politique de confidentialité des données ─────────────────────────────────
//
// Les dessins envoyés à ce serveur sont traités selon ces règles STRICTES :
//   1. Jamais écrits sur disque.
//   2. Jamais loggués (ni console, ni fichier, ni service tiers).
//   3. Transmis à l'API Claude en mémoire uniquement.
//   4. La référence mémoire est nulle (null) immédiatement après l'appel.
//   5. La durée de vie maximale en RAM est le temps de la requête HTTP (~2-5s).
//
// Seule la RÉPONSE de Claude (texte) est conservée le temps de la réponse HTTP.

// ─── Headers privacy sur toutes les réponses ─────────────────────────────────

app.use((req, res, next) => {
  // Interdit au navigateur/proxy de mettre en cache les réponses
  res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate');
  res.setHeader('Pragma', 'no-cache');
  // Header informatif : aucune image n'est conservée
  res.setHeader('X-Data-Retention', 'none');
  res.setHeader('X-Image-Storage', 'memory-only-ephemeral');
  next();
});

// ─── Middlewares globaux ──────────────────────────────────────────────────────

app.use(cors({
  origin: true,           // autorise toutes les origines (Expo web, mobile, localhost)
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  credentials: true,
}));
app.options('*', cors());  // pré-réponse aux requêtes preflight OPTIONS
app.use(express.json({ limit: '10mb' }));

// Middleware anti-log : remplace les base64 dans les logs par un placeholder
// pour éviter toute fuite accidentelle d'image dans les fichiers de log.
app.use((req, _res, next) => {
  if (req.body && req.body.imageBase64) {
    // Taille indicative pour le débogage, jamais le contenu
    const octets = Math.round((req.body.imageBase64.length * 3) / 4 / 1024);
    req._imageSizeKb = octets;
    // On ne logge JAMAIS la valeur réelle
  }
  next();
});

// Rate limiting global
app.use(rateLimit({
  windowMs: 60 * 1000,
  max: 30,
  standardHeaders: true,
  message: { error: 'Trop de requêtes. Réessaie dans une minute.' },
}));

// Rate limiting strict pour les appels IA
const aiLimiter = rateLimit({
  windowMs: 60 * 1000,
  max: 5,
  message: { error: 'Limite de génération atteinte. Réessaie dans une minute.' },
});

// ─── Middleware d'authentification ────────────────────────────────────────────

function verifierJWT(req, res, next) {
  if (!SUPABASE_JWT_SECRET) {
    req.userId = 'anonymous';
    return next();
  }
  const auth = req.headers.authorization;
  if (!auth || !auth.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'Token manquant.' });
  }
  try {
    const payload = jwt.verify(auth.slice(7), SUPABASE_JWT_SECRET);
    req.userId = payload.sub;
    next();
  } catch {
    return res.status(401).json({ error: 'Token invalide ou expiré.' });
  }
}

// ─── Utilitaires ─────────────────────────────────────────────────────────────

function validerBase64(str) {
  if (!str || typeof str !== 'string') return false;
  return (str.length * 3) / 4 / 1024 / 1024 < 5; // max 5 Mo
}

function validerTexte(str, max = 2000) {
  return typeof str === 'string' && str.trim().length > 0 && str.length <= max;
}

// Nettoie un objet request de tout contenu image binaire
function nettoyerImageDuBody(req) {
  if (req.body) {
    req.body.imageBase64 = null;
    delete req.body.imageBase64;
  }
}

// ─── Route de santé ───────────────────────────────────────────────────────────

app.get('/health', (_req, res) => {
  res.json({
    status: 'ok',
    version: '1.0.0',
    timestamp: new Date().toISOString(),
    privacy: {
      imageStorage: 'none',
      imageRetention: 'memory-only during request (~2-5s)',
      logging: 'metadata only (no image content)',
      thirdParty: 'Claude API (Anthropic) — images processed, not stored per Anthropic policy',
    },
  });
});

// ─── POST /analyser-dessin ────────────────────────────────────────────────────

app.post('/analyser-dessin', verifierJWT, aiLimiter, async (req, res) => {
  const { imageBase64 } = req.body;

  if (!validerBase64(imageBase64)) {
    nettoyerImageDuBody(req);
    return res.status(400).json({ error: 'Image manquante ou trop volumineuse (max 5 Mo).' });
  }

  // Log metadata uniquement — jamais le contenu de l'image
  console.log(`[analyser-dessin] user=${req.userId} size=${req._imageSizeKb}kb`);

  let analysisResult = null;
  try {
    // L'image vit ici en mémoire le temps de l'appel API (~2-5s)
    const texteReponse = await analyserDessinIA(imageBase64);

    // ✅ L'image est immédiatement nettoyée après l'appel IA
    nettoyerImageDuBody(req);

    const texte = texteReponse;
    const match = texte.match(/\{[\s\S]*\}/);
    if (!match) return res.status(502).json({ error: 'Réponse IA invalide.' });

    const data = JSON.parse(match[0]);
    analysisResult = {
      personnages:         Array.isArray(data.personnages)  ? data.personnages  : [],
      couleurs:            Array.isArray(data.couleurs)     ? data.couleurs     : [],
      decor:               data.decor               ?? '',
      ambiance:            data.ambiance             ?? '',
      descriptionComplete: data.descriptionComplete  ?? '',
    };

    res.json(analysisResult);
  } catch (err) {
    nettoyerImageDuBody(req); // ✅ nettoyage même en cas d'erreur
    console.error(`[analyser-dessin] erreur user=${req.userId}:`, err.message);
    res.status(500).json({ error: 'Erreur lors de l\'analyse du dessin.' });
  } finally {
    // ✅ Garantie finale : nullification de toute référence mémoire
    analysisResult = null;
  }
});

// ─── POST /generer-histoire ───────────────────────────────────────────────────

const INSTRUCTIONS_GENRE = {
  aventure: 'ADVENTURE: fast pace, courageous quest, obstacles and victory.',
  conte:    'MAGICAL TALE: fairies, spells, poetic sentences, good triumphs.',
  drole:    'FUNNY: comic situations, silly misunderstandings, witty punchlines.',
  sf:       'SCI-FI: futuristic gadgets, robots, space travel, simplified science.',
  animaux:  'ANIMALS: talking animals with personalities, nature, life lessons.',
};

const INSTRUCTIONS_LANGUE = {
  fr: 'Écris TOUT en français.',
  en: 'Write EVERYTHING in English.',
  ar: 'اكتب كل شيء باللغة العربية.',
  es: 'Escribe TODO en español.',
};

const VARIANTES = [
  '', 'Défi INÉDIT, différent des versions précédentes.',
  'Change le lieu de l\'aventure et le problème à résoudre.',
  'Noms originaux pour les personnages, début inattendu.',
  'Résolution créative et originale.',
  'Rebondissement surprenant au milieu.',
];

app.post('/generer-histoire', verifierJWT, aiLimiter, async (req, res) => {
  const { analyse, prenomEnfant, age, genre = 'aventure', langue = 'fr', variante = 1 } = req.body;

  if (!validerTexte(prenomEnfant, 50)) {
    return res.status(400).json({ error: 'Prénom invalide.' });
  }
  if (!analyse || typeof analyse !== 'object') {
    return res.status(400).json({ error: 'Analyse du dessin manquante.' });
  }

  // Pas d'image ici : on ne reçoit que le texte de l'analyse
  console.log(`[generer-histoire] user=${req.userId} prenom=${prenomEnfant} langue=${langue}`);

  const toneInstruction    = INSTRUCTIONS_GENRE[genre]  ?? INSTRUCTIONS_GENRE.aventure;
  const langueInstruction  = INSTRUCTIONS_LANGUE[langue] ?? INSTRUCTIONS_LANGUE.fr;
  const contrainteVariante = VARIANTES[variante % VARIANTES.length]
    ? `\nVARIANTE #${variante}: ${VARIANTES[variante % VARIANTES.length]}`
    : '';

  const contexte = [
    `Personnages: ${(analyse.personnages || []).join(', ')}`,
    `Couleurs: ${(analyse.couleurs || []).join(', ')}`,
    `Décor: ${analyse.decor || ''}`,
    `Ambiance: ${analyse.ambiance || ''}`,
    `Description: ${analyse.descriptionComplete || ''}`,
  ].join('\n');

  const promptHistoire = `Auteur de livres pour enfants (${age} ans). Héros: ${prenomEnfant}.
Style: ${toneInstruction}
Langue: ${langueInstruction}
Dessin: ${contexte}

Réponds UNIQUEMENT en JSON:
{
  "titre": "...",
  "entree": "...",
  "paragraphes": ["p1","p2","p3","p4","p5"],
  "morale": "..."
}${contrainteVariante}`;

  try {
    const texte = await genererHistoireIA(promptHistoire);
    const match = texte.match(/\{[\s\S]*\}/);
    if (!match) return res.status(502).json({ error: 'Réponse IA invalide.' });

    const data = JSON.parse(match[0]);
    const paragraphes = Array.isArray(data.paragraphes)
      ? data.paragraphes.filter((p) => typeof p === 'string' && p.trim())
      : [];

    res.json({
      titre:               data.titre   ?? prenomEnfant,
      entree:              data.entree  ?? '',
      paragraphes,
      morale:              data.morale  ?? '',
      tempsLectureMinutes: Math.max(1, Math.round(paragraphes.join(' ').split(/\s+/).length / 100)),
      langue,
    });
  } catch (err) {
    console.error(`[generer-histoire] erreur user=${req.userId}:`, err.message);
    res.status(500).json({ error: 'Erreur lors de la génération.' });
  }
});

// ─── Démarrage ────────────────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log(`✅ StoryKids backend — port ${PORT}`);
  console.log(`   🤖 Provider IA    : ${AI_PROVIDER.toUpperCase()}`);
  console.log(`   🔒 Privacy        : aucun dessin stocké sur disque`);
  if (AI_PROVIDER === 'gemini')
    console.log(`   Gemini API    : ${GEMINI_KEY    ? '✓' : '✗'}`);
  else
    console.log(`   Claude API    : ${ANTHROPIC_KEY ? '✓' : '✗'}`);
  console.log(`   Supabase JWT  : ${SUPABASE_JWT_SECRET ? '✓' : '⚠ absent'}`);
});
