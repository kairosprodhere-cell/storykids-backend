require('dotenv').config();
const express    = require('express');
const cors       = require('cors');
const rateLimit  = require('express-rate-limit');
const Anthropic  = require('@anthropic-ai/sdk');
const jwt        = require('jsonwebtoken');

// ─── Configuration ────────────────────────────────────────────────────────────

const PORT             = process.env.PORT || 3000;
const ANTHROPIC_KEY    = process.env.ANTHROPIC_API_KEY;
const SUPABASE_JWT_SECRET = process.env.SUPABASE_JWT_SECRET;
const ALLOWED_ORIGINS  = (process.env.ALLOWED_ORIGINS || '*').split(',');

if (!ANTHROPIC_KEY) {
  console.error('❌ ANTHROPIC_API_KEY manquante dans .env');
  process.exit(1);
}

const anthropic = new Anthropic.default({ apiKey: ANTHROPIC_KEY });
const app = express();

// ─── Middlewares globaux ──────────────────────────────────────────────────────

app.use(cors({ origin: ALLOWED_ORIGINS }));
app.use(express.json({ limit: '10mb' }));   // images en base64

// Rate limiting global
app.use(rateLimit({
  windowMs: 60 * 1000,    // 1 minute
  max: 30,                 // 30 req/min par IP
  standardHeaders: true,
  message: { error: 'Trop de requêtes. Réessaie dans une minute.' },
}));

// Rate limiting strict pour les appels IA (coûteux)
const aiLimiter = rateLimit({
  windowMs: 60 * 1000,
  max: 5,                  // 5 générations/min par IP
  message: { error: 'Limite de génération atteinte. Réessaie dans une minute.' },
});

// ─── Middleware d'authentification ────────────────────────────────────────────

function verifierJWT(req, res, next) {
  // Si pas de secret Supabase configuré → on laisse passer (dev/demo)
  if (!SUPABASE_JWT_SECRET) {
    req.userId = 'anonymous';
    return next();
  }

  const auth = req.headers.authorization;
  if (!auth || !auth.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'Token manquant.' });
  }

  const token = auth.slice(7);
  try {
    const payload = jwt.verify(token, SUPABASE_JWT_SECRET);
    req.userId = payload.sub;
    next();
  } catch (err) {
    return res.status(401).json({ error: 'Token invalide ou expiré.' });
  }
}

// ─── Utilitaires de validation ────────────────────────────────────────────────

function validerBase64(str) {
  if (!str || typeof str !== 'string') return false;
  const size = (str.length * 3) / 4 / 1024 / 1024;
  return size < 5; // max 5 Mo
}

function validerTexte(str, max = 2000) {
  return typeof str === 'string' && str.trim().length > 0 && str.length <= max;
}

// ─── Route de santé ───────────────────────────────────────────────────────────

app.get('/health', (_req, res) => {
  res.json({ status: 'ok', version: '1.0.0', timestamp: new Date().toISOString() });
});

// ─── POST /analyser-dessin ────────────────────────────────────────────────────
// Corps : { imageBase64: string }
// Retour : { personnages, couleurs, decor, ambiance, descriptionComplete }

app.post('/analyser-dessin', verifierJWT, aiLimiter, async (req, res) => {
  const { imageBase64 } = req.body;

  if (!validerBase64(imageBase64)) {
    return res.status(400).json({ error: 'Image manquante ou trop volumineuse (max 5 Mo).' });
  }

  try {
    const response = await anthropic.messages.create({
      model: 'claude-opus-4-7',
      max_tokens: 600,
      messages: [{
        role: 'user',
        content: [
          {
            type: 'image',
            source: { type: 'base64', media_type: 'image/jpeg', data: imageBase64 },
          },
          {
            type: 'text',
            text: `C'est le dessin fait par un enfant. Analyse-le et réponds UNIQUEMENT en JSON valide :

{
  "personnages": ["personnage 1", "personnage 2"],
  "couleurs": ["couleur 1", "couleur 2", "couleur 3"],
  "decor": "description du décor en une phrase",
  "ambiance": "ambiance générale en 2-3 mots",
  "descriptionComplete": "description complète et imaginative en 2-3 phrases"
}

- personnages : êtres vivants ou objets principaux (max 5)
- couleurs : 3-5 couleurs dominantes
- Réponds UNIQUEMENT avec le JSON, rien d'autre`,
          },
        ],
      }],
    });

    const texte = response.content[0].text;
    const match = texte.match(/\{[\s\S]*\}/);
    if (!match) return res.status(502).json({ error: 'Réponse IA invalide.' });

    const data = JSON.parse(match[0]);
    res.json({
      personnages:         Array.isArray(data.personnages)        ? data.personnages        : [],
      couleurs:            Array.isArray(data.couleurs)           ? data.couleurs           : [],
      decor:               typeof data.decor === 'string'         ? data.decor              : '',
      ambiance:            typeof data.ambiance === 'string'      ? data.ambiance           : '',
      descriptionComplete: typeof data.descriptionComplete === 'string' ? data.descriptionComplete : '',
    });
  } catch (err) {
    console.error('[analyser-dessin]', err.message);
    res.status(500).json({ error: 'Erreur lors de l\'analyse du dessin.' });
  }
});

// ─── POST /generer-histoire ───────────────────────────────────────────────────
// Corps : { analyse, prenomEnfant, age, genre, langue, variante? }
// Retour : { titre, entree, paragraphes, morale, tempsLectureMinutes, langue }

const INSTRUCTIONS_GENRE = {
  aventure: 'STORY STYLE — ADVENTURE: Fast pace, courageous challenges, quest with obstacles. Hero shows bravery. Action verbs. Suspense in paragraph 3. Earned victory.',
  conte:    'STORY STYLE — MAGICAL TALE: Fairy-tale atmosphere. Magic spells, benevolent creatures, stardust. Poetic and gentle sentences. Good always triumphs.',
  drole:    'STORY STYLE — FUNNY: Comic situations and unexpected twists. Silly misunderstandings, clumsy characters, funny punchlines. Child-friendly humour, onomatopoeia, witty lines.',
  sf:       'STORY STYLE — SCIENCE-FICTION: Futuristic gadgets, robots, space or time travel. Invent technology names. Simplified scientific vocabulary for the child\'s age.',
  animaux:  'STORY STYLE — ANIMALS: Drawn characters are talking expressive animals with distinct personalities. Natural setting. Each animal has a unique trait. Life lesson about cooperation.',
};

const INSTRUCTIONS_LANGUE = {
  fr: 'LANGUE : Français. Écris TOUT en français.',
  en: 'LANGUAGE: English. Write EVERYTHING in English.',
  ar: 'اللغة: العربية. اكتب كل شيء باللغة العربية.',
  es: 'IDIOMA: Español. Escribe TODO en español.',
};

const VARIANTES = [
  '',
  'Le défi principal doit être INÉDIT, différent de toute version précédente.',
  'Change complètement le lieu de l\'aventure et le problème à résoudre.',
  'Donne des noms originaux aux personnages. Histoire qui commence de façon inattendue.',
  'Le héros résout le problème par une approche créative et originale.',
  'Ajoute un rebondissement surprenant au milieu de l\'histoire.',
];

app.post('/generer-histoire', verifierJWT, aiLimiter, async (req, res) => {
  const { analyse, prenomEnfant, age, genre = 'aventure', langue = 'fr', variante = 1 } = req.body;

  if (!validerTexte(prenomEnfant, 50)) {
    return res.status(400).json({ error: 'Prénom invalide.' });
  }
  if (!analyse || typeof analyse !== 'object') {
    return res.status(400).json({ error: 'Analyse du dessin manquante.' });
  }

  const toneInstruction   = INSTRUCTIONS_GENRE[genre]  ?? INSTRUCTIONS_GENRE.aventure;
  const langueInstruction = INSTRUCTIONS_LANGUE[langue] ?? INSTRUCTIONS_LANGUE.fr;
  const contrainteVariante = VARIANTES[variante % VARIANTES.length]
    ? `\nVARIANTE #${variante} — CONTRAINTE : ${VARIANTES[variante % VARIANTES.length]}`
    : '';

  const contexte = [
    `Personnages dessinés : ${(analyse.personnages || []).join(', ')}`,
    `Couleurs utilisées : ${(analyse.couleurs || []).join(', ')}`,
    `Décor : ${analyse.decor || ''}`,
    `Ambiance : ${analyse.ambiance || ''}`,
    `Description : ${analyse.descriptionComplete || ''}`,
  ].join('\n');

  try {
    const response = await anthropic.messages.create({
      model: 'claude-opus-4-7',
      max_tokens: 1600,
      messages: [{
        role: 'user',
        content: `You are a magical children's book author. A ${age}-year-old child made a drawing. Tell the story where the child enters their own drawing.

=== WHAT THE CHILD DREW ===
${contexte}

=== THE CHILD HERO ===
Name: ${prenomEnfant}
Age: ${age}

=== STORY STYLE ===
${toneInstruction}

=== ${langueInstruction} ===

=== MANDATORY NARRATIVE ARC ===
1. SETTING: Bring the drawing to life
2. MEETING: The drawn characters welcome ${prenomEnfant}
3. ADVENTURE: ${prenomEnfant} faces an obstacle
4. CLIMAX: ${prenomEnfant} solves the challenge
5. END: ${prenomEnfant} leaves with a magical memory

=== RESPONSE FORMAT ===
Reply ONLY with valid JSON:
{
  "titre": "poetic title (max 8 words)",
  "entree": "unique magical sentence how ${prenomEnfant} enters the drawing",
  "paragraphes": ["p1 (2-3 sentences)","p2","p3","p4","p5"],
  "morale": "short moral"
}

Rules:
- ALL text in the language specified
- ${prenomEnfant} appears in at least 4 paragraphs
- Each paragraph mentions a drawing element
- Reply ONLY with the JSON${contrainteVariante}`,
      }],
    });

    const texte = response.content[0].text;
    const match = texte.match(/\{[\s\S]*\}/);
    if (!match) return res.status(502).json({ error: 'Réponse IA invalide.' });

    const data = JSON.parse(match[0]);
    const paragraphes = Array.isArray(data.paragraphes)
      ? data.paragraphes.filter((p) => typeof p === 'string' && p.trim())
      : [];
    const totalMots = paragraphes.join(' ').split(/\s+/).length;

    res.json({
      titre:               typeof data.titre === 'string'  ? data.titre   : prenomEnfant,
      entree:              typeof data.entree === 'string' ? data.entree  : '',
      paragraphes,
      morale:              typeof data.morale === 'string' ? data.morale  : '',
      tempsLectureMinutes: Math.max(1, Math.round(totalMots / 100)),
      langue,
    });
  } catch (err) {
    console.error('[generer-histoire]', err.message);
    res.status(500).json({ error: 'Erreur lors de la génération de l\'histoire.' });
  }
});

// ─── Démarrage ────────────────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log(`✅ StoryKids backend démarré sur le port ${PORT}`);
  console.log(`   Claude API : ${ANTHROPIC_KEY ? '✓ configurée' : '✗ manquante'}`);
  console.log(`   Supabase JWT : ${SUPABASE_JWT_SECRET ? '✓ configuré' : '⚠ absent (pas de vérification JWT)'}`);
});
