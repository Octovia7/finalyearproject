const normalize = (value) => String(value || '').toLowerCase()
  .replace(/[_-]+/g, ' ').replace(/[^\p{L}\p{N}+ #./]/gu, ' ')
  .replace(/\s+/g, ' ').trim();
const skillsOf = (values) => [...new Set((Array.isArray(values) ? values : [values]).map(normalize).filter(Boolean))];
const cosine = (a = {}, b = {}) => {
  let dot = 0, na = 0, nb = 0;
  for (const [k, v] of Object.entries(a)) { na += v * v; if (b[k]) dot += v * b[k]; }
  for (const v of Object.values(b)) nb += v * v;
  return na && nb ? dot / Math.sqrt(na * nb) : 0;
};
// Train a small skip-gram Word2Vec model with negative sampling (SGNS) over
// job skill lists. Each posting is a sentence and its skills are the tokens.
// Training is deterministic and cached by the posting-corpus signature below.
const buildIndex = (jobs) => {
  // Keep request-time training bounded; use a stable slice from the ordered job query.
  const docs = jobs.map((j) => skillsOf(j.skills).slice(0, 30)).filter((doc) => doc.length).slice(0, 300);
  const df = new Map();
  for (const doc of docs) for (const skill of new Set(doc)) df.set(skill, (df.get(skill) || 0) + 1);
  const idf = new Map([...df].map(([skill, count]) => [skill, Math.log((docs.length + 1) / (count + 1)) + 1]));
  const vocab = [...df.keys()].sort();
  const vectors = new Map();
  if (vocab.length < 2) return { idf, vectors };

  const dimension = 32, window = 3, epochs = 10, negativeSamples = 3;
  const wordIndex = new Map(vocab.map((word, i) => [word, i]));
  const counts = vocab.map((word) => df.get(word));
  const poweredCounts = counts.map((count) => count ** 0.75);
  const countTotal = poweredCounts.reduce((sum, count) => sum + count, 0);
  const cumulative = [];
  poweredCounts.reduce((sum, count, i) => (cumulative[i] = sum + count / countTotal), 0);

  // Seed the pseudo-random generator from the normalized corpus for repeatable
  // training across requests and server restarts.
  let seed = 2166136261;
  for (const doc of docs) for (const word of doc) {
    for (let i = 0; i < word.length; i++) seed = Math.imul(seed ^ word.charCodeAt(i), 16777619);
  }
  const random = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  const input = vocab.map(() => Float64Array.from({ length: dimension }, () => (random() - 0.5) / dimension));
  const output = vocab.map(() => new Float64Array(dimension));
  const pairs = [];
  for (const doc of docs) {
    const tokens = doc.map((word) => wordIndex.get(word)).filter((index) => index !== undefined);
    for (let i = 0; i < tokens.length; i++) {
      for (let j = Math.max(0, i - window); j <= Math.min(tokens.length - 1, i + window); j++) {
        if (i !== j) pairs.push([tokens[i], tokens[j]]);
      }
    }
  }
  const sigmoid = (value) => value > 8 ? 0.9997 : value < -8 ? 0.0003 : 1 / (1 + Math.exp(-value));
  const trainPair = (target, context, label, rate) => {
    let dot = 0;
    for (let k = 0; k < dimension; k++) dot += input[target][k] * output[context][k];
    const gradient = (label - sigmoid(dot)) * rate;
    for (let k = 0; k < dimension; k++) {
      const oldInput = input[target][k];
      input[target][k] += gradient * output[context][k];
      output[context][k] += gradient * oldInput;
    }
  };
  for (let epoch = 0; epoch < epochs; epoch++) {
    const rate = Math.max(0.0025, 0.025 * (1 - epoch / epochs));
    for (const [target, context] of pairs) {
      trainPair(target, context, 1, rate);
      for (let n = 0; n < negativeSamples; n++) {
        const sample = random();
        let lo = 0, hi = cumulative.length - 1;
        while (lo < hi) { const mid = (lo + hi) >> 1; if (sample <= cumulative[mid]) hi = mid; else lo = mid + 1; }
        if (lo !== context) trainPair(target, lo, 0, rate);
      }
    }
  }
  vocab.forEach((word, i) => vectors.set(word, Array.from(input[i])));
  return { idf, vectors };
};
const cosineVectors = (a = [], b = []) => {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i++) { dot += a[i] * b[i]; na += a[i] ** 2; nb += b[i] ** 2; }
  return na && nb ? dot / Math.sqrt(na * nb) : 0;
};
const embedSet = (set, index) => {
  const out = new Array(32).fill(0);
  let totalWeight = 0;
  for (const skill of set) {
    const vector = index.vectors.get(skill), weight = index.idf.get(skill) || 1;
    if (!vector) continue;
    totalWeight += weight;
    for (let i = 0; i < vector.length; i++) out[i] += vector[i] * weight;
  }
  return totalWeight ? out.map((value) => value / totalWeight) : [];
};
const scoreJob = (profile, prefs, job, index) => {
  const cs = skillsOf(profile.skills), js = skillsOf(job.skills), cset = new Set(cs), jset = new Set(js);
  const matched = cs.filter((s) => jset.has(s));
  const cv = Object.fromEntries(cs.map((s) => [s, index.idf.get(s) || 1]));
  const jv = Object.fromEntries(js.map((s) => [s, index.idf.get(s) || 1]));
  const tfidf = cosine(cv, jv), embedding = cosine(embedSet(cs, index), embedSet(js, index));
  const content = .45 * tfidf + .55 * embedding;
  const roles = skillsOf([profile.primaryRole, ...(profile.openToRoles || [])]);
  const jobRoles = skillsOf([job.title, job.primaryRole, ...(job.additionalRoles || [])]);
  const roleFit = jobRoles.some((r) => roles.includes(r));
  const location = normalize(prefs.workLocation || profile.location);
  const locations = skillsOf(job.location || []);
  const locationFit = !!location && (locations.includes(location) ||
    (!!prefs.openToRemote && /remote/i.test(job.remotePolicy || '')));
  const target = Number(prefs.desiredSalary), max = Number(job.salary && job.salary.max);
  const currencyFit = !prefs.salaryCurrency || !job.currency || prefs.salaryCurrency.toLowerCase() === job.currency.toLowerCase();
  const salaryFit = target > 0 && Number.isFinite(target) && Number.isFinite(max) && max >= target && currencyFit;
  const context = (roleFit ? .5 : 0) + (locationFit ? .3 : 0) + (salaryFit ? .2 : 0);
  const score = .8 * content + .2 * context;
  const relatedSkills = [];
  for (const skill of js) {
    if (cset.has(skill)) continue;
    let best = null, similarity = 0;
    for (const candidateSkill of cs) {
      const value = cosineVectors(index.vectors.get(skill), index.vectors.get(candidateSkill));
      if (value > similarity) { best = candidateSkill; similarity = value; }
    }
    if (best && similarity >= .6) relatedSkills.push({ jobSkill: skill, candidateSkill: best, similarity: Number(similarity.toFixed(3)) });
  }
  const reasons = [];
  if (matched.length) reasons.push('Matched skills: ' + matched.join(', ') + '.');
  for (const x of relatedSkills) reasons.push(x.jobSkill + ' is related to your ' + x.candidateSkill + ' skill.');
  if (roleFit) reasons.push('The role matches your profile or job preferences.');
  if (locationFit) reasons.push('The location or remote policy matches your preferences.');
  if (salaryFit) reasons.push('The listed maximum salary meets your target.');
  reasons.push(job.visaSponsorship ? 'This employer indicates visa sponsorship is available.' : 'This employer does not indicate visa sponsorship.');
  return { score, breakdown: { tfidf: +tfidf.toFixed(4), embedding: +embedding.toFixed(4), content: +content.toFixed(4), context: +context.toFixed(4), final: +score.toFixed(4) }, matchedSkills: matched, relatedSkills, reasons };
};
let cachedSignature = null;
let cachedSkillIndex = null;
const getSkillIndex = (jobs) => {
  const hash = require('crypto').createHash('sha1');
  for (const job of jobs) {
    hash.update(String(job._id || ''));
    hash.update(String.fromCharCode(0));
    hash.update(skillsOf(job.skills).sort().join(String.fromCharCode(0)));
    hash.update(String.fromCharCode(1));
  }
  const signature = hash.digest('hex');
  if (signature !== cachedSignature) {
    cachedSkillIndex = buildIndex(jobs);
    cachedSignature = signature;
  }
  return cachedSkillIndex;
};

const rankJobs = (profile, prefs, jobs, limit = 10) => {
  const index = getSkillIndex(jobs);
  return jobs.map((job) => ({ ...job, recommendation: scoreJob(profile, prefs, job, index) }))
    .sort((a, b) => b.recommendation.score - a.recommendation.score ||
      new Date(b.postedAt || 0) - new Date(a.postedAt || 0))
    .slice(0, limit);
};
module.exports = { rankJobs, normalizeSkill: normalize };

