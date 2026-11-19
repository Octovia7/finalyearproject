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
const buildIndex = (jobs) => {
  const docs = jobs.map((j) => skillsOf(j.skills)), df = new Map(), freq = new Map(), pairs = new Map();
  for (const doc of docs) {
    for (const s of doc) { df.set(s, (df.get(s) || 0) + 1); freq.set(s, (freq.get(s) || 0) + 1); }
    for (let i = 0; i < doc.length; i++) for (let j = i + 1; j < doc.length; j++) {
      const [a, b] = [doc[i], doc[j]].sort(), key = a + String.fromCharCode(0) + b;
      pairs.set(key, (pairs.get(key) || 0) + 1);
    }
  }
  const n = Math.max(docs.length, 1);
  const idf = new Map([...df].map(([s, f]) => [s, Math.log((n + 1) / (f + 1)) + 1]));
  const vectors = new Map();
  for (const [key, count] of pairs) {
    const [a, b] = key.split(String.fromCharCode(0));
    const pmi = Math.log(count * n / ((freq.get(a) || 1) * (freq.get(b) || 1)));
    if (pmi <= 0) continue;
    if (!vectors.has(a)) vectors.set(a, {});
    if (!vectors.has(b)) vectors.set(b, {});
    vectors.get(a)[b] = pmi; vectors.get(b)[a] = pmi;
  }
  return { idf, vectors };
};
const embedSet = (set, index) => {
  const out = {};
  for (const s of set) for (const [neighbor, value] of Object.entries(index.vectors.get(s) || {}))
    out[neighbor] = (out[neighbor] || 0) + (index.idf.get(s) || 1) * value;
  return out;
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
      const value = cosine(index.vectors.get(skill), index.vectors.get(candidateSkill));
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

