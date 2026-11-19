const Job = require('../models/Job');
const Profile = require('../models/Profile');
const JobPreferences = require('../models/JobPreferences');
const { rankJobs } = require('../utils/aiRecommender');

const getRecommendedJobs = async (req, res) => {
  try {
    const userId = req.user.id;
    const profile = await Profile.findOne({ userId }).lean();
    if (!profile) return res.status(404).json({ message: 'Profile not found' });
    const preferences = await JobPreferences.findOne({ userId }).lean() || {};
    const jobs = await Job.find({
      active: true, isDraft: { $ne: true }, isHiring: { $ne: false },
      ...(preferences.needsSponsorship ? { visaSponsorship: true, autoSkipVisaCandidates: { $ne: true } } : {}),
    }).lean();
    const requestedLimit = Number.parseInt(req.query.limit, 10);
    const limit = Number.isFinite(requestedLimit) ? Math.min(Math.max(requestedLimit, 1), 50) : 10;
    const recommendedJobs = rankJobs(profile, preferences, jobs, limit);
    return res.json({ project: 'SmartHire', count: recommendedJobs.length, recommendedJobs });
  } catch (err) {
    console.error('AI recommended jobs error:', err);
    return res.status(500).json({ message: 'Unable to generate job recommendations' });
  }
};

const browseJobs = async (req, res) => {
  try {
    const { remote, jobType, location, page = 1, limit = 10 } = req.query;
    const filters = { active: true };
    if (remote) filters.remotePolicy = new RegExp(remote, 'i');
    if (jobType) filters.jobType = jobType;
    if (location) filters.location = { $regex: location, $options: 'i' };
    const jobs = await Job.find(filters).sort({ postedAt: -1 }).skip((page - 1) * limit).limit(parseInt(limit, 10));
    const total = await Job.countDocuments(filters);
    return res.json({ jobs, total, page: parseInt(page, 10), pages: Math.ceil(total / limit) });
  } catch (err) {
    console.error('Browse jobs error:', err);
    return res.status(500).json({ message: 'Server error' });
  }
};

const getAllJobs = async (req, res) => {
  try {
    const jobs = await Job.find({ active: true }).sort({ postedAt: -1 });
    return res.json(jobs);
  } catch (err) {
    console.error('Get all jobs error:', err);
    return res.status(500).json({ message: 'Server error' });
  }
};
module.exports = { browseJobs, getAllJobs, getRecommendedJobs };
