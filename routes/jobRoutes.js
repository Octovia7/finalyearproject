const express = require('express');
const router = express.Router();
const { browseJobs, getAllJobs, getRecommendedJobs } = require('../controllers/jobController');
const { authenticate } = require('../middleware/authMiddleware');

router.get('/browse', browseJobs);
router.get('/all', getAllJobs);
router.get('/recommended', authenticate, getRecommendedJobs);
router.get('/ai-recommended', authenticate, getRecommendedJobs);

module.exports = router;
