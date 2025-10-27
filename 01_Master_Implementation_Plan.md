🎯 MASTER IMPLEMENTATION PROMPT
AI-Powered Spam & Duplicate Grievance Detection System
OBJECTIVE
Build a robust, multilingual text similarity and classification system to automatically detect spam and duplicate grievances in a civic complaint management platform, reducing manual review workload by 60% while maintaining <1% false rejection rate.
PHASE 1: DATA COLLECTION & PREPARATION
Data Requirements:
Minimum Dataset Size: 10,000 labeled grievances
Distribution:Legitimate: 60% (6,000 samples)
Duplicates: 25% (2,500 samples)
Spam: 15% (1,500 samples)
Data Sources:
Historical Data: Extract from existing grievance database
Synthetic Generation: Use GPT-4/Claude to generate realistic samples
Crowdsourcing: Hire annotators for real-world diversity
Public Datasets: Scrape similar domains (customer complaints, forum posts)
Annotation Guidelines:
For each grievance, annotate:
- Primary Label: [legitimate, spam, duplicate]
- Spam Type: [null, irrelevant, abusive, promotional, test, gibberish, bot]
- Duplicate Info: [original_id, similarity_level]
- Category: [water, electricity, sanitation, roads, tax, healthcare, education, police, corruption, other]
- Language: [english, hindi, hinglish, marathi, tamil, bengali, etc.]
- Quality Flags: [has_location, has_details, actionable, urgent]
- Metadata: timestamps, user_id, resolution_status

Quality Checks:
- Inter-annotator agreement: Cohen's Kappa > 0.75
- Ambiguous cases: Triple annotation + majority vote
- Regular calibration sessions with annotators
PHASE 2: FEATURE ENGINEERING
Text Features:
# Embeddings
- Sentence-BERT embeddings (768-dim)
- IndicBERT for Indian languages
- FastText for word-level features
- TF-IDF for keyword extraction

# Linguistic Features
- Sentence count, avg sentence length
- Unique word ratio (vocabulary richness)
- Stopword ratio
- POS tag distribution
- Named entity counts (PERSON, LOC, ORG)
- Readability scores (Flesch-Kincaid)
Metadata Features:
# User Behavior
- total_submissions, submission_frequency
- spam_ratio, legitimate_ratio
- account_age_days, is_new_user
- avg_response_time_to_resolutions

# Temporal
- hour_of_day (sin/cos encoded)
- day_of_week, is_weekend, is_holiday
- days_since_last_submission
- submission_rate_last_7_days

# Content Statistics
- char_count, word_count
- capital_ratio, special_char_ratio
- has_urls, has_phone, has_email
- profanity_score
- sentiment_polarity, sentiment_subjectivity
Similarity Features:
# For each new submission
- cosine_similarity_top_5_recent
- max_similarity_same_category
- location_match_count_last_7_days
- category_frequency_multiplier
- semantic_similarity_cluster_center
PHASE 3: MODEL ARCHITECTURE
Approach 1: Ensemble Model (Recommended)
Component 1: Rule-Based Filter (Fast Pre-screening)
├─ Gibberish detector (dictionary word ratio < 30%)
├─ URL/Phone spam detector (regex patterns)
├─ Test submission detector (keywords: test, demo, check)
├─ Extreme length filter (< 10 words or > 1000 words)
└─ Execution time: < 10ms

Component 2: Deep Learning Classifier
├─ Base: IndicBERT / XLM-RoBERTa
├─ Task 1: Binary Spam Classifier (spam vs non-spam)
├─ Task 2: Multi-class (legitimate/spam/duplicate)
├─ Custom heads with dropout and batch normalization
└─ Execution time: < 150ms

Component 3: Similarity Scorer (Siamese Network)
├─ Shared IndicBERT encoder
├─ Contrastive loss / Triplet loss training
├─ Output: Similarity score [0-1]
├─ Compare with last 100 grievances (cached embeddings)
└─ Execution time: < 100ms

Final Decision Logic:
if rule_based_spam_score > 0.95:
    return "SPAM" (high confidence)
elif ml_spam_probability > 0.85:
    return "SPAM" (ml confidence)
elif similarity_score > 0.90 and location_match:
    return "DUPLICATE" (auto-merge)
elif similarity_score > 0.75:
    return "POTENTIAL_DUPLICATE" (human review)
else:
    return "LEGITIMATE"
Approach 2: Unified Transformer Model
Architecture:
Input Text → IndicBERT Encoder → 
  Concatenate [CLS_embedding + metadata_features] →
    ├─→ Spam Detection Head (sigmoid)
    ├─→ Category Classification Head (softmax)
    └─→ Similarity Embedding (L2 normalized)

Training:
- Multi-task learning with weighted loss
- Loss = 0.4 * L_spam + 0.3 * L_category + 0.3 * L_similarity
- Hard negative mining for duplicate detection
- Focal loss for imbalanced classes
PHASE 4: TRAINING PROTOCOL
Data Preprocessing:
# Text cleaning
- Remove extra whitespaces
- Normalize unicode characters
- Expand contractions (don't → do not)
- Handle emojis (convert to text or remove)
- Language detection and tagging

# Tokenization
- Max length: 512 tokens
- Padding: post
- Truncation: longest_first
- Handle multilingual: language-specific tokenizers
Training Configuration:
{
    "model": "ai4bharat/indic-bert",
    "batch_size": 32,
    "learning_rate": 2e-5,
    "epochs": 15,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    
    "optimizer": "AdamW",
    "scheduler": "linear_with_warmup",
    
    "loss_weights": {
        "spam_detection": 0.4,
        "duplicate_detection": 0.3,
        "category_classification": 0.3
    },
    
    "class_weights": {
        "legitimate": 1.0,
        "spam": 2.5,
        "duplicate": 2.0
    },
    
    "early_stopping": {
        "patience": 3,
        "monitor": "val_f1_macro",
        "mode": "max"
    }
}
Data Augmentation:
# During training
- Back-translation (en→hi→en, en→mr→en)
- Synonym replacement (10% of words)
- Random insertion (5% probability)
- Random deletion (5% probability)
- Character-level noise (1-2% typos)
- Paraphrase generation using LLM
PHASE 5: EVALUATION & TESTING
Test Datasets:
Held-out test set: 15% of original data
Temporal test set: Recent 1 month data (check drift)
Adversarial test set: Edge cases and hard negatives
Cross-lingual test set: Ensure language fairness
Metrics:
# Spam Detection
- Precision (target: >95%)
- Recall (target: >90%)
- F1-Score (target: >92%)
- False Positive Rate (target: <2% - CRITICAL)
- ROC-AUC (target: >0.95)

# Duplicate Detection
- Pairwise F1 (target: >85%)
- MAP@K (target: >0.80)
- Recall@5 (target: >90%)
- Duplicate Resolution Rate (target: >80%)

# Business Metrics
- Processing time per grievance (target: <300ms)
- Human review reduction (target: 60%)
- False rejection rate (target: <1%)
- User satisfaction score (target: >85%)
Error Analysis:
Analyze failures by:
- Category (which categories have more errors?)
- Language (fairness across languages?)
- Length (short vs long complaints)
- User type (new vs repeat users)
- Time of day (off-hours submissions)
- Similarity threshold (optimize cutoff)

Create confusion matrices for:
- Spam vs Legitimate
- Duplicate vs Unique
- Per-category performance
PHASE 6: DEPLOYMENT & MONITORING
Production Pipeline:
1. Request arrives → API Gateway
2. Preprocessing (50ms)
   ├─ Text cleaning
   ├─ Language detection
   └─ Feature extraction

3. Rule-based filters (10ms)
   └─ Catch obvious spam

4. ML Model inference (150ms)
   ├─ Spam probability
   ├─ Category prediction
   └─ Embedding generation

5. Similarity search (100ms)
   ├─ Query vector DB (FAISS/Pinecone)
   ├─ Retrieve top-K similar
   └─ Apply filters (location, time, category)

6. Decision logic (10ms)
   └─ Apply business rules

7. Response (10ms)
   └─ Return decision + metadata

Total latency: <350ms (target: <500ms)
Monitoring Dashboard:
Track in real-time:
- Throughput (requests/second)
- Latency (p50, p95, p99)
- Model predictions distribution
- Spam detection rate
- Duplicate detection rate
- False positive alerts
- Model confidence scores
- Data drift indicators

Weekly reviews:
- Manual audit of flagged cases
- User feedback on rejections
- Retrain triggers (accuracy drop >3%)
- A/B test new model versions
PHASE 7: CONTINUOUS IMPROVEMENT
Active Learning:
1. Identify low-confidence predictions (0.4-0.6 range)
2. Route to human annotators
3. Collect feedback
4. Add to training set
5. Retrain monthly with new data
Model Updates:
- Weekly: Update similarity database with new grievances
- Monthly: Retrain with new labeled data
- Quarterly: Full model architecture review
- Yearly: Benchmark against SOTA models
Feedback Loop:
- Track false positives (legitimate marked as spam)
- User appeal mechanism
- Admin override tracking
- Incorporate corrections into training data
- Maintain appeal success rate <5%
SUCCESS CRITERIA
Technical Metrics:
✅ Spam Detection F1 > 92%
✅ Duplicate Detection F1 > 85%
✅ False Positive Rate < 2%
✅ Inference Latency < 500ms
✅ System Uptime > 99.5%

Business Metrics:
✅ 60% reduction in manual review workload
✅ 40% faster grievance resolution time
✅ <1% false rejection rate
✅ >85% citizen satisfaction score
✅ ROI positive within 6 months
RISK MITIGATION
False Rejection Prevention:
Conservative thresholds (high confidence required)
Human review for medium confidence
User appeal mechanism with SLA
Regular audit of rejected complaints
Bias detection across demographics
Adversarial Robustness:
Test against spam evasion techniques
Rate limiting per user
CAPTCHA for suspicious patterns
IP-based anomaly detection
Regular security audits
Fairness & Ethics:
Equal performance across languages
No discrimination by user demographics
Transparent decision explanations
Privacy-preserving (no PII leakage)
Compliance with data protection laws
DELIVERABLES
✅ Trained model with >92% F1 score
✅ REST API with <500ms latency
✅ Admin dashboard for monitoring
✅ Documentation (technical + user guides)
✅ Test suite with 95% coverage
✅ Deployment scripts (Docker + K8s)
✅ Model cards and data sheets
✅ Maintenance playbook
TIMELINE (12 Weeks)
Week 1-2: Data collection & annotation
Week 3-4: Feature engineering & EDA
Week 5-7: Model development & training
Week 8-9: Evaluation & hyperparameter tuning
Week 10: Integration & API development
Week 11: Testing & UAT
Week 12: Deployment & documentation
TEAM REQUIREMENTS
1x ML Engineer (model development)
1x Data Engineer (pipeline setup)
1x Backend Developer (API integration)
1x DevOps Engineer (deployment)
3-5x Data Annotators (labeling)
1x Project Manager
BUDGET ESTIMATE
Cloud compute (training): $2,000
Cloud infrastructure (hosting): $500/month
Data annotation: $3,000
Model APIs (IndicBERT fine-tuning): $500
Monitoring tools: $300/month
Total: ~$8,000 initial + $800/month operational.
