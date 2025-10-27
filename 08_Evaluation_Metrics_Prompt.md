## 📈 EVALUATION METRICS PROMPT
Certainly. Here is the content for the 8 files, split directly from the project documentation you provided.

You can copy the text from each block below and save it as the corresponding .md file to upload to your Gem's "Knowledge" section.

01_Master_Implementation_Plan.md
Markdown

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
02_Data_Augmentation_Prompt.md
Markdown

AUGMENTATION PROMPT for expanding dataset:

For each legitimate grievance, create 3-5 variations:
1. **Paraphrased version**: Same meaning, different words
2. **Formal version**: Convert casual to formal language
3. **Translated version**: English ↔ Hindi ↔ Regional language
4. **Expanded version**: Add more context and details
5. **Abbreviated version**: Shorter, concise version

For duplicate generation:
- Take original grievance
- Change 20-30% of words with synonyms
- Alter sentence structure while keeping meaning
- Add/remove minor details
- Introduce typos (1-3 per 100 words)
- Mix languages (English + Hindi words)

Example:
Original: "Water supply is irregular in our area"
Duplicates:
- "Water aata jaata rehta hai hamare area mein"
- "Irregular water suplly problm in locality"
- "Our colony has inconsistent water availability"
- "पानी की आपूर्ति हमारे क्षेत्र में अनियमित है"
03_Data_Collection_Annotation_Guidelines.md
Markdown

### 2. Real-World Data Collection Prompt
If using crowdsourcing or actual grievance data:

ANNOTATION GUIDELINES for Human Annotators:

Task: Label each grievance with appropriate tags

Step 1 - Primary Classification: □ Legitimate □ Spam □ Duplicate

Step 2 - If SPAM, specify type: □ Irrelevant content □ Abusive language □ Promotional □ Test submission □ Gibberish □ Bot-generated

Step 3 - If DUPLICATE, answer:

What is the original grievance ID? ___________

Similarity level? □ Exact □ High □ Moderate □ Low

Is it same person resubmitting? □ Yes □ No

Time gap from original: _____ hours/days

Step 4 - Quality Checks:

Is location mentioned? □ Yes □ No

Is issue clearly described? □ Yes □ No

Is action requested? □ Yes □ No

Language quality: □ Clear □ Moderate □ Poor

Red Flags to mark: □ Contains personal attacks □ Has contact info/URLs □ Excessive capitalization □ Nonsensical content □ Appears automated □ Multiple similar submissions from same user

04_Model_Training_Data_Preparation.md
Markdown

## 🔧 MODEL TRAINING DATA PREPARATION PROMPT
PREPROCESSING PIPELINE PROMPT:

Step 1 - Text Cleaning:

Remove extra whitespaces, special characters (keep ! ? .)

Convert to lowercase for non-NER tasks

Remove URLs, email addresses, phone numbers (flag them separately)

Handle multilingual text (preserve Hindi/regional scripts)

Expand common abbreviations (govt → government, plz → please)

Step 2 - Tokenization:

Use language-appropriate tokenizers

Handle code-mixed text (Hinglish)

Preserve location names and proper nouns

Create subword tokens for unknown words

Step 3 - Feature Extraction:

A. Text Embeddings:

Sentence-BERT for semantic similarity

multilingual-BERT for cross-lingual matching

FastText for word-level features

TF-IDF for keyword extraction

B. Metadata Features:

One-hot encode: category, language, location_cluster

Normalize: char_count, word_count, submission_hour

Binary flags: has_urls, has_contact, high_capital_ratio

User behavior: submission_frequency, account_age_days

Temporal: hour_of_day, day_of_week, days_since_last

C. Similarity Features (for duplicate detection):

Cosine similarity with last 100 grievances in same category

Location-based clustering (grievances from same area)

Time-window matching (submissions within 7 days)

Category co-occurrence patterns

Step 4 - Label Encoding: legitimate → 0 spam → 1 duplicate → 2

spam_type encoding: null → 0, irrelevant → 1, abusive → 2, promotional → 3, test → 4, bot_generated → 5, gibberish → 6

Step 5 - Data Balancing:

Use SMOTE for minority class oversampling

Apply class weights: {legitimate: 1.0, spam: 2.5, duplicate: 2.0}

Stratified sampling for train/val/test split

Step 6 - Create Training Batches: Format: [text, metadata_features, similarity_features, label] Batch size: 32 for model training Shuffle: True for training, False for validation/test.


---

### `05_Model_Architecture_and_Config.md`

```md
MULTI-TASK MODEL ARCHITECTURE:

Task 1: Binary Classification (Spam vs Non-Spam)
- Output: [0, 1]
- Loss: Binary Cross-Entropy
- Metrics: Precision, Recall, F1-Score

Task 2: Multi-Class Classification (Legitimate vs Spam vs Duplicate)
- Output: [0, 1, 2]
- Loss: Categorical Cross-Entropy
- Metrics: Accuracy, Per-class F1

Task 3: Duplicate Group Prediction (Similarity Scoring)
- Output: Similarity score [0.0 - 1.0]
- Loss: Contrastive Loss / Triplet Loss
- Metrics: Cosine similarity, AUC-ROC

Shared Embedding Layer:
- Use pre-trained multilingual-BERT or IndicBERT
- Fine-tune on domain-specific grievance corpus
- Freeze lower layers, train upper layers

Architecture:
Input → [Text Encoder + Metadata Encoder] → Concatenate → 
  ├─→ Spam Classifier Head
  ├─→ Multi-class Classifier Head
  └─→ Similarity Scorer Head

Training Strategy:
- Epoch 1-3: Train all heads equally (equal weights)
- Epoch 4-6: Focus on duplicate detection (higher weight)
- Epoch 7-10: Fine-tune with hard negatives. {
  "model_config": {
    "base_model": "ai4bharat/IndicBERT",
    "max_sequence_length": 512,
    "embedding_dim": 768,
    "dropout": 0.3,
    "num_attention_heads": 12
  },
 
  "training_hyperparameters": {
    "batch_size": 32,
    "learning_rate": 2e-5,
    "epochs": 15,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "gradient_clipping": 1.0,
    "optimizer": "AdamW",
    "scheduler": "linear_with_warmup"
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
  },
 
  "data_augmentation": {
    "back_translation": true,
    "synonym_replacement": 0.1,
    "random_insertion": 0.05,
    "random_deletion": 0.05,
    "character_level_noise": 0.02
  }
}
06_Advanced_Features_Prompt.md
Markdown

## 💡 ADVANCED FEATURES & BEST PRACTICES

### 1. Contextual Features Prompt:
EXTRACT AND ENGINEER THESE FEATURES:

Temporal Features:

submission_hour_sin/cos (cyclical encoding)

day_of_week_sin/cos

is_weekend

is_holiday

days_since_last_submission_by_user

submission_rate_last_7days

User Behavior Features:

user_total_submissions

user_spam_ratio

user_legitimate_ratio

avg_time_between_submissions

account_age_days

first_time_user (binary)

Location-based Features:

location_submission_density

location_spam_rate

location_category_distribution

distance_from_city_center (if coordinates available)

Text Statistics:

reading_level (Flesch-Kincaid score)

sentiment_polarity

sentiment_subjectivity

named_entity_count

question_marks_count

exclamation_marks_count

all_caps_words_ratio

Similarity Features (for each new submission):

max_similarity_with_last_100_grievances

avg_similarity_with_same_category

location_category_match_count_7days

07_Real_Time_Inference_Pipeline.md
Markdown

### 2. Real-Time Inference Pipeline:
DEPLOYMENT INFERENCE PROMPT:

When new grievance arrives:

Step 1: Preprocessing (< 50ms)

Clean text

Extract metadata

Generate embeddings

Step 2: Spam Detection (< 100ms)

Rule-based quick filters (URLs, gibberish patterns)

ML model prediction

Confidence threshold: 0.85

If spam_confidence > 0.85 → Flag and skip duplicate check

Step 3: Duplicate Detection (< 200ms)

Retrieve last 100 grievances from same category

Compute similarity scores

Apply location + time filters

If similarity > 0.80 → Flag as potential duplicate

If similarity > 0.90 → Auto-merge with original

Step 4: Human Review Queue

Medium confidence cases (0.65 - 0.85)

Sensitive categories (corruption, police)

First-time users with complex issues

Step 5: Response Return JSON: { "grievance_id": "...", "status": "accepted/flagged_spam/flagged_duplicate", "spam_probability": 0.XX, "duplicate_probability": 0.XX, "similar_grievances": [...], "requires_human_review": true/false, "confidence": 0.XX }

08_Evaluation_Metrics_Prompt.md
Markdown

## 📈 EVALUATION METRICS PROMPT
COMPREHENSIVE EVALUATION FRAMEWORK:

For Spam Detection:

Precision: Among flagged as spam, how many are actually spam? Target: > 95% (minimize false positives)

Recall: Among actual spam, how many did we catch? Target: > 90%

F1-Score: Harmonic mean Target: > 92%

False Positive Rate: Legitimate marked as spam Target: < 2% (CRITICAL - can't reject valid complaints)

For Duplicate Detection:

Pairwise Accuracy: Correct duplicate/non-duplicate classification Target: > 85%

Mean Average Precision (MAP): Ranking quality Target: > 0.80

Recall@K: Found duplicates in top K results Target: Recall@5 > 90%

Time-based Precision: Duplicates within time window Target: > 80% within 7 days

Business Metrics:

Grievance Processing Time Reduction: Target 40%

Human Review Workload Reduction: Target 60%

Citizen Satisfaction: Target > 85%

False Rejection Rate: Target < 1%

Test on Edge Cases: ✓ Similar words, different meanings ✓ Same issue, different severity levels ✓ Partial information vs complete information ✓ Casual vs formal language ✓ Regional language variations ✓ Recent events causing spike in similar complaints ✓ Sarcasm and indirect complaints.
