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
