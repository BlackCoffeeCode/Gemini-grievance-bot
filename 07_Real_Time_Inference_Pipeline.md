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
