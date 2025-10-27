## 💡 ADVANCED FEATURES & BEST PRACTICES

### 1. Contextual Features Prompt:
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
