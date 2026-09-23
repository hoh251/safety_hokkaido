# Feedback flow

1. Deliver the recommendation with risk level, evidence, source freshness, and degraded-service notices.
2. Collect an optional usefulness/safety feedback signal at `POST /feedback` without sensitive personal details.
3. Use `GET /feedback/summary` for local evaluation before using feedback to improve prompts or models.
