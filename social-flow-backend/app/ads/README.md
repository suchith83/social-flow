# ads

## Purpose

Handles advertisement domain logic: targeting, selection, impression & click tracking, analytics attribution.

## Responsibilities

- Match ads to contextual or user signals
- Track impressions / clicks reliably
- Provide analytics summaries for campaigns

## Key Elements

| Path | Role |
|------|------|
| api/ | Request/response surface for ad operations |
| models/ | ORM models (ad, campaign, impression, click) |
| services/ | Targeting & selection logic |

## Extension Points

- Strategy pattern for ad selection algorithms
- Fraud detection hook chain

## Observability

- Metrics: fill rate, CTR, eCPM
- Logging: selection rationale (debug level)

## Security Considerations

- Validate creative metadata; sanitize third-party inputs
- Prevent replay attacks on click endpoints

## TODO / Roadmap

- [ ] Add real-time bidding abstraction
- [ ] Implement fraud/anomaly detection module
- [ ] Introduce pacing & budget enforcement task
