# payments

## Purpose
Manages payment processing, subscription lifecycle, and financial event handling.

## Structure
| Path | Role |
|------|------|
| api/ | Payment endpoints |
| services/ | Core payment & billing logic |
| schemas/ | Request/response validation models |
| models/ | Persistence of payment entities |

## TODO / Roadmap
- [ ] Add idempotency on payment webhooks
- [ ] Implement payout reconciliation jobs
- [ ] Provide revenue recognition scheduling
