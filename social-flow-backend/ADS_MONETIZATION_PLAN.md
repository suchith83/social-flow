# Ads & Monetization Engine - Implementation Plan

## Overview

Comprehensive ad system with targeting, tracking, revenue sharing, and fraud detection.

**Status**: ⏳ **MODELS CREATED** (Service implementation pending)

**Date**: 2024-01-15

---

## Components Created

### 1. Extended Ad Models (`app/models/ad_extended.py` - 370 lines)

#### AdCampaign
- Campaign-level management
- Budget tracking (total + daily)
- Bidding types: CPM, CPC, CPA, CPV
- Status: draft, scheduled, running, paused, completed, cancelled
- Performance metrics aggregation

#### AdCreative
- Individual ad content
- Multiple formats: banner, video, native, sponsored_post, pre_roll, mid_roll, post_roll, story
- Creative assets: image_url, video_url, thumbnail_url
- Targeting criteria (JSONB for flexible querying)
- Approval workflow with moderation

#### AdImpression
- Detailed impression tracking
- Viewability metrics (viewport_percentage, view_duration)
- Geographic data (country, region, city)
- Device tracking (device_type, platform, user_agent)
- Revenue split (creator 70%, platform 30%)
- Fraud detection flags

#### AdClick
- Click event tracking
- Fraud scoring system
- Click position tracking
- Session and IP tracking
- Revenue attribution

#### AdCreatorRevenue
- Aggregated revenue per content piece
- Period-based calculations
- Revenue breakdown by source
- Payout status tracking
- Integration with Stripe Connect payouts

---

## Revenue Model

### Revenue Sharing
```
Creator Share: 70%
Platform Share: 30%
```

### Bidding Types
1. **CPM** (Cost Per Mille): Pay per 1000 impressions
2. **CPC** (Cost Per Click): Pay per click
3. **CPA** (Cost Per Acquisition): Pay per conversion
4. **CPV** (Cost Per View): Pay per video view

---

## Database Schema

### Tables Created (5 tables, 19 indexes)

#### `ad_campaigns`
- 18 columns
- Indexes: advertiser_id, status, start_date

#### `ad_creatives`
- 20 columns with JSONB targeting
- Indexes: campaign_id, format, is_active

#### `ad_impressions`
- 21 columns
- Indexes: creative_id, campaign_id, user_id, created_at, content_id

#### `ad_clicks`
- 21 columns
- Indexes: creative_id, campaign_id, user_id, created_at, impression_id

#### `ad_creator_revenues`
- 17 columns
- Indexes: creator_id, content_id, period_start, is_paid

---

## Key Features

### Ad Targeting
- **Demographics**: Age, gender, location
- **Interests**: Based on user behavior
- **Context**: Content type, category, keywords
- **Device**: Mobile, desktop, tablet
- **Platform**: iOS, Android, Web
- **Geographic**: Country, region, city

### Fraud Detection
- **Rate Limiting**: 
  - Max 50 impressions per user per hour
  - Max 10 clicks per user per day per creative
- **Rapid Click Detection**: 5-second window
- **Fraud Scoring**: 0.0-1.0 scale
- **Invalid Click Filtering**: Prevents billing for suspicious activity

### Viewability Tracking
- **Minimum Requirements**:
  - 50% of ad visible in viewport
  - 1 second minimum view duration
- **Metrics**: viewport_percentage, view_duration
- **Industry Standard**: IAB viewability guidelines

### Revenue Sharing
- **Automatic Calculation**: On each impression/click
- **Creator Share**: 70% of gross revenue
- **Platform Share**: 30% of gross revenue
- **Payout Integration**: Links with Stripe Connect
- **Period-based**: Daily/weekly/monthly aggregation

---

## Service Implementation (Pending)

The comprehensive `AdsService` should include:

### Campaign Management (8 methods)
```python
create_campaign()
get_campaign()
update_campaign_status()
pause_campaign()
resume_campaign()
list_campaigns()
get_campaign_analytics()
check_budget()
```

### Creative Management (5 methods)
```python
create_creative()
approve_creative()
reject_creative()
update_creative()
list_creatives()
```

### Ad Serving (5 methods)
```python
get_targeted_ads()
_get_user_targeting_profile()
_calculate_age()
_check_frequency_cap()
_filter_by_budget()
```

### Tracking (6 methods)
```python
track_impression()
_check_impression_fraud()
track_click()
_check_click_fraud()
track_conversion()
track_viewability()
```

### Revenue Sharing (4 methods)
```python
calculate_creator_revenue()
aggregate_period_revenue()
get_creator_earnings_summary()
process_ad_payouts()
```

### Analytics (6 methods)
```python
get_campaign_analytics()
get_creative_analytics()
get_platform_analytics()
get_advertiser_dashboard()
get_creator_dashboard()
export_analytics()
```

---

## API Endpoints (Planned)

### Campaign Management
```
POST   /api/v1/ads/campaigns              # Create campaign
GET    /api/v1/ads/campaigns              # List campaigns
GET    /api/v1/ads/campaigns/{id}         # Get campaign
PUT    /api/v1/ads/campaigns/{id}         # Update campaign
DELETE /api/v1/ads/campaigns/{id}         # Delete campaign
POST   /api/v1/ads/campaigns/{id}/pause   # Pause campaign
POST   /api/v1/ads/campaigns/{id}/resume  # Resume campaign
```

### Creative Management
```
POST   /api/v1/ads/creatives                   # Create creative
GET    /api/v1/ads/creatives                   # List creatives
GET    /api/v1/ads/creatives/{id}              # Get creative
PUT    /api/v1/ads/creatives/{id}              # Update creative
DELETE /api/v1/ads/creatives/{id}              # Delete creative
POST   /api/v1/ads/creatives/{id}/approve     # Approve (admin)
POST   /api/v1/ads/creatives/{id}/reject      # Reject (admin)
```

### Ad Serving
```
GET    /api/v1/ads/serve                  # Get ads for placement
POST   /api/v1/ads/impressions            # Track impression
POST   /api/v1/ads/clicks                 # Track click
POST   /api/v1/ads/conversions            # Track conversion
```

### Analytics
```
GET    /api/v1/ads/campaigns/{id}/analytics    # Campaign analytics
GET    /api/v1/ads/creatives/{id}/analytics    # Creative analytics
GET    /api/v1/ads/analytics/platform          # Platform analytics (admin)
GET    /api/v1/ads/analytics/creator           # Creator ad revenue
```

### Revenue
```
GET    /api/v1/ads/revenue/creator             # Creator earnings
GET    /api/v1/ads/revenue/content/{id}        # Content-specific revenue
POST   /api/v1/ads/revenue/calculate           # Calculate period revenue
```

---

## Integration Points

### Payment System Integration
- Link `AdCreatorRevenue.payout_id` to `creator_payouts.id`
- Aggregate ad revenue into monthly/weekly payouts
- Include ad_revenue in CreatorPayout revenue_breakdown

### Content Integration
- Track ads shown on videos (`content_type='video'`)
- Track ads shown on posts (`content_type='post'`)
- Track ads in stories (`content_type='story'`)
- Link via `content_id` foreign key

### User Authentication
- Require authentication for ad serving (optional for logged-out users)
- Track user_id for targeting and fraud detection
- Use session_id for anonymous users

### Analytics Integration
- Real-time impression/click tracking
- Aggregate metrics for dashboards
- Export to data warehouse for deep analysis

---

## Migration Required

### Migration 005: Ad Extended Tables

```python
# Create tables
- ad_campaigns (18 columns, 3 indexes)
- ad_creatives (20 columns, 3 indexes)
- ad_impressions (21 columns, 5 indexes)
- ad_clicks (21 columns, 5 indexes)
- ad_creator_revenues (17 columns, 4 indexes)

# Update existing tables
- ads: Add campaign_id foreign key
- creator_payouts: Already has ad_revenue field

# Seed data
- Create sample campaigns for testing
- Create test creatives in various formats
```

---

## Testing Strategy

### Unit Tests
- Campaign CRUD operations
- Creative approval workflow
- Revenue calculation accuracy
- Fraud detection logic
- Targeting algorithm

### Integration Tests
- End-to-end ad serving
- Impression to click to conversion flow
- Revenue aggregation and payout
- Budget tracking and campaign pause

### Performance Tests
- Ad serving latency (<50ms)
- Concurrent impression tracking
- High-volume click processing
- Analytics query performance

### Fraud Detection Tests
- Rapid clicking scenarios
- High-frequency impressions
- Bot detection accuracy
- Invalid traffic filtering

---

## Configuration

### Environment Variables
```bash
# Ad Service Configuration
ADS_ENABLED=true
ADS_MIN_VIEWPORT_PERCENTAGE=50.0
ADS_MIN_VIEW_DURATION=1000
ADS_CREATOR_REVENUE_SHARE=0.70
ADS_PLATFORM_REVENUE_SHARE=0.30

# Fraud Detection
ADS_MAX_CLICKS_PER_USER_PER_DAY=10
ADS_MAX_IMPRESSIONS_PER_USER_PER_HOUR=50
ADS_RAPID_CLICK_WINDOW=5

# Revenue Processing
ADS_REVENUE_CALCULATION_SCHEDULE="0 0 * * *"  # Daily at midnight
ADS_MINIMUM_PAYOUT=10.00
```

---

## Monitoring & Alerts

### Key Metrics
- **Fill Rate**: Percentage of ad requests filled
- **Viewability Rate**: Percentage of viewable impressions
- **CTR**: Click-through rate
- **Fraud Rate**: Percentage of invalid traffic
- **Revenue per 1000 Impressions** (RPM)
- **Campaign Completion Rate**

### Alerts
- Campaign budget exhausted
- High fraud score detected
- Low fill rate (<80%)
- Revenue calculation failures
- Payout processing errors

---

## Future Enhancements

### Phase 2
- [ ] A/B testing for creatives
- [ ] Dynamic creative optimization
- [ ] Lookalike audience targeting
- [ ] Retargeting campaigns
- [ ] Brand safety controls
- [ ] Third-party ad network integration
- [ ] Programmatic advertising (RTB)
- [ ] Video completion tracking
- [ ] Custom audience segments
- [ ] Conversion pixel tracking

### Phase 3
- [ ] Machine learning for bid optimization
- [ ] Predictive analytics for campaign performance
- [ ] Automated budget allocation
- [ ] Dynamic pricing based on demand
- [ ] Advanced fraud detection with ML
- [ ] Real-time bidding platform
- [ ] Self-serve advertiser portal
- [ ] Automated creative generation
- [ ] Cross-device attribution
- [ ] Privacy-compliant tracking (post-cookies)

---

## Status Summary

### ✅ Completed
1. ✅ Extended ad models (AdCampaign, AdCreative, AdImpression, AdClick, AdCreatorRevenue)
2. ✅ Revenue sharing model defined (70/30 split)
3. ✅ Fraud detection thresholds specified
4. ✅ Viewability standards set
5. ✅ Database schema designed (5 tables, 19 indexes)

### 🔄 In Progress
6. ⏳ Comprehensive ads service implementation
7. ⏳ Database migration 005
8. ⏳ API endpoints for ad management
9. ⏳ Integration with payment system

### ⏭️ Pending
10. ⏭️ Ad serving algorithm implementation
11. ⏭️ Targeting engine development
12. ⏭️ Analytics dashboard
13. ⏭️ Testing suite
14. ⏭️ Documentation

---

## Next Steps

1. Complete `AdsService` implementation (850+ lines)
2. Create database migration 005
3. Implement API routers (4 files)
4. Create ad schemas (Pydantic models)
5. Integrate with Stripe Connect for payouts
6. Add Celery tasks for revenue calculation
7. Create analytics queries
8. Build advertiser and creator dashboards
9. Test fraud detection system
10. Document API endpoints

---

*Document Version: 1.0*  
*Last Updated: 2024-01-15*  
*Implementation: 25% complete (models created)*
