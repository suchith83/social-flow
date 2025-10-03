# 🚨 PHASE 3 BLOCKER: SQLAlchemy 2.0 Type Annotation Issue

## Issue Discovery

When attempting to run Alembic migrations, we discovered that all our model files have incompatible type annotations for SQLAlchemy 2.0+.

### Error Message

```
sqlalchemy.orm.exc.MappedAnnotationError: Type annotation for "User.videos" can't be 
correctly interpreted for Annotated Declarative Table form.  ORM annotations should 
normally make use of the ``Mapped[]`` generic type, or other ORM-compatible generic 
type, as a container for the actual type, which indicates the intent that the attribute 
is mapped.
```

## Root Cause

All our models use:
```python
videos: relationship["Video"] = relationship(...)
```

But SQLAlchemy 2.0 requires:
```python
videos: Mapped[list["Video"]] = relationship(...)
```

## Affected Files

1. **app/models/user.py** (750 lines) - 12 relationships
2. **app/models/video.py** (850 lines) - 5 relationships
3. **app/models/social.py** (700 lines) - 15+ relationships
4. **app/models/payment.py** (850 lines) - 8 relationships
5. **app/models/ad.py** (900 lines) - 10 relationships
6. **app/models/livestream.py** (850 lines) - 8 relationships
7. **app/models/notification.py** (650 lines) - 5 relationships

**Total:** 63+ relationships need annotation fixes

## Required Changes

### Pattern 1: One-to-Many Relationships

**Current (Broken):**
```python
videos: relationship["Video"] = relationship(
    "Video",
    back_populates="owner",
    cascade="all, delete-orphan",
    lazy="dynamic"
)
```

**Fixed:**
```python
videos: Mapped[list["Video"]] = relationship(
    "Video",
    back_populates="owner",
    cascade="all, delete-orphan",
    lazy="dynamic"
)
```

### Pattern 2: Many-to-One Relationships

**Current (Broken):**
```python
owner: relationship["User"] = relationship(
    "User",
    backref="videos",
    foreign_keys=[owner_id]
)
```

**Fixed:**
```python
owner: Mapped["User"] = relationship(
    "User",
    backref="videos",
    foreign_keys=[owner_id]
)
```

### Pattern 3: Many-to-Many Relationships

**Current (Broken):**
```python
followers: relationship["User"] = relationship(
    "User",
    secondary="follows",
    ...
)
```

**Fixed:**
```python
followers: Mapped[list["User"]] = relationship(
    "User",
    secondary="follows",
    ...
)
```

## Import Changes Required

Add to each model file:
```python
from sqlalchemy.orm import Mapped, mapped_column, relationship
```

## Solution Options

### Option 1: Mass Find/Replace (Risky)
- Use regex to replace all `relationship[` with `Mapped[list[` for 1:many
- Use regex to replace all `relationship["User"]` with `Mapped["User"]` for many:1
- **Risk:** May break some edge cases
- **Time:** 30 minutes
- **Accuracy:** 95%

### Option 2: Manual Review and Fix (Safe)
- Go through each file systematically
- Fix each relationship one by one
- Verify context (1:1, 1:many, many:1, many:many)
- **Risk:** Low
- **Time:** 2-3 hours
- **Accuracy:** 100%

### Option 3: Regenerate Models with Correct Annotations
- Use our comprehensive documentation as reference
- Regenerate each model file with correct annotations
- Preserve all logic, enums, indexes
- **Risk:** Medium (might miss some details)
- **Time:** 4-5 hours
- **Accuracy:** 99%

### Option 4: Keep Models but Don't Use Annotations (Legacy Mode)
- Remove all type annotations from relationships
- Use plain `relationship(...)` without type hints
- Loses IDE autocomplete but works
- **Risk:** Low
- **Time:** 1 hour
- **Accuracy:** 100% (just less type safety)

## Recommended Approach

**Option 2: Manual Review and Fix** is recommended because:

1. ✅ Preserves all our careful work
2. ✅ 100% accurate
3. ✅ Learn SQLAlchemy 2.0 patterns deeply
4. ✅ Can be done systematically
5. ✅ Good documentation opportunity

## Implementation Plan

### Step 1: Fix app/models/base.py (if needed)
- Verify base classes don't have relationship annotations
- ✅ Already clean (no relationships in base)

### Step 2: Fix app/models/user.py (12 relationships)
**Relationships to fix:**
1. `videos: Mapped[list["Video"]]` (1:many)
2. `posts: Mapped[list["Post"]]` (1:many)
3. `followers: Mapped[list["Follow"]]` (1:many)
4. `following: Mapped[list["Follow"]]` (1:many)
5. `comments: Mapped[list["Comment"]]` (1:many)
6. `likes: Mapped[list["Like"]]` (1:many)
7. `payments: Mapped[list["Payment"]]` (1:many)
8. `subscriptions: Mapped[list["Subscription"]]` (1:many)
9. `payouts: Mapped[list["Payout"]]` (1:many)
10. `transactions: Mapped[list["Transaction"]]` (1:many)
11. `ad_campaigns: Mapped[list["AdCampaign"]]` (1:many)
12. `live_streams: Mapped[list["LiveStream"]]` (1:many)

### Step 3: Fix app/models/video.py (5 relationships)
1. `owner: Mapped["User"]` (many:1)
2. `views: Mapped[list["VideoView"]]` (1:many)
3. `likes: Mapped[list["Like"]]` (1:many)
4. `comments: Mapped[list["Comment"]]` (1:many)
5. `saves: Mapped[list["Save"]]` (1:many)

### Step 4: Fix app/models/social.py (15+ relationships)
**Post model:**
1. `owner: Mapped["User"]` (many:1)
2. `comments: Mapped[list["Comment"]]` (1:many)
3. `likes: Mapped[list["Like"]]` (1:many)
4. `saves: Mapped[list["Save"]]` (1:many)
5. `original_post: Mapped["Post"]` (many:1, optional)
6. `reposts: Mapped[list["Post"]]` (1:many)

**Comment model:**
7. `user: Mapped["User"]` (many:1)
8. `post: Mapped["Post"]` (many:1, optional)
9. `video: Mapped["Video"]` (many:1, optional)
10. `parent_comment: Mapped["Comment"]` (many:1, optional)
11. `replies: Mapped[list["Comment"]]` (1:many)
12. `likes: Mapped[list["Like"]]` (1:many)

**Like model:**
13. `user: Mapped["User"]` (many:1)
14. `post: Mapped["Post"]` (many:1, optional)
15. `video: Mapped["Video"]` (many:1, optional)
16. `comment: Mapped["Comment"]` (many:1, optional)

**Follow model:**
17. `follower: Mapped["User"]` (many:1)
18. `following: Mapped["User"]` (many:1)

**Save model:**
19. `user: Mapped["User"]` (many:1)
20. `post: Mapped["Post"]` (many:1, optional)
21. `video: Mapped["Video"]` (many:1, optional)

### Step 5: Fix app/models/payment.py (8 relationships)
**Payment model:**
1. `user: Mapped["User"]` (many:1)
2. `subscription: Mapped["Subscription"]` (many:1, optional)
3. `payout: Mapped["Payout"]` (many:1, optional)
4. `transactions: Mapped[list["Transaction"]]` (1:many)

**Subscription model:**
5. `user: Mapped["User"]` (many:1)
6. `payments: Mapped[list["Payment"]]` (1:many)

**Payout model:**
7. `user: Mapped["User"]` (many:1)
8. `payments: Mapped[list["Payment"]]` (1:many)

**Transaction model:**
9. `user: Mapped["User"]` (many:1)
10. `payment: Mapped["Payment"]` (many:1, optional)
11. `payout: Mapped["Payout"]` (many:1, optional)

### Step 6: Fix app/models/ad.py (10 relationships)
**AdCampaign model:**
1. `advertiser: Mapped["User"]` (many:1)
2. `ads: Mapped[list["Ad"]]` (1:many)

**Ad model:**
3. `campaign: Mapped["AdCampaign"]` (many:1)
4. `video: Mapped["Video"]` (many:1, optional)
5. `impressions: Mapped[list["AdImpression"]]` (1:many)
6. `clicks: Mapped[list["AdClick"]]` (1:many)

**AdImpression model:**
7. `ad: Mapped["Ad"]` (many:1)
8. `campaign: Mapped["AdCampaign"]` (many:1)
9. `user: Mapped["User"]` (many:1, optional)

**AdClick model:**
10. `ad: Mapped["Ad"]` (many:1)
11. `campaign: Mapped["AdCampaign"]` (many:1)
12. `impression: Mapped["AdImpression"]` (many:1)
13. `user: Mapped["User"]` (many:1, optional)

### Step 7: Fix app/models/livestream.py (8 relationships)
**LiveStream model:**
1. `streamer: Mapped["User"]` (many:1)
2. `chat_messages: Mapped[list["StreamChat"]]` (1:many)
3. `donations: Mapped[list["StreamDonation"]]` (1:many)
4. `viewers: Mapped[list["StreamViewer"]]` (1:many)

**StreamChat model:**
5. `stream: Mapped["LiveStream"]` (many:1)
6. `user: Mapped["User"]` (many:1)
7. `deleted_by: Mapped["User"]` (many:1, optional)

**StreamDonation model:**
8. `stream: Mapped["LiveStream"]` (many:1)
9. `donor: Mapped["User"]` (many:1, optional)
10. `payment: Mapped["Payment"]` (many:1, optional)

**StreamViewer model:**
11. `stream: Mapped["LiveStream"]` (many:1)
12. `user: Mapped["User"]` (many:1, optional)

### Step 8: Fix app/models/notification.py (5 relationships)
**Notification model:**
1. `user: Mapped["User"]` (many:1)
2. `actor: Mapped["User"]` (many:1, optional)

**NotificationSettings model:**
3. `user: Mapped["User"]` (1:1)

**PushToken model:**
4. `user: Mapped["User"]` (many:1)

## Additional Import Changes

Each file needs:
```python
from sqlalchemy.orm import Mapped, relationship
```

Remove:
```python
from typing import TYPE_CHECKING  # Still keep this
```

The `TYPE_CHECKING` import should be kept for forward references in Mapped[] types.

## Testing After Fixes

```bash
# Test imports
python -c "from app.models import User; print('OK')"

# Test Alembic
python -m alembic current
python -m alembic revision --autogenerate -m "initial_schema"
python -m alembic upgrade head
```

## Time Estimate

- **Reading and understanding:** 30 minutes ✅ (done)
- **Fixing user.py:** 20 minutes
- **Fixing video.py:** 15 minutes
- **Fixing social.py:** 30 minutes
- **Fixing payment.py:** 20 minutes
- **Fixing ad.py:** 25 minutes
- **Fixing livestream.py:** 20 minutes
- **Fixing notification.py:** 15 minutes
- **Testing:** 15 minutes
- **Total:** ~3 hours

## Status

- [x] Issue identified
- [x] Root cause analyzed
- [x] Solution documented
- [ ] Fixes implemented
- [ ] Tests passed
- [ ] Alembic configured
- [ ] Migration generated

## Next Steps

1. **Ask user:** Which option to proceed with?
2. **If Option 2 (recommended):** Start fixing files systematically
3. **Test after each file:** Ensure imports work
4. **Final test:** Run Alembic migration generation
5. **Document:** Update PHASE_3 documentation

## Alternative: Proceed Without Migrations

If time is critical, we can:
1. Document Phase 2 as "Models Complete"
2. Mark Phase 3 as "Blocked - Requires Type Annotation Refactor"
3. Proceed to Phase 4 (business logic) using the old models in other directories
4. Come back to fix annotations later

This is **NOT recommended** because:
- ❌ Models won't work with SQLAlchemy 2.0
- ❌ Can't generate migrations
- ❌ Can't use the new comprehensive models
- ❌ IDE autocomplete won't work properly

**Recommendation: Spend the 3 hours to fix annotations properly. It's worth it for a production system.**
