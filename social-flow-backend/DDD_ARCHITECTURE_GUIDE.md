# 🏗️ Domain-Driven Design (DDD) Architecture Guide

**Social Flow Backend - Enterprise Architecture**  
**Version:** 2.0.0  
**Last Updated:** October 2, 2025

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Core Principles](#core-principles)
3. [Layered Architecture](#layered-architecture)
4. [Bounded Contexts](#bounded-contexts)
5. [Directory Structure](#directory-structure)
6. [Code Organization Patterns](#code-organization-patterns)
7. [Dependency Rules](#dependency-rules)
8. [Migration Strategy](#migration-strategy)
9. [Best Practices](#best-practices)
10. [Examples](#examples)

---

## 🎯 Overview

Social Flow Backend follows **Domain-Driven Design (DDD)** and **Clean Architecture** principles to achieve:

### Goals
- ✅ **Separation of Concerns**: Business logic isolated from infrastructure
- ✅ **Testability**: Easy to unit test domain logic without external dependencies
- ✅ **Maintainability**: Clear structure makes code easy to understand and modify
- ✅ **Scalability**: Bounded contexts can be extracted into microservices
- ✅ **Team Collaboration**: Different teams can work on different bounded contexts

### Architecture Style
- **Domain-Driven Design (DDD)**: Focus on business domain and bounded contexts
- **Clean Architecture**: Dependency inversion, entities at the center
- **Hexagonal Architecture (Ports & Adapters)**: Infrastructure adapts to domain

---

## 🧩 Core Principles

### 1. Domain-Driven Design (DDD)

**Ubiquitous Language**: Use consistent terminology across code, docs, and conversations
- Example: `Video`, `EncodingJob`, `CopyrightClaim` (not `vid`, `job`, `claim`)

**Bounded Contexts**: Self-contained modules with clear boundaries
- Example: `auth`, `videos`, `posts`, `livestream`, `payments`

**Entities**: Objects with unique identity that persist over time
- Example: `User`, `Video`, `Post` (have IDs, can change state)

**Value Objects**: Immutable objects defined by their attributes
- Example: `Email`, `VideoStatus`, `Money` (no ID, immutable)

**Aggregates**: Cluster of entities/value objects treated as a single unit
- Example: `Video` aggregate (contains `EncodingJob`, `Thumbnail`, `ViewCount`)

**Domain Services**: Business logic that doesn't belong to a single entity
- Example: `CopyrightDetectionService`, `RecommendationEngine`

**Domain Events**: Signals that something important happened
- Example: `VideoUploaded`, `UserRegistered`, `PaymentProcessed`

### 2. Clean Architecture Layers

```
┌─────────────────────────────────────────────┐
│         Presentation Layer (FastAPI)        │ ← External Interface
├─────────────────────────────────────────────┤
│        Application Layer (Use Cases)        │ ← Orchestration
├─────────────────────────────────────────────┤
│       Domain Layer (Business Logic)         │ ← Core Business
├─────────────────────────────────────────────┤
│  Infrastructure Layer (DB, S3, External)    │ ← Technical Details
└─────────────────────────────────────────────┘
```

**Dependency Rule**: Dependencies point INWARD
- Presentation → Application → Domain
- Infrastructure → Domain (via interfaces)
- Domain has NO dependencies on outer layers

---

## 📚 Layered Architecture

### Layer 1: Domain Layer (Core)

**Purpose**: Contains pure business logic, no technical concerns

**Components**:
- **Entities**: Objects with identity (e.g., `User`, `Video`)
- **Value Objects**: Immutable descriptors (e.g., `Email`, `VideoStatus`)
- **Domain Services**: Business operations (e.g., `CopyrightDetectionService`)
- **Repositories (Interfaces)**: Abstract data access
- **Domain Events**: Business event definitions

**Rules**:
- ❌ NO framework dependencies (FastAPI, SQLAlchemy, etc.)
- ❌ NO infrastructure code (database, S3, Redis, etc.)
- ✅ Pure Python with business logic only
- ✅ Can depend on other domain concepts
- ✅ Can define repository interfaces (implemented elsewhere)

**Example**:
```python
# app/videos/domain/entities/video.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class Video:
    """Video entity - Core business object."""
    id: str
    user_id: str
    title: str
    description: Optional[str]
    status: VideoStatus
    uploaded_at: datetime
    
    def mark_as_processing(self) -> None:
        """Business rule: Video must be uploaded before processing."""
        if self.status != VideoStatus.UPLOADED:
            raise ValueError(f"Cannot process video in {self.status} status")
        self.status = VideoStatus.PROCESSING
    
    def mark_as_ready(self, hls_url: str) -> None:
        """Business rule: Video must be processed before going live."""
        if self.status != VideoStatus.PROCESSING:
            raise ValueError("Video must be processing to mark as ready")
        self.hls_url = hls_url
        self.status = VideoStatus.READY
```

---

### Layer 2: Application Layer (Use Cases)

**Purpose**: Orchestrates domain logic to fulfill user requests

**Components**:
- **Use Cases**: Application-specific business flows
- **DTOs (Data Transfer Objects)**: Input/output structures
- **Application Services**: Coordinate domain services

**Rules**:
- ✅ CAN depend on domain layer
- ✅ CAN define application-specific DTOs
- ✅ Orchestrates multiple domain services
- ❌ NO direct infrastructure access (use dependency injection)
- ❌ NO presentation concerns (HTTP, JSON, etc.)

**Example**:
```python
# app/videos/application/use_cases/upload_video.py
from dataclasses import dataclass
from typing import Protocol

@dataclass
class UploadVideoRequest:
    """Input DTO for video upload use case."""
    user_id: str
    title: str
    description: str
    file_path: str

@dataclass
class UploadVideoResponse:
    """Output DTO for video upload use case."""
    video_id: str
    status: str
    message: str

class IStorageService(Protocol):
    """Interface for storage (implemented in infrastructure)."""
    async def upload(self, path: str, data: bytes) -> str: ...

class IVideoRepository(Protocol):
    """Interface for video persistence."""
    async def save(self, video: Video) -> None: ...

class UploadVideoUseCase:
    """Use case: Upload a video file and create video record."""
    
    def __init__(
        self,
        storage: IStorageService,
        repository: IVideoRepository,
    ):
        self._storage = storage
        self._repository = repository
    
    async def execute(self, request: UploadVideoRequest) -> UploadVideoResponse:
        """Execute the video upload flow."""
        # 1. Upload file to storage
        s3_url = await self._storage.upload(request.file_path, ...)
        
        # 2. Create domain entity
        video = Video(
            id=generate_id(),
            user_id=request.user_id,
            title=request.title,
            description=request.description,
            status=VideoStatus.UPLOADED,
            uploaded_at=datetime.utcnow(),
        )
        
        # 3. Persist to database
        await self._repository.save(video)
        
        # 4. Return response
        return UploadVideoResponse(
            video_id=video.id,
            status="uploaded",
            message="Video uploaded successfully",
        )
```

---

### Layer 3: Infrastructure Layer (Technical Details)

**Purpose**: Implements technical concerns (database, external services)

**Components**:
- **Persistence**: SQLAlchemy repositories, database models
- **Storage**: S3/Azure/GCS adapters
- **External Services**: AWS MediaConvert, Stripe, FCM
- **Cache**: Redis implementations
- **Messaging**: Celery tasks, message queues

**Rules**:
- ✅ Implements domain repository interfaces
- ✅ Contains framework-specific code (SQLAlchemy, boto3, etc.)
- ✅ Adapts external services to domain interfaces
- ❌ Does NOT contain business logic (delegate to domain)

**Example**:
```python
# app/videos/infrastructure/persistence/video_repository.py
from sqlalchemy.ext.asyncio import AsyncSession
from app.videos.domain.entities.video import Video
from app.videos.domain.repositories.video_repository import IVideoRepository

class SQLAlchemyVideoRepository(IVideoRepository):
    """Concrete implementation using SQLAlchemy."""
    
    def __init__(self, session: AsyncSession):
        self._session = session
    
    async def save(self, video: Video) -> None:
        """Persist video entity to PostgreSQL."""
        db_video = VideoModel(
            id=video.id,
            user_id=video.user_id,
            title=video.title,
            description=video.description,
            status=video.status.value,
            uploaded_at=video.uploaded_at,
        )
        self._session.add(db_video)
        await self._session.commit()
    
    async def find_by_id(self, video_id: str) -> Optional[Video]:
        """Retrieve video by ID."""
        result = await self._session.execute(
            select(VideoModel).where(VideoModel.id == video_id)
        )
        db_video = result.scalar_one_or_none()
        
        if not db_video:
            return None
        
        # Convert database model to domain entity
        return Video(
            id=db_video.id,
            user_id=db_video.user_id,
            title=db_video.title,
            description=db_video.description,
            status=VideoStatus(db_video.status),
            uploaded_at=db_video.uploaded_at,
        )
```

---

### Layer 4: Presentation Layer (API/UI)

**Purpose**: Handles external communication (HTTP, WebSocket, CLI)

**Components**:
- **API Routes**: FastAPI endpoints
- **Schemas**: Pydantic request/response models
- **Controllers**: Thin layer coordinating use cases
- **Middleware**: Authentication, validation, error handling

**Rules**:
- ✅ Converts HTTP/JSON to DTOs
- ✅ Calls application use cases
- ✅ Handles HTTP-specific concerns (status codes, headers)
- ❌ NO business logic (delegate to application/domain)
- ❌ NO direct database access (use repositories via use cases)

**Example**:
```python
# app/videos/presentation/api/video_routes.py
from fastapi import APIRouter, UploadFile, Depends
from app.videos.presentation.schemas.video_schemas import UploadVideoRequest, VideoResponse
from app.videos.application.use_cases.upload_video import UploadVideoUseCase

router = APIRouter(prefix="/videos", tags=["videos"])

@router.post("/upload", response_model=VideoResponse)
async def upload_video(
    request: UploadVideoRequest,
    use_case: UploadVideoUseCase = Depends(get_upload_use_case),
    current_user: User = Depends(get_current_user),
):
    """Upload a new video."""
    # 1. Convert API request to use case DTO
    upload_request = UploadVideoRequest(
        user_id=current_user.id,
        title=request.title,
        description=request.description,
        file_path=request.file_path,
    )
    
    # 2. Execute use case
    result = await use_case.execute(upload_request)
    
    # 3. Convert use case response to API response
    return VideoResponse(
        id=result.video_id,
        status=result.status,
        message=result.message,
    )
```

---

## 🔗 Bounded Contexts

### What is a Bounded Context?

A **bounded context** is a logical boundary where a specific domain model applies. Each context has its own:
- Domain entities and value objects
- Business rules and invariants
- Database tables (or shared with explicit mapping)
- API endpoints

### Social Flow Bounded Contexts

```
┌─────────────────────────────────────────────────────────────┐
│                     SOCIAL FLOW PLATFORM                    │
├─────────────┬─────────────┬─────────────┬──────────────────┤
│    Auth     │   Videos    │    Posts    │   Livestream     │
│  (Identity) │  (Content)  │  (Social)   │   (Streaming)    │
├─────────────┼─────────────┼─────────────┼──────────────────┤
│     Ads     │  Payments   │     ML      │  Notifications   │
│ (Monetize)  │  (Finance)  │    (AI)     │   (Messaging)    │
├─────────────┴─────────────┴─────────────┴──────────────────┤
│                    SHARED KERNEL                             │
│           (Common Value Objects, Infrastructure)             │
└──────────────────────────────────────────────────────────────┘
```

### Context Descriptions

| Context | Purpose | Key Entities | Integration Points |
|---------|---------|--------------|-------------------|
| **auth** | User identity, authentication, authorization | User, Session, Role | All contexts (user_id) |
| **videos** | Video upload, encoding, playback | Video, EncodingJob, Thumbnail | posts, ads, copyright |
| **posts** | Text posts, feed, social interactions | Post, Comment, Like, Repost | videos (embed), ads |
| **livestream** | Live video streaming, chat | Stream, StreamKey, ChatMessage | ads, notifications |
| **ads** | Advertisement serving, targeting | Ad, Campaign, Impression | videos, posts, livestream |
| **payments** | Billing, subscriptions, payouts | Payment, Subscription, Payout | videos, ads (revenue) |
| **ml** | Recommendations, moderation, copyright | Model, Prediction, Training | videos, posts (content) |
| **notifications** | Push, email, WebSocket alerts | Notification, Template | All contexts (events) |
| **analytics** | Metrics, reporting, dashboards | Metric, Report, Dashboard | All contexts (tracking) |

### Context Relationships

**Integration Patterns**:
1. **Shared ID**: Reference by ID (e.g., `user_id`, `video_id`)
2. **Domain Events**: Publish/subscribe for async communication
3. **API Calls**: Synchronous HTTP between contexts (if needed)
4. **Shared Kernel**: Common value objects, infrastructure

**Example**:
```python
# Videos context publishes event
VideoUploadedEvent(video_id="123", user_id="456")

# ML context subscribes to event
@subscribe_to(VideoUploadedEvent)
async def analyze_video(event: VideoUploadedEvent):
    """Run content moderation on newly uploaded video."""
    await moderation_service.analyze(event.video_id)

# Notifications context subscribes to event
@subscribe_to(VideoUploadedEvent)
async def notify_followers(event: VideoUploadedEvent):
    """Notify user's followers of new video."""
    followers = await get_followers(event.user_id)
    await send_notifications(followers, f"New video from {event.user_id}")
```

---

## 📂 Directory Structure

### Complete Structure

```
social-flow-backend/
├── app/
│   ├── shared/                          # 🆕 Shared Kernel (Cross-cutting concerns)
│   │   ├── domain/                      # Shared domain concepts
│   │   │   ├── entities/                # Common entities (if any)
│   │   │   ├── value_objects/           # Email, Money, Address, etc.
│   │   │   └── events/                  # Domain event base classes
│   │   ├── application/                 # Shared application layer
│   │   │   ├── dto/                     # Common DTOs
│   │   │   └── interfaces/              # Common interfaces
│   │   └── infrastructure/              # Shared infrastructure
│   │       ├── database/                # Database session, base repository
│   │       ├── cache/                   # Redis connection, cache service
│   │       └── messaging/               # Event bus, message queue
│   │
│   ├── auth/                            # Auth Bounded Context
│   │   ├── domain/
│   │   │   ├── entities/                # User, Session
│   │   │   ├── value_objects/           # Email, Password, Role
│   │   │   ├── repositories/            # IUserRepository (interface)
│   │   │   └── services/                # AuthenticationService
│   │   ├── application/
│   │   │   ├── use_cases/               # RegisterUser, LoginUser, RefreshToken
│   │   │   └── dto/                     # RegisterRequest, LoginResponse
│   │   ├── infrastructure/
│   │   │   ├── persistence/             # SQLAlchemyUserRepository
│   │   │   └── security/                # JWTHandler, PasswordHasher
│   │   └── presentation/
│   │       ├── api/                     # auth_routes.py
│   │       └── schemas/                 # Pydantic schemas
│   │
│   ├── videos/                          # Videos Bounded Context
│   │   ├── domain/
│   │   │   ├── entities/                # Video, EncodingJob, Thumbnail
│   │   │   ├── value_objects/           # VideoStatus, Resolution, Quality
│   │   │   ├── repositories/            # IVideoRepository (interface)
│   │   │   └── services/                # EncodingService, ViewCountService
│   │   ├── application/
│   │   │   ├── use_cases/               # UploadVideo, EncodeVideo, GetVideo
│   │   │   └── dto/                     # UploadVideoRequest, VideoResponse
│   │   ├── infrastructure/
│   │   │   ├── persistence/             # SQLAlchemyVideoRepository
│   │   │   ├── storage/                 # S3StorageAdapter
│   │   │   └── encoding/                # MediaConvertAdapter, FFmpegService
│   │   └── presentation/
│   │       ├── api/                     # video_routes.py
│   │       └── schemas/                 # VideoSchema, UploadSchema
│   │
│   ├── posts/                           # Posts Bounded Context
│   │   ├── domain/                      # Post, Comment, Like
│   │   ├── application/                 # CreatePost, GetFeed
│   │   ├── infrastructure/              # Repositories, cache
│   │   └── presentation/                # API routes
│   │
│   ├── livestream/                      # Livestream Bounded Context
│   │   ├── domain/                      # Stream, StreamKey, ChatMessage
│   │   ├── application/                 # StartStream, SendChatMessage
│   │   ├── infrastructure/              # AWS IVS, WebSocket
│   │   └── presentation/                # API routes, WebSocket handlers
│   │
│   ├── ads/                             # Ads Bounded Context
│   │   ├── domain/                      # Ad, Campaign, Impression
│   │   ├── application/                 # ServeAd, TrackImpression
│   │   ├── infrastructure/              # Ad targeting, analytics
│   │   └── presentation/                # API routes
│   │
│   ├── payments/                        # Payments Bounded Context
│   │   ├── domain/                      # Payment, Subscription, Payout
│   │   ├── application/                 # ProcessPayment, CalculatePayout
│   │   ├── infrastructure/              # Stripe, payment processor
│   │   └── presentation/                # API routes, webhooks
│   │
│   ├── ml/                              # ML/AI Bounded Context
│   │   ├── domain/                      # Model, Prediction
│   │   ├── application/                 # GetRecommendations, ModerateContent
│   │   ├── infrastructure/              # SageMaker, model serving
│   │   └── presentation/                # API routes
│   │
│   ├── notifications/                   # Notifications Bounded Context
│   │   ├── domain/                      # Notification, Template
│   │   ├── application/                 # SendNotification
│   │   ├── infrastructure/              # FCM, SES, WebSocket
│   │   └── presentation/                # API routes
│   │
│   ├── analytics/                       # Analytics Bounded Context
│   │   ├── domain/                      # Metric, Report
│   │   ├── application/                 # TrackEvent, GenerateReport
│   │   ├── infrastructure/              # Time-series DB, aggregations
│   │   └── presentation/                # API routes
│   │
│   ├── api/                             # 🔄 Legacy API (to be migrated)
│   ├── core/                            # 🔄 Core utilities (move to shared/)
│   ├── models/                          # 🗑️  DEPRECATED (consolidated)
│   └── main.py                          # FastAPI application entry point
│
├── tests/
│   ├── unit/                            # Unit tests (domain, application)
│   │   ├── auth/
│   │   ├── videos/
│   │   └── ...
│   ├── integration/                     # Integration tests (infrastructure)
│   │   ├── api/
│   │   ├── database/
│   │   └── ...
│   └── e2e/                             # End-to-end tests
│       ├── auth_flow/
│       ├── video_upload/
│       └── ...
│
├── alembic/                             # Database migrations
├── scripts/                             # Deployment, maintenance scripts
├── docs/                                # Documentation
└── requirements.txt
```

---

## 🎨 Code Organization Patterns

### 1. Entities vs Value Objects

**Entity** (has identity, mutable):
```python
@dataclass
class User:
    """Entity: User has unique ID and can change state."""
    id: str
    username: str
    email: Email  # Value Object
    created_at: datetime
    
    def change_email(self, new_email: Email) -> None:
        """User can change email while keeping same identity."""
        self.email = new_email
```

**Value Object** (no identity, immutable):
```python
@dataclass(frozen=True)  # Immutable
class Email:
    """Value Object: Defined by its value, no separate identity."""
    value: str
    
    def __post_init__(self):
        """Validate email format on creation."""
        if "@" not in self.value:
            raise ValueError(f"Invalid email: {self.value}")
    
    @property
    def domain(self) -> str:
        """Derive domain from email value."""
        return self.value.split("@")[1]
```

### 2. Repository Pattern

**Interface (in domain)**:
```python
# app/videos/domain/repositories/video_repository.py
from abc import ABC, abstractmethod
from typing import Optional, List

class IVideoRepository(ABC):
    """Abstract repository interface - defined in domain layer."""
    
    @abstractmethod
    async def save(self, video: Video) -> None:
        """Persist video entity."""
        pass
    
    @abstractmethod
    async def find_by_id(self, video_id: str) -> Optional[Video]:
        """Retrieve video by ID."""
        pass
    
    @abstractmethod
    async def find_by_user_id(self, user_id: str) -> List[Video]:
        """Get all videos by user."""
        pass
```

**Implementation (in infrastructure)**:
```python
# app/videos/infrastructure/persistence/video_repository.py
class SQLAlchemyVideoRepository(IVideoRepository):
    """Concrete implementation using SQLAlchemy."""
    
    def __init__(self, session: AsyncSession):
        self._session = session
    
    async def save(self, video: Video) -> None:
        """Persist to PostgreSQL."""
        db_video = self._to_db_model(video)
        self._session.add(db_video)
        await self._session.commit()
    
    async def find_by_id(self, video_id: str) -> Optional[Video]:
        """Retrieve from PostgreSQL."""
        result = await self._session.execute(
            select(VideoModel).where(VideoModel.id == video_id)
        )
        db_video = result.scalar_one_or_none()
        return self._to_entity(db_video) if db_video else None
    
    def _to_db_model(self, video: Video) -> VideoModel:
        """Convert domain entity to database model."""
        return VideoModel(
            id=video.id,
            user_id=video.user_id,
            title=video.title,
            status=video.status.value,
            # ... other fields
        )
    
    def _to_entity(self, db_video: VideoModel) -> Video:
        """Convert database model to domain entity."""
        return Video(
            id=db_video.id,
            user_id=db_video.user_id,
            title=db_video.title,
            status=VideoStatus(db_video.status),
            # ... other fields
        )
```

### 3. Dependency Injection

**Setup (in main.py)**:
```python
# app/main.py
from fastapi import FastAPI, Depends
from app.videos.application.use_cases.upload_video import UploadVideoUseCase
from app.videos.infrastructure.persistence.video_repository import SQLAlchemyVideoRepository
from app.videos.infrastructure.storage.s3_storage import S3StorageService

app = FastAPI()

# Dependency providers
async def get_db_session() -> AsyncSession:
    """Provide database session."""
    async with get_session() as session:
        yield session

def get_video_repository(session: AsyncSession = Depends(get_db_session)):
    """Provide video repository."""
    return SQLAlchemyVideoRepository(session)

def get_storage_service():
    """Provide storage service."""
    return S3StorageService(bucket="videos")

def get_upload_use_case(
    repository = Depends(get_video_repository),
    storage = Depends(get_storage_service),
):
    """Provide upload video use case with dependencies."""
    return UploadVideoUseCase(storage, repository)

# Use in route
@app.post("/videos/upload")
async def upload_video(
    request: UploadRequest,
    use_case: UploadVideoUseCase = Depends(get_upload_use_case),
):
    """Endpoint automatically gets use case with all dependencies injected."""
    result = await use_case.execute(request)
    return result
```

### 4. Domain Events

**Event Definition (in domain)**:
```python
# app/shared/domain/events/base.py
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict

@dataclass
class DomainEvent:
    """Base class for all domain events."""
    event_id: str
    occurred_at: datetime
    event_type: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize event to dictionary."""
        return {
            "event_id": self.event_id,
            "occurred_at": self.occurred_at.isoformat(),
            "event_type": self.event_type,
        }

# app/videos/domain/events/video_events.py
@dataclass
class VideoUploadedEvent(DomainEvent):
    """Event: Video was uploaded."""
    video_id: str
    user_id: str
    title: str
    
    def __post_init__(self):
        self.event_type = "video.uploaded"
```

**Publishing Events (in use case)**:
```python
# app/videos/application/use_cases/upload_video.py
class UploadVideoUseCase:
    def __init__(self, event_bus: IEventBus, ...):
        self._event_bus = event_bus
        # ... other dependencies
    
    async def execute(self, request: UploadVideoRequest):
        # ... upload logic ...
        
        # Publish domain event
        event = VideoUploadedEvent(
            event_id=generate_id(),
            occurred_at=datetime.utcnow(),
            video_id=video.id,
            user_id=video.user_id,
            title=video.title,
        )
        await self._event_bus.publish(event)
        
        return response
```

**Subscribing to Events**:
```python
# app/ml/application/event_handlers/video_handlers.py
@event_handler(VideoUploadedEvent)
async def analyze_uploaded_video(event: VideoUploadedEvent):
    """Run content moderation when video is uploaded."""
    logger.info(f"Running moderation for video {event.video_id}")
    await moderation_service.analyze(event.video_id)

# app/notifications/application/event_handlers/video_handlers.py
@event_handler(VideoUploadedEvent)
async def notify_followers(event: VideoUploadedEvent):
    """Notify followers when user uploads video."""
    logger.info(f"Notifying followers of {event.user_id}")
    followers = await get_followers(event.user_id)
    await send_notifications(followers, event.video_id)
```

---

## 🔒 Dependency Rules

### The Dependency Rule

**Core Principle**: Source code dependencies must point INWARD, toward higher-level policies.

```
     External
         ↓
  Presentation ─→ Application ─→ Domain ←─ Infrastructure
         ↓              ↓           ↑              ↑
      FastAPI       Use Cases   Entities     SQLAlchemy
      Pydantic        DTOs       Services      Boto3
```

### Rules by Layer

| Layer | Can Depend On | Cannot Depend On |
|-------|---------------|------------------|
| **Domain** | Other domain concepts | Application, Infrastructure, Presentation |
| **Application** | Domain | Infrastructure, Presentation |
| **Infrastructure** | Domain (interfaces) | Presentation |
| **Presentation** | Application, Domain (read-only) | Infrastructure |

### Violating the Rule (BAD ❌)

```python
# BAD: Domain entity depends on SQLAlchemy (infrastructure)
from sqlalchemy import Column, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Video(Base):  # ❌ Domain entity coupled to database framework
    __tablename__ = "videos"
    id = Column(String, primary_key=True)
    title = Column(String)
```

### Following the Rule (GOOD ✅)

```python
# GOOD: Domain entity is pure Python
@dataclass
class Video:  # ✅ No framework dependencies
    id: str
    title: str
    status: VideoStatus

# Infrastructure layer handles database mapping
class VideoModel(Base):
    __tablename__ = "videos"
    id = Column(String, primary_key=True)
    title = Column(String)

class SQLAlchemyVideoRepository:
    def _to_entity(self, db_video: VideoModel) -> Video:
        """Convert database model to domain entity."""
        return Video(id=db_video.id, title=db_video.title, ...)
```

---

## 🚀 Migration Strategy

### Phase 1: Create Structure ✅ COMPLETE

- [x] Create `shared/` directory with DDD layers
- [x] Create bounded context directories (auth, videos, posts, etc.)
- [x] Add `__init__.py` files with documentation

### Phase 2: Migrate Auth Module (IN PROGRESS)

**Step 1: Extract Domain Entities**
- [ ] Move `app/auth/models/user.py` → `app/auth/domain/entities/user.py`
- [ ] Convert SQLAlchemy model to pure Python dataclass
- [ ] Extract value objects (Email, Password, Role)
- [ ] Define repository interfaces

**Step 2: Create Application Layer**
- [ ] Create use cases: `RegisterUser`, `LoginUser`, `RefreshToken`
- [ ] Define DTOs for requests/responses
- [ ] Move business logic from routes to use cases

**Step 3: Implement Infrastructure**
- [ ] Create `SQLAlchemyUserRepository` implementing `IUserRepository`
- [ ] Keep SQLAlchemy models in infrastructure/persistence
- [ ] Implement JWT handling in infrastructure/security

**Step 4: Refactor Presentation**
- [ ] Update routes to call use cases
- [ ] Convert route functions to thin controllers
- [ ] Update dependency injection

**Step 5: Test & Verify**
- [ ] Run existing tests
- [ ] Add new unit tests for domain/application
- [ ] Verify no regressions

### Phase 3: Migrate Videos Module

(Same process as auth, adapted for videos)

### Phase 4: Migrate Remaining Modules

- [ ] Posts
- [ ] Livestream
- [ ] Ads
- [ ] Payments
- [ ] ML
- [ ] Notifications
- [ ] Analytics

### Phase 5: Remove Deprecated Code

- [ ] Remove `app/live/` module (deprecated)
- [ ] Remove `app/models/` (consolidated into domains)
- [ ] Clean up unused imports

---

## ✨ Best Practices

### Domain Layer

1. **Keep entities pure** - No framework dependencies
2. **Use value objects** - For descriptive concepts without identity
3. **Define invariants** - Business rules that must always be true
4. **Use domain events** - Signal important business happenings
5. **Aggregate boundaries** - Group related entities together

### Application Layer

1. **One use case = one operation** - Single responsibility
2. **Use DTOs for boundaries** - Don't leak domain entities to presentation
3. **Orchestrate, don't implement** - Coordinate domain services
4. **Handle transactions** - Application layer manages transactions
5. **Publish events** - After successful operations

### Infrastructure Layer

1. **Implement interfaces** - From domain layer
2. **Convert models** - Map between domain entities and database models
3. **Handle technical errors** - Wrap external service errors
4. **Use adapters** - Adapt external APIs to domain interfaces
5. **Keep it swappable** - Easy to replace implementations

### Presentation Layer

1. **Thin controllers** - Just call use cases and return responses
2. **Validate input** - Use Pydantic schemas
3. **Handle HTTP concerns** - Status codes, headers, etc.
4. **Convert formats** - API schemas ↔ DTOs
5. **Document APIs** - OpenAPI/Swagger

---

## 📖 Examples

### Complete Feature: Upload Video

**1. Domain Entity**
```python
# app/videos/domain/entities/video.py
@dataclass
class Video:
    id: str
    user_id: str
    title: str
    status: VideoStatus
    
    def mark_as_uploaded(self, s3_url: str) -> None:
        if self.status != VideoStatus.PENDING:
            raise ValueError("Can only mark pending videos as uploaded")
        self.s3_url = s3_url
        self.status = VideoStatus.UPLOADED
```

**2. Repository Interface**
```python
# app/videos/domain/repositories/video_repository.py
class IVideoRepository(ABC):
    @abstractmethod
    async def save(self, video: Video) -> None: ...
```

**3. Use Case**
```python
# app/videos/application/use_cases/upload_video.py
class UploadVideoUseCase:
    def __init__(self, storage: IStorage, repo: IVideoRepository):
        self._storage = storage
        self._repo = repo
    
    async def execute(self, request: UploadRequest) -> UploadResponse:
        # Create entity
        video = Video(
            id=generate_id(),
            user_id=request.user_id,
            title=request.title,
            status=VideoStatus.PENDING,
        )
        
        # Upload to S3
        s3_url = await self._storage.upload(request.file_data)
        
        # Update entity
        video.mark_as_uploaded(s3_url)
        
        # Persist
        await self._repo.save(video)
        
        return UploadResponse(video_id=video.id, status="uploaded")
```

**4. Infrastructure Implementation**
```python
# app/videos/infrastructure/persistence/video_repository.py
class SQLAlchemyVideoRepository(IVideoRepository):
    def __init__(self, session: AsyncSession):
        self._session = session
    
    async def save(self, video: Video) -> None:
        db_video = VideoModel(
            id=video.id,
            user_id=video.user_id,
            title=video.title,
            status=video.status.value,
            s3_url=video.s3_url,
        )
        self._session.add(db_video)
        await self._session.commit()
```

**5. API Route**
```python
# app/videos/presentation/api/video_routes.py
@router.post("/upload")
async def upload_video(
    request: UploadVideoSchema,
    use_case: UploadVideoUseCase = Depends(get_upload_use_case),
    user: User = Depends(get_current_user),
):
    upload_request = UploadRequest(
        user_id=user.id,
        title=request.title,
        file_data=request.file_data,
    )
    result = await use_case.execute(upload_request)
    return {"video_id": result.video_id, "status": result.status}
```

---

## 🎓 Learning Resources

### Books
- **Domain-Driven Design** by Eric Evans (the "Blue Book")
- **Implementing Domain-Driven Design** by Vaughn Vernon (the "Red Book")
- **Clean Architecture** by Robert C. Martin
- **Patterns of Enterprise Application Architecture** by Martin Fowler

### Articles
- [DDD, Hexagonal, Onion, Clean, CQRS, … How I put it all together](https://herbertograca.com/2017/11/16/explicit-architecture-01-ddd-hexagonal-onion-clean-cqrs-how-i-put-it-all-together/)
- [The Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)

### Videos
- [Domain-Driven Design: Tackling Complexity in the Heart of Software](https://www.youtube.com/watch?v=dnUFEg68ESM)

---

## 📞 Questions?

For questions or clarifications about the DDD architecture:
- Review this guide
- Check existing code examples in migrated modules
- Refer to CHANGELOG_CURSOR.md for migration progress

---

**Document Version:** 1.0.0  
**Last Updated:** October 2, 2025  
**Status:** ✅ Active - Guiding ongoing refactoring
