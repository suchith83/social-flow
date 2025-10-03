"""
User Mapper - Auth Infrastructure

Maps between domain entities (UserEntity) and database models (UserModel).
Handles the impedance mismatch between domain and persistence layers.
"""

from typing import Optional
from datetime import datetime

from app.auth.domain.entities import UserEntity
from app.auth.domain.value_objects import (
    Email,
    Username,
    AccountStatus,
    PrivacyLevel,
    SuspensionDetails,
    BanDetails,
)
from app.shared.domain.value_objects import UserRole
from app.auth.infrastructure.persistence.models import UserModel


class UserMapper:
    """
    Maps between UserEntity (domain) and UserModel (database).
    
    Handles conversion of value objects to primitive types and vice versa.
    """
    
    @staticmethod
    def to_domain(model: UserModel) -> UserEntity:
        """
        Convert database model to domain entity.
        
        Args:
            model: UserModel from database
            
        Returns:
            UserEntity domain object
        """
        # Create value objects from primitives
        email = Email(model.email)
        username = Username(model.username)
        
        # Parse account status
        account_status = AccountStatus(model.account_status)
        
        # Parse privacy level
        privacy_level = PrivacyLevel(model.privacy_level)
        
        # Parse role
        role = UserRole(model.role)
        
        # Create suspension details if suspended
        suspension_details: Optional[SuspensionDetails] = None
        if model.suspension_reason and model.suspended_at:
            suspension_details = SuspensionDetails(
                reason=model.suspension_reason,
                suspended_at=model.suspended_at,
                ends_at=model.suspension_ends_at,
            )
        
        # Create ban details if banned
        ban_details: Optional[BanDetails] = None
        if model.ban_reason and model.banned_at:
            ban_details = BanDetails(
                reason=model.ban_reason,
                banned_at=model.banned_at,
            )
        
        # Create entity
        entity = UserEntity(
            username=username,
            email=email,
            password_hash=model.password_hash,
            display_name=model.display_name,
            id=model.id,
            role=role,
        )
        
        # Set private attributes directly (bypassing business logic)
        # This is OK because we're reconstructing from trusted database state
        entity._bio = model.bio
        entity._avatar_url = model.avatar_url
        entity._website = model.website
        entity._location = model.location
        entity._account_status = account_status
        entity._privacy_level = privacy_level
        entity._suspension_details = suspension_details
        entity._ban_details = ban_details
        entity._followers_count = model.followers_count
        entity._following_count = model.following_count
        entity._posts_count = model.posts_count
        entity._videos_count = model.videos_count
        entity._total_views = model.total_views
        entity._total_likes = model.total_likes
        entity._email_notifications = model.email_notifications
        entity._push_notifications = model.push_notifications
        entity._last_login_at = model.last_login_at
        
        # Set timestamps from database
        entity._created_at = model.created_at
        entity._updated_at = model.updated_at
        entity._version = model.version
        
        # Clear any domain events from reconstruction
        entity.clear_events()
        
        return entity
    
    @staticmethod
    def to_model(entity: UserEntity, existing_model: Optional[UserModel] = None) -> UserModel:
        """
        Convert domain entity to database model.
        
        Args:
            entity: UserEntity domain object
            existing_model: Optional existing model to update
            
        Returns:
            UserModel for database persistence
        """
        # Use existing model or create new one
        model = existing_model or UserModel()
        
        # Set ID (only for new models)
        if not existing_model:
            model.id = entity.id
            model.created_at = entity.created_at
        
        # Basic information
        model.username = entity.username.value
        model.email = entity.email.value
        model.password_hash = entity.password_hash
        model.display_name = entity.display_name
        model.bio = entity.bio
        model.avatar_url = entity.avatar_url
        model.website = entity.website
        model.location = entity.location
        
        # Account status
        model.account_status = entity.account_status.value
        model.privacy_level = entity.privacy_level.value
        model.role = entity.role.value
        
        # Suspension details
        if entity.suspension_details:
            model.suspension_reason = entity.suspension_details.reason
            model.suspended_at = entity.suspension_details.suspended_at
            model.suspension_ends_at = entity.suspension_details.ends_at
        else:
            model.suspension_reason = None
            model.suspended_at = None
            model.suspension_ends_at = None
        
        # Ban details
        if entity.ban_details:
            model.ban_reason = entity.ban_details.reason
            model.banned_at = entity.ban_details.banned_at
        else:
            model.ban_reason = None
            model.banned_at = None
        
        # Social metrics
        model.followers_count = entity.followers_count
        model.following_count = entity.following_count
        model.posts_count = entity.posts_count
        model.videos_count = entity.videos_count
        model.total_views = entity.total_views
        model.total_likes = entity.total_likes
        
        # Preferences
        model.email_notifications = entity.email_notifications
        model.push_notifications = entity.push_notifications
        
        # Timestamps
        model.updated_at = entity.updated_at
        model.last_login_at = entity.last_login_at
        
        # Version for optimistic locking
        model.version = entity.version
        
        return model
    
    @staticmethod
    def update_model_from_entity(entity: UserEntity, model: UserModel) -> None:
        """
        Update an existing model from entity (in-place).
        
        Args:
            entity: UserEntity with updates
            model: UserModel to update
        """
        # This is equivalent to to_model with existing_model
        # but more explicit for update operations
        UserMapper.to_model(entity, existing_model=model)
