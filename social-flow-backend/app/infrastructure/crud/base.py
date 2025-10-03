"""
Base CRUD operations for SQLAlchemy 2.0 async models.

This module provides generic CRUD operations that can be extended
for specific models. All operations use async/await patterns.
"""

from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union
from uuid import UUID

from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from sqlalchemy import select, func, delete as sql_delete, update as sql_update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import Base

# Type variables for generic CRUD operations
ModelType = TypeVar("ModelType", bound=Base)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """
    Base class for CRUD operations.
    
    Provides standard database operations:
    - get: Get single record by ID
    - get_multi: Get multiple records with pagination
    - create: Create new record
    - update: Update existing record
    - delete: Hard delete record
    - soft_delete: Soft delete record (if model supports it)
    - count: Count records
    - exists: Check if record exists
    """

    def __init__(self, model: Type[ModelType]):
        """
        Initialize CRUD object with SQLAlchemy model.
        
        Args:
            model: SQLAlchemy model class
        """
        self.model = model

    async def get(
        self,
        db: AsyncSession,
        id: Union[UUID, int],
        *,
        relationships: Optional[List[str]] = None,
    ) -> Optional[ModelType]:
        """
        Get a single record by ID.
        
        Args:
            db: Database session
            id: Record ID
            relationships: List of relationship names to eagerly load
            
        Returns:
            Model instance or None if not found
        """
        query = select(self.model).where(self.model.id == id)
        
        # Eagerly load relationships if specified
        if relationships:
            for rel in relationships:
                query = query.options(selectinload(getattr(self.model, rel)))
        
        result = await db.execute(query)
        return result.scalar_one_or_none()

    async def get_by_field(
        self,
        db: AsyncSession,
        field_name: str,
        field_value: Any,
        *,
        relationships: Optional[List[str]] = None,
    ) -> Optional[ModelType]:
        """
        Get a single record by a specific field.
        
        Args:
            db: Database session
            field_name: Name of the field to filter by
            field_value: Value to match
            relationships: List of relationship names to eagerly load
            
        Returns:
            Model instance or None if not found
        """
        query = select(self.model).where(getattr(self.model, field_name) == field_value)
        
        if relationships:
            for rel in relationships:
                query = query.options(selectinload(getattr(self.model, rel)))
        
        result = await db.execute(query)
        return result.scalar_one_or_none()

    async def get_multi(
        self,
        db: AsyncSession,
        *,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        order_desc: bool = False,
        relationships: Optional[List[str]] = None,
    ) -> List[ModelType]:
        """
        Get multiple records with pagination and filtering.
        
        Args:
            db: Database session
            skip: Number of records to skip
            limit: Maximum number of records to return
            filters: Dictionary of field:value pairs to filter by
            order_by: Field name to order by
            order_desc: Whether to order descending
            relationships: List of relationship names to eagerly load
            
        Returns:
            List of model instances
        """
        query = select(self.model)
        
        # Apply filters
        if filters:
            for field, value in filters.items():
                if hasattr(self.model, field):
                    query = query.where(getattr(self.model, field) == value)
        
        # Apply ordering
        if order_by and hasattr(self.model, order_by):
            order_field = getattr(self.model, order_by)
            query = query.order_by(order_field.desc() if order_desc else order_field)
        
        # Apply pagination
        query = query.offset(skip).limit(limit)
        
        # Eagerly load relationships
        if relationships:
            for rel in relationships:
                query = query.options(selectinload(getattr(self.model, rel)))
        
        result = await db.execute(query)
        return list(result.scalars().all())

    async def create(
        self,
        db: AsyncSession,
        *,
        obj_in: CreateSchemaType,
        commit: bool = True,
    ) -> ModelType:
        """
        Create a new record.
        
        Args:
            db: Database session
            obj_in: Pydantic schema with data to create
            commit: Whether to commit the transaction
            
        Returns:
            Created model instance
        """
        obj_in_data = jsonable_encoder(obj_in)
        db_obj = self.model(**obj_in_data)
        db.add(db_obj)
        
        if commit:
            await db.commit()
            await db.refresh(db_obj)
        else:
            await db.flush()
        
        return db_obj

    async def create_multi(
        self,
        db: AsyncSession,
        *,
        objs_in: List[CreateSchemaType],
        commit: bool = True,
    ) -> List[ModelType]:
        """
        Create multiple records.
        
        Args:
            db: Database session
            objs_in: List of Pydantic schemas with data to create
            commit: Whether to commit the transaction
            
        Returns:
            List of created model instances
        """
        db_objs = []
        for obj_in in objs_in:
            obj_in_data = jsonable_encoder(obj_in)
            db_obj = self.model(**obj_in_data)
            db.add(db_obj)
            db_objs.append(db_obj)
        
        if commit:
            await db.commit()
            for db_obj in db_objs:
                await db.refresh(db_obj)
        else:
            await db.flush()
        
        return db_objs

    async def update(
        self,
        db: AsyncSession,
        *,
        db_obj: ModelType,
        obj_in: Union[UpdateSchemaType, Dict[str, Any]],
        commit: bool = True,
    ) -> ModelType:
        """
        Update an existing record.
        
        Args:
            db: Database session
            db_obj: Existing model instance
            obj_in: Pydantic schema or dict with update data
            commit: Whether to commit the transaction
            
        Returns:
            Updated model instance
        """
        obj_data = jsonable_encoder(db_obj)
        
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.model_dump(exclude_unset=True)
        
        for field in obj_data:
            if field in update_data:
                setattr(db_obj, field, update_data[field])
        
        db.add(db_obj)
        
        if commit:
            await db.commit()
            await db.refresh(db_obj)
        else:
            await db.flush()
        
        return db_obj

    async def update_by_id(
        self,
        db: AsyncSession,
        *,
        id: Union[UUID, int],
        obj_in: Union[UpdateSchemaType, Dict[str, Any]],
        commit: bool = True,
    ) -> Optional[ModelType]:
        """
        Update a record by ID.
        
        Args:
            db: Database session
            id: Record ID
            obj_in: Pydantic schema or dict with update data
            commit: Whether to commit the transaction
            
        Returns:
            Updated model instance or None if not found
        """
        db_obj = await self.get(db, id)
        if not db_obj:
            return None
        
        return await self.update(db, db_obj=db_obj, obj_in=obj_in, commit=commit)

    async def delete(
        self,
        db: AsyncSession,
        *,
        id: Union[UUID, int],
        commit: bool = True,
    ) -> bool:
        """
        Hard delete a record.
        
        Args:
            db: Database session
            id: Record ID
            commit: Whether to commit the transaction
            
        Returns:
            True if deleted, False if not found
        """
        query = sql_delete(self.model).where(self.model.id == id)
        result = await db.execute(query)
        
        if commit:
            await db.commit()
        
        return result.rowcount > 0

    async def soft_delete(
        self,
        db: AsyncSession,
        *,
        id: Union[UUID, int],
        commit: bool = True,
    ) -> Optional[ModelType]:
        """
        Soft delete a record (if model supports it).
        
        Args:
            db: Database session
            id: Record ID
            commit: Whether to commit the transaction
            
        Returns:
            Updated model instance or None if not found
        """
        if not hasattr(self.model, "is_deleted"):
            raise NotImplementedError(f"{self.model.__name__} does not support soft delete")
        
        db_obj = await self.get(db, id)
        if not db_obj:
            return None
        
        db_obj.is_deleted = True
        db.add(db_obj)
        
        if commit:
            await db.commit()
            await db.refresh(db_obj)
        else:
            await db.flush()
        
        return db_obj

    async def restore(
        self,
        db: AsyncSession,
        *,
        id: Union[UUID, int],
        commit: bool = True,
    ) -> Optional[ModelType]:
        """
        Restore a soft-deleted record.
        
        Args:
            db: Database session
            id: Record ID
            commit: Whether to commit the transaction
            
        Returns:
            Restored model instance or None if not found
        """
        if not hasattr(self.model, "is_deleted"):
            raise NotImplementedError(f"{self.model.__name__} does not support soft delete")
        
        db_obj = await self.get(db, id)
        if not db_obj:
            return None
        
        db_obj.is_deleted = False
        db.add(db_obj)
        
        if commit:
            await db.commit()
            await db.refresh(db_obj)
        else:
            await db.flush()
        
        return db_obj

    async def count(
        self,
        db: AsyncSession,
        *,
        filters: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Count records with optional filtering.
        
        Args:
            db: Database session
            filters: Dictionary of field:value pairs to filter by
            
        Returns:
            Number of matching records
        """
        query = select(func.count()).select_from(self.model)
        
        if filters:
            for field, value in filters.items():
                if hasattr(self.model, field):
                    query = query.where(getattr(self.model, field) == value)
        
        result = await db.execute(query)
        return result.scalar_one()

    async def exists(
        self,
        db: AsyncSession,
        *,
        id: Union[UUID, int],
    ) -> bool:
        """
        Check if a record exists.
        
        Args:
            db: Database session
            id: Record ID
            
        Returns:
            True if exists, False otherwise
        """
        query = select(func.count()).select_from(self.model).where(self.model.id == id)
        result = await db.execute(query)
        return result.scalar_one() > 0

    async def exists_by_field(
        self,
        db: AsyncSession,
        field_name: str,
        field_value: Any,
    ) -> bool:
        """
        Check if a record exists by a specific field.
        
        Args:
            db: Database session
            field_name: Name of the field to check
            field_value: Value to match
            
        Returns:
            True if exists, False otherwise
        """
        query = select(func.count()).select_from(self.model).where(
            getattr(self.model, field_name) == field_value
        )
        result = await db.execute(query)
        return result.scalar_one() > 0
