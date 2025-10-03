"""
Social interaction endpoints for posts, comments, likes, and saves.
"""

from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import (
    get_current_user,
    get_current_user_optional,
    get_db,
    require_admin,
)
from app.infrastructure.crud import (
    post as crud_post,
    comment as crud_comment,
    like as crud_like,
    save as crud_save,
    follow as crud_follow,
)
from app.models.user import User
from app.schemas.base import PaginatedResponse, SuccessResponse
from app.schemas.social import (
    PostCreate,
    PostUpdate,
    PostResponse,
    PostDetailResponse,
    CommentCreate,
    CommentUpdate,
    CommentResponse,
    CommentDetailResponse,
    LikeCreate,
    SaveCreate,
    SaveResponse,
)

router = APIRouter()


# ==================== POST ENDPOINTS ====================

@router.post("/posts", response_model=PostResponse, status_code=status.HTTP_201_CREATED)
async def create_post(
    *,
    db: AsyncSession = Depends(get_db),
    post_in: PostCreate,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Create a new post.
    
    - **content**: Post content (1-5000 chars, required for original posts)
    - **images**: List of image URLs (max 10)
    - **visibility**: public, followers, private
    - **repost_of_id**: ID of post being reposted (optional)
    - **allow_comments**: Enable/disable comments (default: true)
    - **allow_likes**: Enable/disable likes (default: true)
    
    Automatically extracts hashtags (#tag) and mentions (@username).
    """
    # If reposting, verify original post exists
    if post_in.repost_of_id:
        original_post = await crud_post.get(db, post_in.repost_of_id)
        if not original_post:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Original post not found",
            )
    
    # Create post
    post = await crud_post.create_with_owner(
        db,
        obj_in=post_in,
        owner_id=current_user.id,
    )
    
    return post


@router.get("/posts", response_model=PaginatedResponse[PostResponse])
async def list_posts(
    *,
    db: AsyncSession = Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    user_id: Optional[UUID] = None,
    current_user: Optional[User] = Depends(get_current_user_optional),
) -> Any:
    """
    List posts.
    
    - **skip**: Number of posts to skip (pagination)
    - **limit**: Maximum number of posts to return (1-100)
    - **user_id**: Filter by user ID (optional)
    
    Returns public posts, or user-specific posts if user_id provided.
    """
    if user_id:
        # Get specific user's posts
        posts = await crud_post.get_by_user(
            db,
            user_id=user_id,
            skip=skip,
            limit=limit,
        )
        total = await crud_post.count(db, filters={"user_id": user_id})
    else:
        # Get all public posts
        posts = await crud_post.get_multi(
            db,
            skip=skip,
            limit=limit,
            filters={"visibility": "public", "status": "published"},
            order_by="created_at",
            order_desc=True,
        )
        total = await crud_post.count(
            db,
            filters={"visibility": "public", "status": "published"}
        )
    
    return {
        "items": posts,
        "total": total,
        "skip": skip,
        "limit": limit,
    }


@router.get("/posts/feed", response_model=PaginatedResponse[PostResponse])
async def get_feed(
    *,
    db: AsyncSession = Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Get personalized feed.
    
    Returns posts from users the current user follows.
    
    - **skip**: Number of posts to skip
    - **limit**: Maximum number of posts to return (1-100)
    """
    posts = await crud_post.get_feed(
        db,
        user_id=current_user.id,
        skip=skip,
        limit=limit,
    )
    
    # Count total feed posts (simplified)
    total = len(posts)  # Could be a separate count query in production
    
    return {
        "items": posts,
        "total": total,
        "skip": skip,
        "limit": limit,
    }


@router.get("/posts/trending", response_model=PaginatedResponse[PostResponse])
async def get_trending_posts(
    *,
    db: AsyncSession = Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    days: int = Query(7, ge=1, le=30),
) -> Any:
    """
    Get trending posts.
    
    Returns posts sorted by engagement (likes + comments*2) within time period.
    
    - **skip**: Number of posts to skip
    - **limit**: Maximum number of posts to return (1-100)
    - **days**: Number of days to look back (1-30)
    """
    posts = await crud_post.get_trending(
        db,
        skip=skip,
        limit=limit,
        days=days,
    )
    
    total = len(posts)
    
    return {
        "items": posts,
        "total": total,
        "skip": skip,
        "limit": limit,
    }


@router.get("/posts/{post_id}", response_model=PostDetailResponse)
async def get_post(
    *,
    db: AsyncSession = Depends(get_db),
    post_id: UUID,
    current_user: Optional[User] = Depends(get_current_user_optional),
) -> Any:
    """
    Get post details by ID.
    
    Returns detailed post information including interaction flags.
    Visibility-aware access control applied.
    """
    post = await crud_post.get(db, post_id)
    if not post:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Post not found",
        )
    
    # Check visibility
    if post.visibility == "private":
        if not current_user or current_user.id != post.owner_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Post is private",
            )
    
    elif post.visibility == "followers":
        if not current_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
            )
        
        if current_user.id != post.owner_id:
            # Check if following
            is_following = await crud_follow.is_following(
                db,
                follower_id=current_user.id,
                followed_id=post.owner_id,
            )
            if not is_following:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Post is only visible to followers",
                )
    
    # Add interaction flags if authenticated
    post_dict = post.__dict__.copy()
    if current_user:
        # Check if user liked the post
        user_like = await crud_like.get_by_user_and_post(
            db,
            user_id=current_user.id,
            post_id=post_id,
        )
        post_dict["is_liked"] = user_like is not None
        
        # Check if user saved the post
        user_save = await crud_save.get_by_user_and_post(
            db,
            user_id=current_user.id,
            post_id=post_id,
        )
        post_dict["is_saved"] = user_save is not None
    else:
        post_dict["is_liked"] = False
        post_dict["is_saved"] = False
    
    # Load original post if repost
    if post.original_post_id:
        original = await crud_post.get(db, post.original_post_id)
        post_dict["original_post"] = original
    
    return post_dict


@router.put("/posts/{post_id}", response_model=PostResponse)
async def update_post(
    *,
    db: AsyncSession = Depends(get_db),
    post_id: UUID,
    post_in: PostUpdate,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Update post.
    
    Only post owner can update.
    
    - **content**: Updated content
    - **images**: Updated image URLs
    - **visibility**: Updated visibility
    - **allow_comments**: Enable/disable comments
    - **allow_likes**: Enable/disable likes
    """
    post = await crud_post.get(db, post_id)
    if not post:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Post not found",
        )
    
    # Check ownership
    if post.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to modify this post",
        )
    
    post = await crud_post.update(db, db_obj=post, obj_in=post_in)
    return post


@router.delete("/posts/{post_id}", response_model=SuccessResponse)
async def delete_post(
    *,
    db: AsyncSession = Depends(get_db),
    post_id: UUID,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Delete post.
    
    Only post owner or admin can delete.
    """
    post = await crud_post.get(db, post_id)
    if not post:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Post not found",
        )
    
    # Check ownership or admin
    from app.models.user import UserRole
    if post.owner_id != current_user.id and current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this post",
        )
    
    await crud_post.delete(db, id=post_id)
    
    return {"success": True, "message": "Post deleted successfully"}


# ==================== COMMENT ENDPOINTS ====================

@router.post("/posts/{post_id}/comments", response_model=CommentResponse, status_code=status.HTTP_201_CREATED)
async def create_comment_on_post(
    *,
    db: AsyncSession = Depends(get_db),
    post_id: UUID,
    comment_in: CommentCreate,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Create comment on post.
    
    - **content**: Comment content (1-2000 chars)
    - **parent_comment_id**: Parent comment ID for replies (optional)
    """
    # Verify post exists
    post = await crud_post.get(db, post_id)
    if not post:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Post not found",
        )
    
    # TODO: Implement allow_comments feature
    # if not post.allow_comments:
    #     raise HTTPException(
    #         status_code=status.HTTP_400_BAD_REQUEST,
    #         detail="Comments are disabled for this post",
    #     )
    
    # Set post_id in comment data
    comment_data = comment_in.model_dump()
    comment_data["post_id"] = post_id
    
    # Create comment
    from app.schemas.social import CommentCreate as CommentCreateSchema
    comment = await crud_comment.create_with_user(
        db,
        obj_in=CommentCreateSchema(**comment_data),
        user_id=current_user.id,
    )
    
    # Increment post comment count
    await crud_post.increment_comment_count(db, post_id=post_id)
    
    return comment


@router.get("/posts/{post_id}/comments", response_model=PaginatedResponse[CommentResponse])
async def get_post_comments(
    *,
    db: AsyncSession = Depends(get_db),
    post_id: UUID,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
) -> Any:
    """
    Get comments for a post.
    
    Returns top-level comments (not replies).
    
    - **skip**: Number of comments to skip
    - **limit**: Maximum number of comments to return (1-100)
    """
    # Verify post exists
    post = await crud_post.get(db, post_id)
    if not post:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Post not found",
        )
    
    comments = await crud_comment.get_by_post(
        db,
        post_id=post_id,
        skip=skip,
        limit=limit,
    )
    
    total = await crud_comment.get_comment_count(db, post_id=post_id)
    
    return {
        "items": comments,
        "total": total,
        "skip": skip,
        "limit": limit,
    }


@router.get("/comments/{comment_id}", response_model=CommentDetailResponse)
async def get_comment(
    *,
    db: AsyncSession = Depends(get_db),
    comment_id: UUID,
    current_user: Optional[User] = Depends(get_current_user_optional),
) -> Any:
    """
    Get comment details with replies.
    
    Returns comment and its direct replies (one level deep).
    """
    comment = await crud_comment.get(db, comment_id)
    if not comment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Comment not found",
        )
    
    # Get replies
    replies = await crud_comment.get_replies(db, parent_comment_id=comment_id, limit=10)
    
    # Add interaction flags
    comment_dict = comment.__dict__.copy()
    comment_dict["replies"] = replies
    
    if current_user:
        user_like = await crud_like.get_by_user_and_comment(
            db,
            user_id=current_user.id,
            comment_id=comment_id,
        )
        comment_dict["is_liked"] = user_like is not None
    else:
        comment_dict["is_liked"] = False
    
    return comment_dict


@router.get("/comments/{comment_id}/replies", response_model=PaginatedResponse[CommentResponse])
async def get_comment_replies(
    *,
    db: AsyncSession = Depends(get_db),
    comment_id: UUID,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
) -> Any:
    """
    Get replies to a comment.
    
    - **skip**: Number of replies to skip
    - **limit**: Maximum number of replies to return (1-100)
    """
    # Verify parent comment exists
    parent = await crud_comment.get(db, comment_id)
    if not parent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Comment not found",
        )
    
    replies = await crud_comment.get_replies(
        db,
        parent_comment_id=comment_id,
        skip=skip,
        limit=limit,
    )
    
    # Simplified total
    total = len(replies)
    
    return {
        "items": replies,
        "total": total,
        "skip": skip,
        "limit": limit,
    }


@router.put("/comments/{comment_id}", response_model=CommentResponse)
async def update_comment(
    *,
    db: AsyncSession = Depends(get_db),
    comment_id: UUID,
    comment_in: CommentUpdate,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Update comment.
    
    Only comment owner can update.
    
    - **content**: Updated content
    """
    comment = await crud_comment.get(db, comment_id)
    if not comment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Comment not found",
        )
    
    # Check ownership
    if comment.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to modify this comment",
        )
    
    comment = await crud_comment.update(db, db_obj=comment, obj_in=comment_in)
    
    # Mark as edited
    comment.is_edited = True
    from datetime import datetime, timezone
    comment.edited_at = datetime.now(timezone.utc)
    db.add(comment)
    await db.commit()
    await db.refresh(comment)
    
    return comment


@router.delete("/comments/{comment_id}", response_model=SuccessResponse)
async def delete_comment(
    *,
    db: AsyncSession = Depends(get_db),
    comment_id: UUID,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Delete comment.
    
    Only comment owner or admin can delete.
    """
    comment = await crud_comment.get(db, comment_id)
    if not comment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Comment not found",
        )
    
    # Check ownership or admin
    from app.models.user import UserRole
    if comment.user_id != current_user.id and current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this comment",
        )
    
    # Decrement parent post/video comment count
    if comment.post_id:
        await crud_post.decrement_comment_count(db, post_id=comment.post_id)
    
    await crud_comment.delete(db, id=comment_id)
    
    return {"success": True, "message": "Comment deleted successfully"}


# ==================== LIKE ENDPOINTS ====================

@router.post("/posts/{post_id}/like", response_model=SuccessResponse)
async def like_post(
    *,
    db: AsyncSession = Depends(get_db),
    post_id: UUID,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Like a post.
    
    Creates a like relationship for the post.
    """
    post = await crud_post.get(db, post_id)
    if not post:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Post not found",
        )
    
    # TODO: Implement allow_likes feature
    # if not post.allow_likes:
    #     raise HTTPException(
    #         status_code=status.HTTP_400_BAD_REQUEST,
    #         detail="Likes are disabled for this post",
    #     )
    
    # Check if already liked
    existing_like = await crud_like.get_by_user_and_post(
        db,
        user_id=current_user.id,
        post_id=post_id,
    )
    
    if existing_like:
        return {"success": True, "message": "Post already liked"}
    
    # Create like
    await crud_like.create(
        db,
        obj_in=LikeCreate(
            user_id=current_user.id,
            post_id=post_id,
        ),
    )
    
    # Increment like count
    await crud_post.increment_like_count(db, post_id=post_id)
    
    return {"success": True, "message": "Post liked successfully"}


@router.delete("/posts/{post_id}/like", response_model=SuccessResponse)
async def unlike_post(
    *,
    db: AsyncSession = Depends(get_db),
    post_id: UUID,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Unlike a post.
    
    Removes the like relationship for the post.
    """
    like = await crud_like.get_by_user_and_post(
        db,
        user_id=current_user.id,
        post_id=post_id,
    )
    
    if not like:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Post not liked",
        )
    
    await crud_like.delete(db, id=like.id)
    
    # Decrement like count
    await crud_post.decrement_like_count(db, post_id=post_id)
    
    return {"success": True, "message": "Post unliked successfully"}


@router.post("/comments/{comment_id}/like", response_model=SuccessResponse)
async def like_comment(
    *,
    db: AsyncSession = Depends(get_db),
    comment_id: UUID,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Like a comment.
    
    Creates a like relationship for the comment.
    """
    comment = await crud_comment.get(db, comment_id)
    if not comment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Comment not found",
        )
    
    # Check if already liked
    existing_like = await crud_like.get_by_user_and_comment(
        db,
        user_id=current_user.id,
        comment_id=comment_id,
    )
    
    if existing_like:
        return {"success": True, "message": "Comment already liked"}
    
    # Create like
    await crud_like.create(
        db,
        obj_in=LikeCreate(
            user_id=current_user.id,
            comment_id=comment_id,
        ),
    )
    
    # Increment like count
    await crud_comment.increment_like_count(db, comment_id=comment_id)
    
    return {"success": True, "message": "Comment liked successfully"}


@router.delete("/comments/{comment_id}/like", response_model=SuccessResponse)
async def unlike_comment(
    *,
    db: AsyncSession = Depends(get_db),
    comment_id: UUID,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Unlike a comment.
    
    Removes the like relationship for the comment.
    """
    like = await crud_like.get_by_user_and_comment(
        db,
        user_id=current_user.id,
        comment_id=comment_id,
    )
    
    if not like:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Comment not liked",
        )
    
    await crud_like.delete(db, id=like.id)
    
    # Decrement like count
    await crud_comment.decrement_like_count(db, comment_id=comment_id)
    
    return {"success": True, "message": "Comment unliked successfully"}


# ==================== SAVE ENDPOINTS ====================

@router.post("/posts/{post_id}/save", response_model=SuccessResponse)
async def save_post(
    *,
    db: AsyncSession = Depends(get_db),
    post_id: UUID,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Save/bookmark a post.
    
    Adds post to user's saved collection.
    """
    post = await crud_post.get(db, post_id)
    if not post:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Post not found",
        )
    
    # Check if already saved
    existing_save = await crud_save.get_by_user_and_post(
        db,
        user_id=current_user.id,
        post_id=post_id,
    )
    
    if existing_save:
        return {"success": True, "message": "Post already saved"}
    
    # Create save
    await crud_save.create_with_user(
        db,
        obj_in=SaveCreate(
            post_id=post_id,
        ),
        user_id=current_user.id,
    )
    
    return {"success": True, "message": "Post saved successfully"}


@router.delete("/posts/{post_id}/save", response_model=SuccessResponse)
async def unsave_post(
    *,
    db: AsyncSession = Depends(get_db),
    post_id: UUID,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Unsave/unbookmark a post.
    
    Removes post from user's saved collection.
    """
    save = await crud_save.get_by_user_and_post(
        db,
        user_id=current_user.id,
        post_id=post_id,
    )
    
    if not save:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Post not saved",
        )
    
    await crud_save.delete(db, id=save.id)
    
    return {"success": True, "message": "Post unsaved successfully"}


@router.get("/saves", response_model=PaginatedResponse[SaveResponse])
async def get_saved_content(
    *,
    db: AsyncSession = Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Get user's saved content.
    
    Returns all posts and videos saved by the user.
    
    - **skip**: Number of items to skip
    - **limit**: Maximum number of items to return (1-100)
    """
    saves = await crud_save.get_saved_by_user(
        db,
        user_id=current_user.id,
        skip=skip,
        limit=limit,
    )
    
    total = await crud_save.count(db, filters={"user_id": current_user.id})
    
    return {
        "items": saves,
        "total": total,
        "skip": skip,
        "limit": limit,
    }


# ==================== ADMIN ENDPOINTS ====================

@router.post("/posts/{post_id}/admin/flag", response_model=SuccessResponse)
async def admin_flag_post(
    *,
    db: AsyncSession = Depends(get_db),
    post_id: UUID,
    admin: User = Depends(require_admin),
) -> Any:
    """
    Admin: Flag a post for review.
    
    Marks post as flagged for content moderation.
    """
    post = await crud_post.get(db, post_id)
    if not post:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Post not found",
        )
    
    post.is_flagged = True
    post.status = "flagged"
    db.add(post)
    await db.commit()
    
    return {"success": True, "message": "Post flagged for review"}


@router.post("/posts/{post_id}/admin/remove", response_model=SuccessResponse)
async def admin_remove_post(
    *,
    db: AsyncSession = Depends(get_db),
    post_id: UUID,
    admin: User = Depends(require_admin),
) -> Any:
    """
    Admin: Remove a post.
    
    Sets post status to removed (content policy violation).
    """
    post = await crud_post.get(db, post_id)
    if not post:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Post not found",
        )
    
    post.status = "removed"
    db.add(post)
    await db.commit()
    
    return {"success": True, "message": "Post removed"}


@router.post("/comments/{comment_id}/admin/remove", response_model=SuccessResponse)
async def admin_remove_comment(
    *,
    db: AsyncSession = Depends(get_db),
    comment_id: UUID,
    admin: User = Depends(require_admin),
) -> Any:
    """
    Admin: Remove a comment.
    
    Sets comment status to removed (content policy violation).
    """
    comment = await crud_comment.get(db, comment_id)
    if not comment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Comment not found",
        )
    
    comment.status = "removed"
    db.add(comment)
    await db.commit()
    
    return {"success": True, "message": "Comment removed"}
