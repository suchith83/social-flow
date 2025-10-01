"""
Development seed data.

This script populates the database with sample data for development and testing.
"""

import asyncio
import uuid
from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.auth.models.user import User
from app.videos.models.video import Video, VideoStatus, VideoVisibility
from app.posts.models.post import Post
from app.core.security import get_password_hash


async def seed_users(db: AsyncSession) -> list[User]:
    """Seed sample users."""
    users_data = [
        {
            "username": "johndoe",
            "email": "john@example.com",
            "display_name": "John Doe",
            "bio": "Content creator and tech enthusiast",
            "password_hash": get_password_hash("password123"),
            "is_verified": True,
        },
        {
            "username": "janedoe",
            "email": "jane@example.com",
            "display_name": "Jane Doe",
            "bio": "Video editor and social media manager",
            "password_hash": get_password_hash("password123"),
            "is_verified": True,
        },
        {
            "username": "creator",
            "email": "creator@example.com",
            "display_name": "Pro Creator",
            "bio": "Professional content creator",
            "password_hash": get_password_hash("password123"),
            "is_verified": True,
        },
    ]

    users = []
    for user_data in users_data:
        user = User(**user_data)
        db.add(user)
        users.append(user)

    await db.commit()
    for user in users:
        await db.refresh(user)

    return users


async def seed_videos(db: AsyncSession, users: list[User]) -> list[Video]:
    """Seed sample videos."""
    videos_data = [
        {
            "title": "Getting Started with FastAPI",
            "description": "Learn how to build APIs with FastAPI",
            "filename": "fastapi-tutorial.mp4",
            "file_size": 104857600,  # 100MB
            "duration": 1800.0,  # 30 minutes
            "resolution": "1920x1080",
            "bitrate": 5000000,
            "codec": "H.264",
            "s3_key": f"videos/{users[0].id}/{uuid.uuid4()}/fastapi-tutorial.mp4",
            "s3_bucket": "social-flow-videos",
            "status": VideoStatus.PROCESSED,
            "visibility": VideoVisibility.PUBLIC,
            "user_id": users[0].id,
            "views_count": 1500,
            "likes_count": 120,
            "comments_count": 45,
        },
        {
            "title": "Python Best Practices",
            "description": "Writing clean and maintainable Python code",
            "filename": "python-best-practices.mp4",
            "file_size": 78643200,  # 75MB
            "duration": 2400.0,  # 40 minutes
            "resolution": "1920x1080",
            "bitrate": 4000000,
            "codec": "H.264",
            "s3_key": f"videos/{users[1].id}/{uuid.uuid4()}/python-best-practices.mp4",
            "s3_bucket": "social-flow-videos",
            "status": VideoStatus.PROCESSED,
            "visibility": VideoVisibility.PUBLIC,
            "user_id": users[1].id,
            "views_count": 2300,
            "likes_count": 200,
            "comments_count": 78,
        },
        {
            "title": "Machine Learning Basics",
            "description": "Introduction to machine learning concepts",
            "filename": "ml-basics.mp4",
            "file_size": 157286400,  # 150MB
            "duration": 3600.0,  # 1 hour
            "resolution": "1920x1080",
            "bitrate": 6000000,
            "codec": "H.264",
            "s3_key": f"videos/{users[2].id}/{uuid.uuid4()}/ml-basics.mp4",
            "s3_bucket": "social-flow-videos",
            "status": VideoStatus.PROCESSED,
            "visibility": VideoVisibility.PUBLIC,
            "user_id": users[2].id,
            "views_count": 3200,
            "likes_count": 350,
            "comments_count": 120,
        },
    ]

    videos = []
    for video_data in videos_data:
        video = Video(**video_data)
        db.add(video)
        videos.append(video)

    await db.commit()
    for video in videos:
        await db.refresh(video)

    return videos


async def seed_posts(db: AsyncSession, users: list[User], videos: list[Video]) -> list[Post]:
    """Seed sample posts."""
    posts_data = [
        {
            "content": "Just uploaded a new video about FastAPI! Check it out and let me know what you think. #FastAPI #Python #WebDevelopment",
            "user_id": users[0].id,
            "video_id": videos[0].id,
            "likes_count": 25,
            "reposts_count": 5,
            "comments_count": 12,
        },
        {
            "content": "Working on some exciting Python projects. The best practices video is now live! #Python #Programming",
            "user_id": users[1].id,
            "video_id": videos[1].id,
            "likes_count": 40,
            "reposts_count": 8,
            "comments_count": 20,
        },
        {
            "content": "Machine learning is revolutionizing everything. Here's my latest video explaining the basics. #MachineLearning #AI",
            "user_id": users[2].id,
            "video_id": videos[2].id,
            "likes_count": 60,
            "reposts_count": 15,
            "comments_count": 35,
        },
        {
            "content": "What are your favorite Python libraries for web development? I'm always looking for new tools to try!",
            "user_id": users[0].id,
            "likes_count": 15,
            "reposts_count": 2,
            "comments_count": 8,
        },
    ]

    posts = []
    for post_data in posts_data:
        post = Post(**post_data)
        db.add(post)
        posts.append(post)

    await db.commit()
    for post in posts:
        await db.refresh(post)

    return posts


async def main():
    """Main seed function."""
    async for db in get_db():
        try:
            print("Seeding development data...")

            # Seed users
            users = await seed_users(db)
            print(f"Created {len(users)} users")

            # Seed videos
            videos = await seed_videos(db, users)
            print(f"Created {len(videos)} videos")

            # Seed posts
            posts = await seed_posts(db, users, videos)
            print(f"Created {len(posts)} posts")

            print("Seeding completed successfully!")

        except Exception as e:
            print(f"Error seeding data: {e}")
            await db.rollback()
        finally:
            await db.close()


if __name__ == "__main__":
    asyncio.run(main())