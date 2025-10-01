"""
Email Service for sending emails via SMTP.

Handles email verification, password reset, and notification emails.
"""

import logging
from typing import Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import ssl

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmailService:
    """Email service for sending emails via SMTP."""
    
    def __init__(self):
        self.smtp_host = settings.SMTP_HOST
        self.smtp_port = settings.SMTP_PORT
        self.smtp_user = settings.SMTP_USER
        self.smtp_password = settings.SMTP_PASSWORD
        self.smtp_tls = settings.SMTP_TLS
        self.smtp_ssl = settings.SMTP_SSL
        self.email_from = settings.EMAIL_FROM or "noreply@socialflow.com"
    
    def _create_smtp_connection(self):
        """Create SMTP connection."""
        if not self.smtp_host:
            raise ValueError("SMTP_HOST not configured")
        
        try:
            if self.smtp_ssl:
                context = ssl.create_default_context()
                server = smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, context=context)
            else:
                server = smtplib.SMTP(self.smtp_host, self.smtp_port)
                if self.smtp_tls:
                    context = ssl.create_default_context()
                    server.starttls(context=context)
            
            if self.smtp_user and self.smtp_password:
                server.login(self.smtp_user, self.smtp_password)
            
            return server
        except Exception as e:
            logger.error(f"Failed to create SMTP connection: {e}")
            raise
    
    async def send_email(
        self,
        to_email: str,
        subject: str,
        body_html: str,
        body_text: Optional[str] = None
    ) -> bool:
        """Send email via SMTP."""
        try:
            # Create message
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = self.email_from
            message["To"] = to_email
            
            # Add text and HTML parts
            if body_text:
                part1 = MIMEText(body_text, "plain")
                message.attach(part1)
            
            part2 = MIMEText(body_html, "html")
            message.attach(part2)
            
            # Send email
            with self._create_smtp_connection() as server:
                server.sendmail(self.email_from, to_email, message.as_string())
            
            logger.info(f"Email sent successfully to {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {e}")
            return False
    
    async def send_verification_email(
        self,
        to_email: str,
        verification_token: str,
        username: str
    ) -> bool:
        """Send email verification email."""
        verification_url = f"{settings.FRONTEND_URL}/verify-email?token={verification_token}"
        
        subject = "Verify your Social Flow account"
        
        body_html = f"""
        <html>
          <body>
            <h2>Welcome to Social Flow, {username}!</h2>
            <p>Thank you for registering. Please verify your email address by clicking the link below:</p>
            <p><a href="{verification_url}">Verify Email Address</a></p>
            <p>If you didn't create this account, you can safely ignore this email.</p>
            <p>This link will expire in 24 hours.</p>
            <br>
            <p>Best regards,<br>The Social Flow Team</p>
          </body>
        </html>
        """
        
        body_text = f"""
        Welcome to Social Flow, {username}!
        
        Thank you for registering. Please verify your email address by visiting:
        {verification_url}
        
        If you didn't create this account, you can safely ignore this email.
        This link will expire in 24 hours.
        
        Best regards,
        The Social Flow Team
        """
        
        return await self.send_email(to_email, subject, body_html, body_text)
    
    async def send_password_reset_email(
        self,
        to_email: str,
        reset_token: str,
        username: str
    ) -> bool:
        """Send password reset email."""
        reset_url = f"{settings.FRONTEND_URL}/reset-password?token={reset_token}"
        
        subject = "Reset your Social Flow password"
        
        body_html = f"""
        <html>
          <body>
            <h2>Password Reset Request</h2>
            <p>Hi {username},</p>
            <p>We received a request to reset your password. Click the link below to create a new password:</p>
            <p><a href="{reset_url}">Reset Password</a></p>
            <p>If you didn't request a password reset, you can safely ignore this email.</p>
            <p>This link will expire in 1 hour.</p>
            <br>
            <p>Best regards,<br>The Social Flow Team</p>
          </body>
        </html>
        """
        
        body_text = f"""
        Password Reset Request
        
        Hi {username},
        
        We received a request to reset your password. Visit the link below to create a new password:
        {reset_url}
        
        If you didn't request a password reset, you can safely ignore this email.
        This link will expire in 1 hour.
        
        Best regards,
        The Social Flow Team
        """
        
        return await self.send_email(to_email, subject, body_html, body_text)
    
    async def send_welcome_email(
        self,
        to_email: str,
        username: str
    ) -> bool:
        """Send welcome email after verification."""
        subject = "Welcome to Social Flow!"
        
        body_html = f"""
        <html>
          <body>
            <h2>Welcome to Social Flow, {username}!</h2>
            <p>Your email has been verified and your account is now active.</p>
            <p>Start exploring:</p>
            <ul>
              <li>Create your first post</li>
              <li>Upload a video</li>
              <li>Follow interesting creators</li>
              <li>Customize your profile</li>
            </ul>
            <p><a href="{settings.FRONTEND_URL}/feed">Go to Social Flow</a></p>
            <br>
            <p>Best regards,<br>The Social Flow Team</p>
          </body>
        </html>
        """
        
        body_text = f"""
        Welcome to Social Flow, {username}!
        
        Your email has been verified and your account is now active.
        
        Start exploring:
        - Create your first post
        - Upload a video
        - Follow interesting creators
        - Customize your profile
        
        Visit: {settings.FRONTEND_URL}/feed
        
        Best regards,
        The Social Flow Team
        """
        
        return await self.send_email(to_email, subject, body_html, body_text)


# Global email service instance
email_service = EmailService()
