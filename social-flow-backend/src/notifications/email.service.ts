import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import * as nodemailer from 'nodemailer';
import { LoggerService } from '../shared/logger/logger.service';

export interface EmailTemplate {
  subject: string;
  html: string;
  text: string;
}

export interface NotificationEmailData {
  title: string;
  message: string;
  actionUrl?: string;
  imageUrl?: string;
}

@Injectable()
export class EmailService {
  private transporter: nodemailer.Transporter;

  constructor(
    private configService: ConfigService,
    private logger: LoggerService,
  ) {
    this.transporter = nodemailer.createTransporter({
      host: this.configService.get('app.emailHost'),
      port: this.configService.get('app.emailPort'),
      secure: false,
      auth: {
        user: this.configService.get('app.emailUser'),
        pass: this.configService.get('app.emailPassword'),
      },
    });
  }

  async sendEmail(to: string, subject: string, html: string, text?: string): Promise<void> {
    try {
      await this.transporter.sendMail({
        from: this.configService.get('app.emailFrom'),
        to,
        subject,
        html,
        text: text || this.htmlToText(html),
      });

      this.logger.logBusiness('email_sent', undefined, {
        to,
        subject,
      });
    } catch (error) {
      this.logger.logError(error, 'EmailService.sendEmail', {
        to,
        subject,
      });
      throw error;
    }
  }

  async sendWelcomeEmail(email: string, name: string): Promise<void> {
    const subject = 'Welcome to Social Flow!';
    const html = this.getWelcomeEmailTemplate(name);
    const text = this.getWelcomeEmailTextTemplate(name);

    await this.sendEmail(email, subject, html, text);
  }

  async sendVerificationEmail(email: string, token: string): Promise<void> {
    const subject = 'Verify your email address';
    const html = this.getVerificationEmailTemplate(token);
    const text = this.getVerificationEmailTextTemplate(token);

    await this.sendEmail(email, subject, html, text);
  }

  async sendPasswordResetEmail(email: string, token: string): Promise<void> {
    const subject = 'Reset your password';
    const html = this.getPasswordResetEmailTemplate(token);
    const text = this.getPasswordResetEmailTextTemplate(token);

    await this.sendEmail(email, subject, html, text);
  }

  async sendNotificationEmail(email: string, data: NotificationEmailData): Promise<void> {
    const subject = data.title;
    const html = this.getNotificationEmailTemplate(data);
    const text = this.getNotificationEmailTextTemplate(data);

    await this.sendEmail(email, subject, html, text);
  }

  async sendPaymentReceiptEmail(email: string, paymentData: any): Promise<void> {
    const subject = 'Payment Receipt';
    const html = this.getPaymentReceiptEmailTemplate(paymentData);
    const text = this.getPaymentReceiptEmailTextTemplate(paymentData);

    await this.sendEmail(email, subject, html, text);
  }

  async sendSubscriptionConfirmationEmail(email: string, subscriptionData: any): Promise<void> {
    const subject = 'Subscription Confirmation';
    const html = this.getSubscriptionConfirmationEmailTemplate(subscriptionData);
    const text = this.getSubscriptionConfirmationEmailTextTemplate(subscriptionData);

    await this.sendEmail(email, subject, html, text);
  }

  private getWelcomeEmailTemplate(name: string): string {
    return `
      <!DOCTYPE html>
      <html>
      <head>
        <meta charset="utf-8">
        <title>Welcome to Social Flow</title>
        <style>
          body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
          .container { max-width: 600px; margin: 0 auto; padding: 20px; }
          .header { background: #007bff; color: white; padding: 20px; text-align: center; }
          .content { padding: 20px; background: #f9f9f9; }
          .button { display: inline-block; padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; }
        </style>
      </head>
      <body>
        <div class="container">
          <div class="header">
            <h1>Welcome to Social Flow!</h1>
          </div>
          <div class="content">
            <h2>Hello ${name}!</h2>
            <p>Welcome to Social Flow, the ultimate social media platform for creators and viewers alike.</p>
            <p>Get started by:</p>
            <ul>
              <li>Uploading your first video</li>
              <li>Following your favorite creators</li>
              <li>Creating engaging posts</li>
              <li>Exploring trending content</li>
            </ul>
            <p style="text-align: center;">
              <a href="${process.env.FRONTEND_URL}/dashboard" class="button">Get Started</a>
            </p>
            <p>If you have any questions, feel free to reach out to our support team.</p>
            <p>Best regards,<br>The Social Flow Team</p>
          </div>
        </div>
      </body>
      </html>
    `;
  }

  private getWelcomeEmailTextTemplate(name: string): string {
    return `
      Welcome to Social Flow!
      
      Hello ${name}!
      
      Welcome to Social Flow, the ultimate social media platform for creators and viewers alike.
      
      Get started by:
      - Uploading your first video
      - Following your favorite creators
      - Creating engaging posts
      - Exploring trending content
      
      Visit: ${process.env.FRONTEND_URL}/dashboard
      
      If you have any questions, feel free to reach out to our support team.
      
      Best regards,
      The Social Flow Team
    `;
  }

  private getVerificationEmailTemplate(token: string): string {
    return `
      <!DOCTYPE html>
      <html>
      <head>
        <meta charset="utf-8">
        <title>Verify your email</title>
        <style>
          body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
          .container { max-width: 600px; margin: 0 auto; padding: 20px; }
          .header { background: #007bff; color: white; padding: 20px; text-align: center; }
          .content { padding: 20px; background: #f9f9f9; }
          .button { display: inline-block; padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; }
        </style>
      </head>
      <body>
        <div class="container">
          <div class="header">
            <h1>Verify your email address</h1>
          </div>
          <div class="content">
            <p>Please click the button below to verify your email address:</p>
            <p style="text-align: center;">
              <a href="${process.env.FRONTEND_URL}/verify-email?token=${token}" class="button">Verify Email</a>
            </p>
            <p>If the button doesn't work, you can also copy and paste this link into your browser:</p>
            <p>${process.env.FRONTEND_URL}/verify-email?token=${token}</p>
            <p>This link will expire in 24 hours.</p>
            <p>Best regards,<br>The Social Flow Team</p>
          </div>
        </div>
      </body>
      </html>
    `;
  }

  private getVerificationEmailTextTemplate(token: string): string {
    return `
      Verify your email address
      
      Please click the link below to verify your email address:
      
      ${process.env.FRONTEND_URL}/verify-email?token=${token}
      
      This link will expire in 24 hours.
      
      Best regards,
      The Social Flow Team
    `;
  }

  private getPasswordResetEmailTemplate(token: string): string {
    return `
      <!DOCTYPE html>
      <html>
      <head>
        <meta charset="utf-8">
        <title>Reset your password</title>
        <style>
          body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
          .container { max-width: 600px; margin: 0 auto; padding: 20px; }
          .header { background: #dc3545; color: white; padding: 20px; text-align: center; }
          .content { padding: 20px; background: #f9f9f9; }
          .button { display: inline-block; padding: 10px 20px; background: #dc3545; color: white; text-decoration: none; border-radius: 5px; }
        </style>
      </head>
      <body>
        <div class="container">
          <div class="header">
            <h1>Reset your password</h1>
          </div>
          <div class="content">
            <p>You requested to reset your password. Click the button below to create a new password:</p>
            <p style="text-align: center;">
              <a href="${process.env.FRONTEND_URL}/reset-password?token=${token}" class="button">Reset Password</a>
            </p>
            <p>If the button doesn't work, you can also copy and paste this link into your browser:</p>
            <p>${process.env.FRONTEND_URL}/reset-password?token=${token}</p>
            <p>This link will expire in 24 hours.</p>
            <p>If you didn't request this password reset, please ignore this email.</p>
            <p>Best regards,<br>The Social Flow Team</p>
          </div>
        </div>
      </body>
      </html>
    `;
  }

  private getPasswordResetEmailTextTemplate(token: string): string {
    return `
      Reset your password
      
      You requested to reset your password. Click the link below to create a new password:
      
      ${process.env.FRONTEND_URL}/reset-password?token=${token}
      
      This link will expire in 24 hours.
      
      If you didn't request this password reset, please ignore this email.
      
      Best regards,
      The Social Flow Team
    `;
  }

  private getNotificationEmailTemplate(data: NotificationEmailData): string {
    return `
      <!DOCTYPE html>
      <html>
      <head>
        <meta charset="utf-8">
        <title>${data.title}</title>
        <style>
          body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
          .container { max-width: 600px; margin: 0 auto; padding: 20px; }
          .header { background: #007bff; color: white; padding: 20px; text-align: center; }
          .content { padding: 20px; background: #f9f9f9; }
          .button { display: inline-block; padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; }
        </style>
      </head>
      <body>
        <div class="container">
          <div class="header">
            <h1>${data.title}</h1>
          </div>
          <div class="content">
            <p>${data.message}</p>
            ${data.actionUrl ? `
              <p style="text-align: center;">
                <a href="${data.actionUrl}" class="button">View Details</a>
              </p>
            ` : ''}
            <p>Best regards,<br>The Social Flow Team</p>
          </div>
        </div>
      </body>
      </html>
    `;
  }

  private getNotificationEmailTextTemplate(data: NotificationEmailData): string {
    return `
      ${data.title}
      
      ${data.message}
      
      ${data.actionUrl ? `View Details: ${data.actionUrl}` : ''}
      
      Best regards,
      The Social Flow Team
    `;
  }

  private getPaymentReceiptEmailTemplate(paymentData: any): string {
    return `
      <!DOCTYPE html>
      <html>
      <head>
        <meta charset="utf-8">
        <title>Payment Receipt</title>
        <style>
          body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
          .container { max-width: 600px; margin: 0 auto; padding: 20px; }
          .header { background: #28a745; color: white; padding: 20px; text-align: center; }
          .content { padding: 20px; background: #f9f9f9; }
          .receipt { background: white; padding: 20px; border-radius: 5px; margin: 20px 0; }
        </style>
      </head>
      <body>
        <div class="container">
          <div class="header">
            <h1>Payment Receipt</h1>
          </div>
          <div class="content">
            <div class="receipt">
              <h3>Payment Details</h3>
              <p><strong>Amount:</strong> ${paymentData.currency.toUpperCase()} ${(paymentData.amount / 100).toFixed(2)}</p>
              <p><strong>Date:</strong> ${new Date(paymentData.createdAt).toLocaleDateString()}</p>
              <p><strong>Status:</strong> ${paymentData.status}</p>
              <p><strong>Description:</strong> ${paymentData.description}</p>
            </div>
            <p>Thank you for your payment!</p>
            <p>Best regards,<br>The Social Flow Team</p>
          </div>
        </div>
      </body>
      </html>
    `;
  }

  private getPaymentReceiptEmailTextTemplate(paymentData: any): string {
    return `
      Payment Receipt
      
      Payment Details:
      Amount: ${paymentData.currency.toUpperCase()} ${(paymentData.amount / 100).toFixed(2)}
      Date: ${new Date(paymentData.createdAt).toLocaleDateString()}
      Status: ${paymentData.status}
      Description: ${paymentData.description}
      
      Thank you for your payment!
      
      Best regards,
      The Social Flow Team
    `;
  }

  private getSubscriptionConfirmationEmailTemplate(subscriptionData: any): string {
    return `
      <!DOCTYPE html>
      <html>
      <head>
        <meta charset="utf-8">
        <title>Subscription Confirmation</title>
        <style>
          body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
          .container { max-width: 600px; margin: 0 auto; padding: 20px; }
          .header { background: #007bff; color: white; padding: 20px; text-align: center; }
          .content { padding: 20px; background: #f9f9f9; }
          .subscription { background: white; padding: 20px; border-radius: 5px; margin: 20px 0; }
        </style>
      </head>
      <body>
        <div class="container">
          <div class="header">
            <h1>Subscription Confirmation</h1>
          </div>
          <div class="content">
            <div class="subscription">
              <h3>Subscription Details</h3>
              <p><strong>Plan:</strong> ${subscriptionData.plan}</p>
              <p><strong>Amount:</strong> ${subscriptionData.currency.toUpperCase()} ${(subscriptionData.amount / 100).toFixed(2)}</p>
              <p><strong>Billing Cycle:</strong> ${subscriptionData.interval}</p>
              <p><strong>Next Billing Date:</strong> ${new Date(subscriptionData.currentPeriodEnd).toLocaleDateString()}</p>
            </div>
            <p>Your subscription has been activated successfully!</p>
            <p>Best regards,<br>The Social Flow Team</p>
          </div>
        </div>
      </body>
      </html>
    `;
  }

  private getSubscriptionConfirmationEmailTextTemplate(subscriptionData: any): string {
    return `
      Subscription Confirmation
      
      Subscription Details:
      Plan: ${subscriptionData.plan}
      Amount: ${subscriptionData.currency.toUpperCase()} ${(subscriptionData.amount / 100).toFixed(2)}
      Billing Cycle: ${subscriptionData.interval}
      Next Billing Date: ${new Date(subscriptionData.currentPeriodEnd).toLocaleDateString()}
      
      Your subscription has been activated successfully!
      
      Best regards,
      The Social Flow Team
    `;
  }

  private htmlToText(html: string): string {
    // Simple HTML to text conversion
    return html
      .replace(/<[^>]*>/g, '')
      .replace(/&nbsp;/g, ' ')
      .replace(/&amp;/g, '&')
      .replace(/&lt;/g, '<')
      .replace(/&gt;/g, '>')
      .replace(/&quot;/g, '"')
      .replace(/&#39;/g, "'")
      .trim();
  }
}
