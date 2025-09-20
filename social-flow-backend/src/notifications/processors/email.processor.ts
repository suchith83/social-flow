import { Process, Processor } from '@nestjs/bull';
import { Job } from 'bull';
import { LoggerService } from '../../shared/logger/logger.service';
import { EmailService } from '../email.service';

@Processor('email')
export class EmailProcessor {
  constructor(
    private logger: LoggerService,
    private emailService: EmailService,
  ) {}

  @Process('send-email')
  async handleEmailSending(job: Job<any>) {
    const { to, subject, template, context } = job.data;

    try {
      this.logger.logBusiness('email_processing_started', undefined, {
        to,
        subject,
        jobId: job.id,
      });

      // Send email based on template
      switch (template) {
        case 'welcome':
          await this.emailService.sendWelcomeEmail(to, context.name);
          break;
        case 'verification':
          await this.emailService.sendVerificationEmail(to, context.token);
          break;
        case 'password-reset':
          await this.emailService.sendPasswordResetEmail(to, context.token);
          break;
        case 'notification':
          await this.emailService.sendNotificationEmail(to, context);
          break;
        case 'payment-receipt':
          await this.emailService.sendPaymentReceiptEmail(to, context);
          break;
        case 'subscription-confirmation':
          await this.emailService.sendSubscriptionConfirmationEmail(to, context);
          break;
        default:
          await this.emailService.sendEmail(to, subject, context.html, context.text);
      }

      this.logger.logBusiness('email_processing_completed', undefined, {
        to,
        subject,
        jobId: job.id,
      });
    } catch (error) {
      this.logger.logError(error, 'EmailProcessor.handleEmailSending', {
        to,
        subject,
        jobId: job.id,
      });
      throw error;
    }
  }
}
