import { Injectable, UnauthorizedException, ConflictException, BadRequestException } from '@nestjs/common';
import { JwtService } from '@nestjs/jwt';
import { ConfigService } from '@nestjs/config';
import * as bcrypt from 'bcrypt';
import { v4 as uuidv4 } from 'uuid';

import { UserRepository } from '../shared/database/repositories/user.repository';
import { User, UserRole, UserStatus } from '../shared/database/entities/user.entity';
import { RedisService } from '../shared/redis/redis.service';
import { LoggerService } from '../shared/logger/logger.service';

export interface LoginResponse {
  user: User;
  accessToken: string;
  refreshToken: string;
}

export interface RegisterRequest {
  email: string;
  username: string;
  password: string;
  firstName?: string;
  lastName?: string;
}

export interface LoginRequest {
  emailOrUsername: string;
  password: string;
}

export interface RefreshTokenRequest {
  refreshToken: string;
}

export interface ForgotPasswordRequest {
  email: string;
}

export interface ResetPasswordRequest {
  token: string;
  newPassword: string;
}

export interface ChangePasswordRequest {
  userId: string;
  currentPassword: string;
  newPassword: string;
}

@Injectable()
export class AuthService {
  constructor(
    private userRepository: UserRepository,
    private jwtService: JwtService,
    private configService: ConfigService,
    private redisService: RedisService,
    private logger: LoggerService,
  ) {}

  async register(registerData: RegisterRequest): Promise<LoginResponse> {
    const { email, username, password, firstName, lastName } = registerData;

    // Check if user already exists
    const existingUserByEmail = await this.userRepository.findByEmail(email);
    if (existingUserByEmail) {
      throw new ConflictException('User with this email already exists');
    }

    const existingUserByUsername = await this.userRepository.findByUsername(username);
    if (existingUserByUsername) {
      throw new ConflictException('User with this username already exists');
    }

    // Hash password
    const saltRounds = 12;
    const hashedPassword = await bcrypt.hash(password, saltRounds);

    // Create user
    const user = await this.userRepository.create({
      email,
      username,
      password: hashedPassword,
      firstName,
      lastName,
      role: UserRole.USER,
      status: UserStatus.ACTIVE,
      emailVerified: false,
      emailVerificationToken: uuidv4(),
    });

    // Generate tokens
    const tokens = await this.generateTokens(user);

    // Log registration
    this.logger.logBusiness('user_registered', user.id, {
      email: user.email,
      username: user.username,
    });

    return {
      user,
      ...tokens,
    };
  }

  async login(loginData: LoginRequest): Promise<LoginResponse> {
    const { emailOrUsername, password } = loginData;

    // Find user by email or username
    let user = await this.userRepository.findByEmail(emailOrUsername);
    if (!user) {
      user = await this.userRepository.findByUsername(emailOrUsername);
    }

    if (!user) {
      throw new UnauthorizedException('Invalid credentials');
    }

    // Check if user is active
    if (user.status !== UserStatus.ACTIVE) {
      throw new UnauthorizedException('Account is not active');
    }

    // Verify password
    const isPasswordValid = await bcrypt.compare(password, user.password);
    if (!isPasswordValid) {
      throw new UnauthorizedException('Invalid credentials');
    }

    // Update last login
    await this.userRepository.updateLastActive(user.id);

    // Generate tokens
    const tokens = await this.generateTokens(user);

    // Log login
    this.logger.logBusiness('user_login', user.id, {
      email: user.email,
      username: user.username,
    });

    return {
      user,
      ...tokens,
    };
  }

  async refreshToken(refreshData: RefreshTokenRequest): Promise<{ accessToken: string; refreshToken: string }> {
    const { refreshToken } = refreshData;

    try {
      // Verify refresh token
      const payload = this.jwtService.verify(refreshToken, {
        secret: this.configService.get('app.jwtSecret'),
      });

      // Get user
      const user = await this.userRepository.findById(payload.sub);
      if (!user || user.status !== UserStatus.ACTIVE) {
        throw new UnauthorizedException('Invalid refresh token');
      }

      // Generate new tokens
      return this.generateTokens(user);
    } catch (error) {
      throw new UnauthorizedException('Invalid refresh token');
    }
  }

  async logout(userId: string): Promise<void> {
    // In a real implementation, you might want to blacklist the token
    // For now, we'll just log the logout
    this.logger.logBusiness('user_logout', userId);
  }

  async forgotPassword(forgotData: ForgotPasswordRequest): Promise<void> {
    const { email } = forgotData;

    const user = await this.userRepository.findByEmail(email);
    if (!user) {
      // Don't reveal if user exists or not
      return;
    }

    // Generate reset token
    const resetToken = uuidv4();
    const resetExpires = new Date(Date.now() + 24 * 60 * 60 * 1000); // 24 hours

    // Update user with reset token
    await this.userRepository.updatePasswordResetToken(user.id, resetToken, resetExpires);

    // Send reset email (implement email service)
    // await this.emailService.sendPasswordResetEmail(user.email, resetToken);

    this.logger.logBusiness('password_reset_requested', user.id, {
      email: user.email,
    });
  }

  async resetPassword(resetData: ResetPasswordRequest): Promise<void> {
    const { token, newPassword } = resetData;

    const user = await this.userRepository.findByPasswordResetToken(token);
    if (!user || !user.passwordResetExpires || user.passwordResetExpires < new Date()) {
      throw new BadRequestException('Invalid or expired reset token');
    }

    // Hash new password
    const saltRounds = 12;
    const hashedPassword = await bcrypt.hash(newPassword, saltRounds);

    // Update password
    await this.userRepository.updatePassword(user.id, hashedPassword);

    this.logger.logBusiness('password_reset_completed', user.id, {
      email: user.email,
    });
  }

  async changePassword(changeData: ChangePasswordRequest): Promise<void> {
    const { userId, currentPassword, newPassword } = changeData;

    const user = await this.userRepository.findById(userId);
    if (!user) {
      throw new UnauthorizedException('User not found');
    }

    // Verify current password
    const isCurrentPasswordValid = await bcrypt.compare(currentPassword, user.password);
    if (!isCurrentPasswordValid) {
      throw new UnauthorizedException('Current password is incorrect');
    }

    // Hash new password
    const saltRounds = 12;
    const hashedPassword = await bcrypt.hash(newPassword, saltRounds);

    // Update password
    await this.userRepository.updatePassword(user.id, hashedPassword);

    this.logger.logBusiness('password_changed', user.id, {
      email: user.email,
    });
  }

  async verifyEmail(token: string): Promise<void> {
    const user = await this.userRepository.findByEmailVerificationToken(token);
    if (!user) {
      throw new BadRequestException('Invalid verification token');
    }

    await this.userRepository.updateEmailVerification(user.id, true);

    this.logger.logBusiness('email_verified', user.id, {
      email: user.email,
    });
  }

  async resendVerificationEmail(email: string): Promise<void> {
    const user = await this.userRepository.findByEmail(email);
    if (!user) {
      throw new BadRequestException('User not found');
    }

    if (user.emailVerified) {
      throw new BadRequestException('Email already verified');
    }

    // Generate new verification token
    const verificationToken = uuidv4();
    await this.userRepository.update(user.id, {
      emailVerificationToken: verificationToken,
    });

    // Send verification email (implement email service)
    // await this.emailService.sendVerificationEmail(user.email, verificationToken);

    this.logger.logBusiness('verification_email_resent', user.id, {
      email: user.email,
    });
  }

  async validateUser(emailOrUsername: string, password: string): Promise<User | null> {
    const user = await this.userRepository.findByEmail(emailOrUsername) ||
                 await this.userRepository.findByUsername(emailOrUsername);

    if (user && await bcrypt.compare(password, user.password)) {
      return user;
    }

    return null;
  }

  async getUserById(userId: string): Promise<User | null> {
    return this.userRepository.findById(userId);
  }

  private async generateTokens(user: User): Promise<{ accessToken: string; refreshToken: string }> {
    const payload = {
      sub: user.id,
      email: user.email,
      username: user.username,
      role: user.role,
    };

    const accessToken = this.jwtService.sign(payload);
    const refreshToken = this.jwtService.sign(payload, {
      expiresIn: this.configService.get('app.refreshTokenExpiresIn'),
    });

    return { accessToken, refreshToken };
  }
}
