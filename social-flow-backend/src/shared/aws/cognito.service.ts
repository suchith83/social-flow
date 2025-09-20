import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { AwsService } from './aws.service';
import {
  AdminCreateUserCommand,
  AdminDeleteUserCommand,
  AdminGetUserCommand,
  AdminUpdateUserAttributesCommand,
  AdminSetUserPasswordCommand,
  AdminConfirmSignUpCommand,
  AdminResendConfirmationCodeCommand,
  InitiateAuthCommand,
  RespondToAuthChallengeCommand,
  GlobalSignOutCommand,
  AdminInitiateAuthCommand,
  AdminRespondToAuthChallengeCommand,
} from '@aws-sdk/client-cognito-identity-provider';

@Injectable()
export class CognitoService {
  private readonly userPoolId: string;
  private readonly clientId: string;

  constructor(
    private awsService: AwsService,
    private configService: ConfigService,
  ) {
    this.userPoolId = this.configService.get('aws.cognito.userPoolId');
    this.clientId = this.configService.get('aws.cognito.clientId');
  }

  async createUser(email: string, username: string, temporaryPassword?: string): Promise<any> {
    const command = new AdminCreateUserCommand({
      UserPoolId: this.userPoolId,
      Username: username,
      UserAttributes: [
        { Name: 'email', Value: email },
        { Name: 'email_verified', Value: 'true' },
      ],
      TemporaryPassword: temporaryPassword,
      MessageAction: 'SUPPRESS',
    });

    return this.awsService.cognitoClient.send(command);
  }

  async getUser(username: string): Promise<any> {
    const command = new AdminGetUserCommand({
      UserPoolId: this.userPoolId,
      Username: username,
    });

    return this.awsService.cognitoClient.send(command);
  }

  async updateUserAttributes(username: string, attributes: Record<string, string>): Promise<any> {
    const userAttributes = Object.entries(attributes).map(([name, value]) => ({
      Name: name,
      Value: value,
    }));

    const command = new AdminUpdateUserAttributesCommand({
      UserPoolId: this.userPoolId,
      Username: username,
      UserAttributes: userAttributes,
    });

    return this.awsService.cognitoClient.send(command);
  }

  async setUserPassword(username: string, password: string, permanent: boolean = true): Promise<any> {
    const command = new AdminSetUserPasswordCommand({
      UserPoolId: this.userPoolId,
      Username: username,
      Password: password,
      Permanent: permanent,
    });

    return this.awsService.cognitoClient.send(command);
  }

  async confirmSignUp(username: string, confirmationCode: string): Promise<any> {
    const command = new AdminConfirmSignUpCommand({
      UserPoolId: this.userPoolId,
      Username: username,
      ConfirmationCode: confirmationCode,
    });

    return this.awsService.cognitoClient.send(command);
  }

  async resendConfirmationCode(username: string): Promise<any> {
    const command = new AdminResendConfirmationCodeCommand({
      UserPoolId: this.userPoolId,
      Username: username,
    });

    return this.awsService.cognitoClient.send(command);
  }

  async authenticateUser(username: string, password: string): Promise<any> {
    const command = new AdminInitiateAuthCommand({
      UserPoolId: this.userPoolId,
      ClientId: this.clientId,
      AuthFlow: 'ADMIN_NO_SRP_AUTH',
      AuthParameters: {
        USERNAME: username,
        PASSWORD: password,
      },
    });

    return this.awsService.cognitoClient.send(command);
  }

  async refreshToken(refreshToken: string): Promise<any> {
    const command = new InitiateAuthCommand({
      ClientId: this.clientId,
      AuthFlow: 'REFRESH_TOKEN_AUTH',
      AuthParameters: {
        REFRESH_TOKEN: refreshToken,
      },
    });

    return this.awsService.cognitoClient.send(command);
  }

  async signOut(accessToken: string): Promise<any> {
    const command = new GlobalSignOutCommand({
      AccessToken: accessToken,
    });

    return this.awsService.cognitoClient.send(command);
  }

  async deleteUser(username: string): Promise<any> {
    const command = new AdminDeleteUserCommand({
      UserPoolId: this.userPoolId,
      Username: username,
    });

    return this.awsService.cognitoClient.send(command);
  }
}
