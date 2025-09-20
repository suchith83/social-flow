import { Injectable } from '@nestjs/common';
import { PassportStrategy } from '@nestjs/passport';
import { Strategy } from 'passport-twitter';
import { ConfigService } from '@nestjs/config';

@Injectable()
export class TwitterStrategy extends PassportStrategy(Strategy, 'twitter') {
  constructor(private configService: ConfigService) {
    super({
      consumerKey: configService.get('TWITTER_CONSUMER_KEY'),
      consumerSecret: configService.get('TWITTER_CONSUMER_SECRET'),
      callbackURL: configService.get('TWITTER_CALLBACK_URL'),
      includeEmail: true,
    });
  }

  async validate(
    accessToken: string,
    refreshToken: string,
    profile: any,
    done: Function,
  ): Promise<any> {
    const { id, username, displayName, photos, emails } = profile;
    const user = {
      id,
      email: emails[0].value,
      firstName: displayName.split(' ')[0],
      lastName: displayName.split(' ').slice(1).join(' '),
      avatar: photos[0].value,
      provider: 'twitter',
      accessToken,
      refreshToken,
    };
    done(null, user);
  }
}
