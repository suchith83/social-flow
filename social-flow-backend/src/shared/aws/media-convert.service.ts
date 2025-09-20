import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { AwsService } from './aws.service';
import {
  CreateJobCommand,
  GetJobCommand,
  ListJobsCommand,
  CancelJobCommand,
} from '@aws-sdk/client-mediaconvert';

export interface VideoTranscodingJob {
  videoId: string;
  inputKey: string;
  outputPrefix: string;
  resolutions: string[];
  thumbnails: boolean;
  metadata: Record<string, any>;
}

@Injectable()
export class MediaConvertService {
  private readonly role: string;

  constructor(
    private awsService: AwsService,
    private configService: ConfigService,
  ) {
    this.role = this.configService.get('aws.mediaConvert.role');
  }

  async createTranscodingJob(job: VideoTranscodingJob): Promise<string> {
    const jobSettings = this.createJobSettings(job);
    
    const command = new CreateJobCommand({
      Role: this.role,
      Settings: jobSettings,
      UserMetadata: {
        videoId: job.videoId,
        ...job.metadata,
      },
    });

    const response = await this.awsService.mediaConvertClient.send(command);
    return response.Job?.Id || '';
  }

  async getJobStatus(jobId: string): Promise<any> {
    const command = new GetJobCommand({ Id: jobId });
    const response = await this.awsService.mediaConvertClient.send(command);
    return response.Job;
  }

  async listJobs(status?: string, maxResults?: number): Promise<any[]> {
    const command = new ListJobsCommand({
      Status: status,
      MaxResults: maxResults,
    });

    const response = await this.awsService.mediaConvertClient.send(command);
    return response.Jobs || [];
  }

  async cancelJob(jobId: string): Promise<void> {
    const command = new CancelJobCommand({ Id: jobId });
    await this.awsService.mediaConvertClient.send(command);
  }

  private createJobSettings(job: VideoTranscodingJob): any {
    const bucketName = this.configService.get('aws.s3.bucket');
    const region = this.awsService.getRegion();

    return {
      Inputs: [
        {
          FileInput: `s3://${bucketName}/${job.inputKey}`,
        },
      ],
      OutputGroups: [
        {
          Name: 'HLS',
          OutputGroupSettings: {
            Type: 'HLS_GROUP_SETTINGS',
            HlsGroupSettings: {
              Destination: `s3://${bucketName}/${job.outputPrefix}/hls/`,
              SegmentLength: 10,
              MinSegmentLength: 0,
              SegmentModifier: '',
            },
          },
          Outputs: this.createHLSOutputs(job.resolutions),
        },
        {
          Name: 'DASH',
          OutputGroupSettings: {
            Type: 'DASH_ISO_GROUP_SETTINGS',
            DashIsoGroupSettings: {
              Destination: `s3://${bucketName}/${job.outputPrefix}/dash/`,
              SegmentLength: 10,
              MinSegmentLength: 0,
              SegmentModifier: '',
            },
          },
          Outputs: this.createDASHOutputs(job.resolutions),
        },
        ...(job.thumbnails ? [this.createThumbnailOutput(bucketName, job.outputPrefix)] : []),
      ],
    };
  }

  private createHLSOutputs(resolutions: string[]): any[] {
    return resolutions.map(resolution => {
      const [width, height] = resolution.split('x').map(Number);
      return {
        NameModifier: `_${resolution}`,
        VideoDescription: {
          CodecSettings: {
            Codec: 'H_264',
            H264Settings: {
              RateControlMode: 'QVBR',
              QvbrSettings: {
                QvbrQualityLevel: 7,
              },
              MaxBitrate: this.getBitrateForResolution(resolution),
            },
          },
          Width: width,
          Height: height,
        },
        AudioDescriptions: [
          {
            CodecSettings: {
              Codec: 'AAC',
              AacSettings: {
                Bitrate: 128000,
                CodingMode: 'CODING_MODE_2_0',
                SampleRate: 48000,
              },
            },
          },
        ],
        ContainerSettings: {
          Container: 'M3U8',
          M3u8Settings: {
            AudioFramesPerPes: 4,
            PcrControl: 'PCR_EVERY_PES_PACKET',
            PmtPid: 480,
            PrivateMetadataPid: 503,
            ProgramNumber: 1,
            PatInterval: 0,
            PmtInterval: 0,
            Scte35Source: 'NONE',
            TimedMetadata: 'NONE',
            VideoPid: 481,
          },
        },
      };
    });
  }

  private createDASHOutputs(resolutions: string[]): any[] {
    return resolutions.map(resolution => {
      const [width, height] = resolution.split('x').map(Number);
      return {
        NameModifier: `_${resolution}`,
        VideoDescription: {
          CodecSettings: {
            Codec: 'H_264',
            H264Settings: {
              RateControlMode: 'QVBR',
              QvbrSettings: {
                QvbrQualityLevel: 7,
              },
              MaxBitrate: this.getBitrateForResolution(resolution),
            },
          },
          Width: width,
          Height: height,
        },
        AudioDescriptions: [
          {
            CodecSettings: {
              Codec: 'AAC',
              AacSettings: {
                Bitrate: 128000,
                CodingMode: 'CODING_MODE_2_0',
                SampleRate: 48000,
              },
            },
          },
        ],
        ContainerSettings: {
          Container: 'MP4',
          Mp4Settings: {
            CslgAtom: 'INCLUDE',
            FreeSpaceBox: 'EXCLUDE',
            MoovPlacement: 'PROGRESSIVE_DOWNLOAD',
          },
        },
      };
    });
  }

  private createThumbnailOutput(bucketName: string, outputPrefix: string): any {
    return {
      Name: 'Thumbnails',
      OutputGroupSettings: {
        Type: 'FILE_GROUP_SETTINGS',
        FileGroupSettings: {
          Destination: `s3://${bucketName}/${outputPrefix}/thumbnails/`,
        },
      },
      Outputs: [
        {
          NameModifier: '_thumb',
          VideoDescription: {
            CodecSettings: {
              Codec: 'FRAME_CAPTURE',
              FrameCaptureSettings: {
                MaxCaptures: 10000000,
                Quality: 80,
              },
            },
            Width: 1280,
            Height: 720,
          },
          ContainerSettings: {
            Container: 'RAW',
          },
        },
      ],
    };
  }

  private getBitrateForResolution(resolution: string): number {
    const bitrates: Record<string, number> = {
      '240p': 400000,
      '360p': 800000,
      '480p': 1200000,
      '720p': 2500000,
      '1080p': 5000000,
      '1440p': 10000000,
      '2160p': 20000000,
    };
    return bitrates[resolution] || 1000000;
  }
}
