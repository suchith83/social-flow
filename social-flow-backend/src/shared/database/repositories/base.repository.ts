import { Repository, FindManyOptions, FindOneOptions, DeepPartial, UpdateResult, DeleteResult } from 'typeorm';
import { BaseEntity } from '../entities/base.entity';

export abstract class BaseRepository<T extends BaseEntity> {
  constructor(protected readonly repository: Repository<T>) {}

  async create(data: DeepPartial<T>): Promise<T> {
    const entity = this.repository.create(data);
    return this.repository.save(entity);
  }

  async findById(id: string): Promise<T | null> {
    return this.repository.findOne({ where: { id } as any });
  }

  async findOne(options: FindOneOptions<T>): Promise<T | null> {
    return this.repository.findOne(options);
  }

  async findMany(options: FindManyOptions<T>): Promise<T[]> {
    return this.repository.find(options);
  }

  async findAll(): Promise<T[]> {
    return this.repository.find();
  }

  async update(id: string, data: DeepPartial<T>): Promise<UpdateResult> {
    return this.repository.update(id, data);
  }

  async delete(id: string): Promise<DeleteResult> {
    return this.repository.delete(id);
  }

  async count(options?: FindManyOptions<T>): Promise<number> {
    return this.repository.count(options);
  }

  async exists(options: FindOneOptions<T>): Promise<boolean> {
    const count = await this.repository.count(options);
    return count > 0;
  }

  async save(entity: T): Promise<T> {
    return this.repository.save(entity);
  }

  async saveMany(entities: T[]): Promise<T[]> {
    return this.repository.save(entities);
  }

  async softDelete(id: string): Promise<UpdateResult> {
    return this.repository.softDelete(id);
  }

  async restore(id: string): Promise<UpdateResult> {
    return this.repository.restore(id);
  }

  async paginate(
    page: number = 1,
    limit: number = 10,
    options: FindManyOptions<T> = {}
  ): Promise<{ data: T[]; total: number; page: number; limit: number; totalPages: number }> {
    const skip = (page - 1) * limit;
    const [data, total] = await this.repository.findAndCount({
      ...options,
      skip,
      take: limit,
    });

    return {
      data,
      total,
      page,
      limit,
      totalPages: Math.ceil(total / limit),
    };
  }
}
