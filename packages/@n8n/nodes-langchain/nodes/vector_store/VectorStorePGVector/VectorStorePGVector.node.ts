import {
	PGVectorStore,
	type DistanceStrategy,
	type PGVectorStoreArgs,
} from '@langchain/community/vectorstores/pgvector';
import type { EmbeddingsInterface } from '@langchain/core/embeddings';
import { configurePostgres } from 'n8n-nodes-base/dist/nodes/Postgres/transport/index';
import type { PostgresNodeCredentials } from 'n8n-nodes-base/dist/nodes/Postgres/v2/helpers/interfaces';
import type { INodeProperties } from 'n8n-workflow';
import type pg from 'pg';

import { metadataFilterField } from '@utils/sharedFields';

import { createVectorStoreNode } from '../shared/createVectorStoreNode/createVectorStoreNode';

type CollectionOptions = {
	useCollection?: boolean;
	collectionName?: string;
	collectionTableName?: string;
};

type ColumnOptions = {
	idColumnName: string;
	vectorColumnName: string;
	contentColumnName: string;
	metadataColumnName: string;
};

const sharedFields: INodeProperties[] = [
	{
		displayName: 'Table Name',
		name: 'tableName',
		type: 'string',
		default: 'n8n_vectors',
		description:
			'The table name to store the vectors in. If table does not exist, it will be created.',
	},
];

const collectionField: INodeProperties = {
	displayName: 'Collection',
	name: 'collection',
	type: 'fixedCollection',
	description: 'Collection of vectors',
	default: {
		values: {
			useCollection: false,
			collectionName: 'n8n',
			collectionTable: 'n8n_vector_collections',
		},
	},
	typeOptions: {},
	placeholder: 'Add Collection Settings',
	options: [
		{
			name: 'values',
			displayName: 'Collection Settings',
			values: [
				{
					displayName: 'Use Collection',
					name: 'useCollection',
					type: 'boolean',
					default: false,
				},
				{
					displayName: 'Collection Name',
					name: 'collectionName',
					type: 'string',
					default: 'n8n',
					required: true,
					displayOptions: { show: { useCollection: [true] } },
				},
				{
					displayName: 'Collection Table Name',
					name: 'collectionTableName',
					type: 'string',
					default: 'n8n_vector_collections',
					required: true,
					displayOptions: { show: { useCollection: [true] } },
				},
			],
		},
	],
};

const columnNamesField: INodeProperties = {
	displayName: 'Column Names',
	name: 'columnNames',
	type: 'fixedCollection',
	description: 'The names of the columns in the PGVector table',
	default: {
		values: {
			idColumnName: 'id',
			vectorColumnName: 'embedding',
			contentColumnName: 'text',
			metadataColumnName: 'metadata',
		},
	},
	typeOptions: {},
	placeholder: 'Set Column Names',
	options: [
		{
			name: 'values',
			displayName: 'Column Name Settings',
			values: [
				{
					displayName: 'ID Column Name',
					name: 'idColumnName',
					type: 'string',
					default: 'id',
					required: true,
				},
				{
					displayName: 'Vector Column Name',
					name: 'vectorColumnName',
					type: 'string',
					default: 'embedding',
					required: true,
				},
				{
					displayName: 'Content Column Name',
					name: 'contentColumnName',
					type: 'string',
					default: 'text',
					required: true,
				},
				{
					displayName: 'Metadata Column Name',
					name: 'metadataColumnName',
					type: 'string',
					default: 'metadata',
					required: true,
				},
			],
		},
	],
};

const distanceStrategyField: INodeProperties = {
	displayName: 'Distance Strategy',
	name: 'distanceStrategy',
	type: 'options',
	default: 'cosine',
	description: 'The method to calculate the distance between two vectors',
	options: [
		{
			name: 'Cosine',
			value: 'cosine',
		},
		{
			name: 'Inner Product',
			value: 'innerProduct',
		},
		{
			name: 'Euclidean',
			value: 'euclidean',
		},
	],
};

const insertFields: INodeProperties[] = [
	{
		displayName: 'Options',
		name: 'options',
		type: 'collection',
		placeholder: 'Add Option',
		default: {},
		options: [collectionField, columnNamesField],
	},
];

const retrieveFields: INodeProperties[] = [
	{
		displayName: 'Options',
		name: 'options',
		type: 'collection',
		placeholder: 'Add Option',
		default: {},
		options: [distanceStrategyField, collectionField, columnNamesField, metadataFilterField],
	},
];

/**
 * Normalizes a vector to unit length (L2 normalization).
 * For cosine similarity, vectors must be normalized to ensure accurate results.
 * @param vector - The vector to normalize
 * @returns The normalized vector
 */
function normalizeVector(vector: number[]): number[] {
	const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
	if (magnitude === 0) {
		return vector; // Return zero vector as-is
	}
	// Always normalize to ensure exact unit length, even if close to 1
	// This ensures consistency and handles floating point precision issues
	// Re-normalizing an already normalized vector is safe (dividing by ~1)
	return vector.map((val) => val / magnitude);
}

/**
 * Extended PGVectorStore class to handle custom filtering.
 * This wrapper is necessary because when used as a retriever,
 * similaritySearchVectorWithScore should use this.filter instead of
 * expecting it from the parameter
 */
class ExtendedPGVectorStore extends PGVectorStore {
	distanceStrategy?: DistanceStrategy;

	static async initialize(
		embeddings: EmbeddingsInterface,
		args: PGVectorStoreArgs & { dimensions?: number },
	): Promise<ExtendedPGVectorStore> {
		const { dimensions, ...rest } = args;
		const postgresqlVectorStore = new this(embeddings, rest);

		// Store distance strategy for normalization check
		postgresqlVectorStore.distanceStrategy = rest.distanceStrategy;

		await postgresqlVectorStore._initializeClient();
		await postgresqlVectorStore.ensureTableInDatabase(dimensions);
		if (postgresqlVectorStore.collectionTableName) {
			await postgresqlVectorStore.ensureCollectionTableInDatabase();
		}

		return postgresqlVectorStore;
	}

	async addVectors(
		vectors: number[][],
		documents: Array<{ pageContent: string; metadata?: Record<string, unknown> }>,
		ids?: string[],
	): Promise<string[]> {
		// Normalize vectors before adding them if using cosine similarity
		// Cosine similarity measures the angle between vectors, not magnitude,
		// so normalization is essential for accurate similarity calculations
		let normalizedVectors = vectors;
		if (this.distanceStrategy === 'cosine') {
			normalizedVectors = vectors.map((vector) => normalizeVector(vector));
		}

		return await super.addVectors(normalizedVectors, documents, ids);
	}

	async similaritySearchVectorWithScore(
		query: number[],
		k: number,
		filter?: PGVectorStore['FilterType'],
	) {
		const mergedFilter = { ...this.filter, ...filter };

		// Normalize query vector for cosine similarity to ensure accurate results
		// Cosine similarity measures the angle between vectors, not magnitude,
		// so normalization is important for correct similarity calculations
		let normalizedQuery = query;
		if (this.distanceStrategy === 'cosine') {
			normalizedQuery = normalizeVector(query);
		}

		return await super.similaritySearchVectorWithScore(normalizedQuery, k, mergedFilter);
	}
}

export class VectorStorePGVector extends createVectorStoreNode<ExtendedPGVectorStore>({
	meta: {
		description: 'Work with your data in Postgresql with the PGVector extension',
		icon: 'file:postgres.svg',
		displayName: 'Postgres PGVector Store',
		docsUrl:
			'https://docs.n8n.io/integrations/builtin/cluster-nodes/root-nodes/n8n-nodes-langchain.vectorstorepgvector/',
		name: 'vectorStorePGVector',
		credentials: [
			{
				name: 'postgres',
				required: true,
				testedBy: 'postgresConnectionTest',
			},
		],
		operationModes: ['load', 'insert', 'retrieve', 'retrieve-as-tool'],
	},
	sharedFields,
	insertFields,
	loadFields: retrieveFields,
	retrieveFields,
	async getVectorStoreClient(context, filter, embeddings, itemIndex) {
		const tableName = context.getNodeParameter('tableName', itemIndex, '', {
			extractValue: true,
		}) as string;
		const credentials = await context.getCredentials('postgres');
		const pgConf = await configurePostgres.call(context, credentials as PostgresNodeCredentials);
		const pool = pgConf.db.$pool as unknown as pg.Pool;

		const config: PGVectorStoreArgs & { dimensions?: number } = {
			pool,
			tableName,
			filter,
		};

		const collectionOptions = context.getNodeParameter(
			'options.collection.values',
			0,
			{},
		) as CollectionOptions;

		if (collectionOptions?.useCollection) {
			config.collectionName = collectionOptions.collectionName;
			config.collectionTableName = collectionOptions.collectionTableName;
		}

		config.columns = context.getNodeParameter('options.columnNames.values', 0, {
			idColumnName: 'id',
			vectorColumnName: 'embedding',
			contentColumnName: 'text',
			metadataColumnName: 'metadata',
		}) as ColumnOptions;

		config.distanceStrategy = context.getNodeParameter(
			'options.distanceStrategy',
			0,
			'cosine',
		) as DistanceStrategy;

		// Extract dimensions from embeddings if available (e.g., from outputDimensionality parameter)
		// Check if embeddings has a dimensions property (added by logWrapper for embeddings with outputDimensionality)
		const embeddingsWithDimensions = embeddings as EmbeddingsInterface & { dimensions?: number };
		if (embeddingsWithDimensions.dimensions !== undefined) {
			config.dimensions = embeddingsWithDimensions.dimensions;
		}

		return await ExtendedPGVectorStore.initialize(embeddings, config);
	},

	async populateVectorStore(context, embeddings, documents, itemIndex) {
		// NOTE: if you are to create the HNSW index before use, you need to consider moving the distanceStrategy field to
		// shared fields, because you need that strategy when creating the index.
		const tableName = context.getNodeParameter('tableName', itemIndex, '', {
			extractValue: true,
		}) as string;
		const credentials = await context.getCredentials('postgres');
		const pgConf = await configurePostgres.call(context, credentials as PostgresNodeCredentials);
		const pool = pgConf.db.$pool as unknown as pg.Pool;

		const config: PGVectorStoreArgs & { dimensions?: number } = {
			pool,
			tableName,
		};

		const collectionOptions = context.getNodeParameter(
			'options.collection.values',
			0,
			{},
		) as CollectionOptions;

		if (collectionOptions?.useCollection) {
			config.collectionName = collectionOptions.collectionName;
			config.collectionTableName = collectionOptions.collectionTableName;
		}

		config.columns = context.getNodeParameter('options.columnNames.values', 0, {
			idColumnName: 'id',
			vectorColumnName: 'embedding',
			contentColumnName: 'text',
			metadataColumnName: 'metadata',
		}) as ColumnOptions;

		// Get distance strategy for normalization (default to cosine)
		config.distanceStrategy = context.getNodeParameter(
			'options.distanceStrategy',
			0,
			'cosine',
		) as DistanceStrategy;

		// Extract dimensions from embeddings if available (e.g., from outputDimensionality parameter)
		const embeddingsWithDimensions = embeddings as EmbeddingsInterface & { dimensions?: number };
		if (embeddingsWithDimensions.dimensions !== undefined) {
			config.dimensions = embeddingsWithDimensions.dimensions;
		}

		// Use ExtendedPGVectorStore to ensure vectors are normalized for cosine similarity
		const vectorStore = await ExtendedPGVectorStore.initialize(embeddings, config);
		await vectorStore.addDocuments(documents);
		vectorStore.client?.release();
	},

	releaseVectorStoreClient(vectorStore) {
		vectorStore.client?.release();
	},
}) {}
