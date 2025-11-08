import { ProjectsClient } from '@google-cloud/resource-manager';
import { VertexAIEmbeddings } from '@langchain/google-vertexai';
import { formatPrivateKey } from 'n8n-nodes-base/dist/utils/utilities';
import { NodeConnectionTypes } from 'n8n-workflow';
import type {
	ILoadOptionsFunctions,
	INodeType,
	INodeTypeDescription,
	ISupplyDataFunctions,
	SupplyData,
} from 'n8n-workflow';

import { logWrapper } from '@utils/logWrapper';
import { getConnectionHintNoticeField } from '@utils/sharedFields';

export class EmbeddingsGoogleVertex implements INodeType {
	methods = {
		listSearch: {
			async gcpProjectsList(this: ILoadOptionsFunctions) {
				const results: Array<{ name: string; value: string }> = [];

				const credentials = await this.getCredentials('googleApi');
				const privateKey = formatPrivateKey(credentials.privateKey as string);
				const email = (credentials.email as string).trim();

				const client = new ProjectsClient({
					credentials: {
						client_email: email,
						private_key: privateKey,
					},
				});

				const [projects] = await client.searchProjects();

				for (const project of projects) {
					if (project.projectId) {
						results.push({
							name: project.displayName ?? project.projectId,
							value: project.projectId,
						});
					}
				}

				return { results };
			},
		},
	};

	description: INodeTypeDescription = {
		displayName: 'Embeddings Google Vertex',
		name: 'embeddingsGoogleVertex',
		icon: 'file:google.svg',
		group: ['transform'],
		version: 1,
		description: 'Use Google Vertex Embeddings',
		defaults: {
			name: 'Embeddings Google Vertex',
		},
		requestDefaults: {
			ignoreHttpStatusErrors: true,
			baseURL: '={{ $credentials.host }}',
		},
		credentials: [
			{
				name: 'googleApi',
				required: true,
			},
		],
		codex: {
			categories: ['AI'],
			subcategories: {
				AI: ['Embeddings'],
			},
			resources: {
				primaryDocumentation: [
					{
						url: 'https://docs.n8n.io/integrations/builtin/cluster-nodes/sub-nodes/n8n-nodes-langchain.embeddingsgooglevertex/',
					},
				],
			},
		},

		inputs: [],

		outputs: [NodeConnectionTypes.AiEmbedding],
		outputNames: ['Embeddings'],

		properties: [
			getConnectionHintNoticeField([NodeConnectionTypes.AiVectorStore]),
			{
				displayName:
					'Each model is using different dimensional density for embeddings. Please make sure to use the same dimensionality for your vector store. The default model is using 768-dimensional embeddings. You can find available models <a href="https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api">here</a>.',
				name: 'notice',
				type: 'notice',
				default: '',
			},
			{
				displayName: 'Project ID',
				name: 'projectId',
				type: 'resourceLocator',
				default: { mode: 'list', value: '' },
				required: true,
				description: 'Select or enter your Google Cloud project ID',
				modes: [
					{
						displayName: 'From List',
						name: 'list',
						type: 'list',
						typeOptions: {
							searchListMethod: 'gcpProjectsList',
						},
					},
					{
						displayName: 'ID',
						name: 'id',
						type: 'string',
					},
				],
			},
			{
				displayName: 'Model Name',
				name: 'modelName',
				type: 'string',
				description:
					'The model which will generate the embeddings. <a href="https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api">Learn more</a>.',
				default: 'text-embedding-005',
			},
			{
				displayName: 'Options',
				name: 'options',
				placeholder: 'Add Option',
				description: 'Additional options to add',
				type: 'collection',
				default: {},
				options: [
					{
						displayName: 'Output Dimensionality',
						name: 'outputDimensionality',
						default: undefined,
						description:
							'The number of dimensions the resulting output embeddings should have. Supported in text-embedding-005 and later models.',
						type: 'number',
						typeOptions: {
							minValue: 1,
							maxValue: 3072,
						},
					},
				],
			},
		],
	};

	async supplyData(this: ISupplyDataFunctions, itemIndex: number): Promise<SupplyData> {
		const credentials = await this.getCredentials('googleApi');
		const privateKey = formatPrivateKey(credentials.privateKey as string);
		const email = (credentials.email as string).trim();
		const region = credentials.region as string;

		const modelName = this.getNodeParameter('modelName', itemIndex) as string;

		const projectId = this.getNodeParameter('projectId', itemIndex, '', {
			extractValue: true,
		}) as string;

		const options = this.getNodeParameter('options', itemIndex, {}) as {
			outputDimensionality?: number;
		};

		const embeddingsConfig: {
			authOptions: {
				projectId: string;
				credentials: {
					client_email: string;
					private_key: string;
				};
			};
			location: string;
			model: string;
			outputDimensionality?: number;
		} = {
			authOptions: {
				projectId,
				credentials: {
					client_email: email,
					private_key: privateKey,
				},
			},
			location: region,
			model: modelName,
		};

		if (options.outputDimensionality !== undefined) {
			embeddingsConfig.outputDimensionality = options.outputDimensionality;
		}

		const embeddings = new VertexAIEmbeddings(embeddingsConfig);

		// Store dimensions on the embeddings instance for vector stores to access
		const wrappedEmbeddings = logWrapper(embeddings, this);
		if (options.outputDimensionality !== undefined) {
			// Store dimensions on the wrapped embeddings proxy for vector stores to access
			Object.defineProperty(wrappedEmbeddings, 'dimensions', {
				value: options.outputDimensionality,
				writable: false,
				enumerable: false,
			});
		}

		return {
			response: wrappedEmbeddings,
		};
	}
}
