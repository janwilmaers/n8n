import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import {
	NodeConnectionTypes,
	type INodeType,
	type INodeTypeDescription,
	type ISupplyDataFunctions,
	type SupplyData,
} from 'n8n-workflow';

import { logWrapper } from '@utils/logWrapper';
import { getConnectionHintNoticeField } from '@utils/sharedFields';

export class EmbeddingsGoogleGemini implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'Embeddings Google Gemini',
		name: 'embeddingsGoogleGemini',
		icon: 'file:google.svg',
		group: ['transform'],
		version: 1,
		description: 'Use Google Gemini Embeddings',
		defaults: {
			name: 'Embeddings Google Gemini',
		},
		requestDefaults: {
			ignoreHttpStatusErrors: true,
			baseURL: '={{ $credentials.host }}',
		},
		credentials: [
			{
				name: 'googlePalmApi',
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
						url: 'https://docs.n8n.io/integrations/builtin/cluster-nodes/sub-nodes/n8n-nodes-langchain.embeddingsgooglegemini/',
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
					'Each model is using different dimensional density for embeddings. Please make sure to use the same dimensionality for your vector store. The default model is using 768-dimensional embeddings.',
				name: 'notice',
				type: 'notice',
				default: '',
			},
			{
				displayName: 'Model',
				name: 'modelName',
				type: 'options',
				description:
					'The model which will generate the embeddings. <a href="https://developers.generativeai.google/api/rest/generativelanguage/models/list">Learn more</a>.',
				typeOptions: {
					loadOptions: {
						routing: {
							request: {
								method: 'GET',
								url: '/v1beta/models',
							},
							output: {
								postReceive: [
									{
										type: 'rootProperty',
										properties: {
											property: 'models',
										},
									},
									{
										type: 'filter',
										properties: {
											pass: "={{ $responseItem.name.includes('embedding') }}",
										},
									},
									{
										type: 'setKeyValue',
										properties: {
											name: '={{$responseItem.name}}',
											value: '={{$responseItem.name}}',
											description: '={{$responseItem.description}}',
										},
									},
									{
										type: 'sort',
										properties: {
											key: 'name',
										},
									},
								],
							},
						},
					},
				},
				routing: {
					send: {
						type: 'body',
						property: 'model',
					},
				},
				default: 'models/text-embedding-004',
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
							'The number of dimensions the resulting output embeddings should have. Only supported in gemini-embedding-001 and later models.',
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
		this.logger.debug('Supply data for embeddings Google Gemini');
		const modelName = this.getNodeParameter(
			'modelName',
			itemIndex,
			'models/text-embedding-004',
		) as string;
		const credentials = await this.getCredentials('googlePalmApi');

		const options = this.getNodeParameter('options', itemIndex, {}) as {
			outputDimensionality?: number;
		};

		const embeddingsConfig: {
			apiKey: string;
			baseUrl: string;
			model: string;
			outputDimensionality?: number;
		} = {
			apiKey: credentials.apiKey as string,
			baseUrl: credentials.host as string,
			model: modelName,
		};

		if (options.outputDimensionality !== undefined) {
			embeddingsConfig.outputDimensionality = options.outputDimensionality;
		}

		const embeddings = new GoogleGenerativeAIEmbeddings(embeddingsConfig);

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
