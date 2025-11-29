export interface ErrorLike {
	message?: string;
	description?: string;
}

export interface ErrorContext {
	modelName?: string;
}

export function makeErrorFromStatus(statusCode: number, context?: ErrorContext): ErrorLike {
	const errorMessages: Record<number, ErrorLike> = {
		403: {
			message: 'Unauthorized for this project',
			description:
				'Check your Google Cloud project ID, that your credential has access to that project and that billing is enabled',
		},
		404: {
			message: context?.modelName
				? `No model found called '${context.modelName}'`
				: 'No model found',
			description:
				'This could mean:\n' +
				'1. The model name is incorrect or misspelled\n' +
				'2. The model is not available in your region or project\n' +
				'3. The model requires special access/permissions\n' +
				'4. For Vertex AI, model names should not include the "models/" prefix\n' +
				'5. Some models (like gemini-3-pro-preview) require the "global" region endpoint, not regional endpoints\n' +
				'Please verify the model name and check <a href="https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models" target="_blank">available models</a> for your region.',
		},
	};

	return errorMessages[statusCode];
}
