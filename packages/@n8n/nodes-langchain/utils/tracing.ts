import type { BaseCallbackConfig } from '@langchain/core/callbacks/manager';
import type { IExecuteFunctions } from 'n8n-workflow';
import { CallbackHandler } from 'langfuse-langchain';
import { appendFileSync } from 'fs';

interface TracingConfig {
	additionalMetadata?: Record<string, unknown>;
}

function log(message: string) {
	try {
		appendFileSync('/tmp/langfuse-debug.log', `${new Date().toISOString()} - ${message}\n`);
	} catch (e) {
		// Ignore file write errors
	}
}

function getLangfuseHandler(context: IExecuteFunctions): CallbackHandler | null {
	try {
		const publicKey = process.env.LANGFUSE_PUBLIC_KEY;
		const secretKey = process.env.LANGFUSE_SECRET_KEY;
		const baseUrl = process.env.LANGFUSE_HOST || 'https://cloud.langfuse.com';

		log(`[Langfuse] getLangfuseHandler called - pk: ${!!publicKey}, sk: ${!!secretKey}`);

		if (!publicKey || !secretKey) {
			log('[Langfuse] Missing credentials - not initializing');
			return null;
		}

		log(`[Langfuse] Initializing handler for execution: ${context.getExecutionId()}`);

		const handler = new CallbackHandler({
			publicKey,
			secretKey,
			baseUrl,
			flushAt: 1,
			sessionId: context.getExecutionId(),
			userId: context.getWorkflow().id,
			metadata: {
				workflow: context.getWorkflow().name,
				node: context.getNode().name,
				execution_id: context.getExecutionId(),
			},
		});

		log('[Langfuse] Handler created successfully');
		return handler;
	} catch (error: any) {
		log(`[Langfuse] Initialization failed: ${error.message}`);
		log(`[Langfuse] Error stack: ${error.stack}`);
		context.logger?.debug('Langfuse handler initialization failed', { error });
		return null;
	}
}

export function getTracingConfig(
	context: IExecuteFunctions,
	config: TracingConfig = {},
): BaseCallbackConfig {
	log(
		`[Langfuse] getTracingConfig called for node: ${context.getNode().name} (type: ${context.getNode().type})`,
	);

	const parentRunManager = context.getParentCallbackManager
		? context.getParentCallbackManager()
		: undefined;

	const callbacks: any[] = parentRunManager ? [parentRunManager] : [];

	const langfuseHandler = getLangfuseHandler(context);
	if (langfuseHandler) {
		callbacks.push(langfuseHandler);
		log('[Langfuse] Handler added to callbacks array');
		context.logger?.info('Langfuse tracing enabled for this execution');
	} else {
		log('[Langfuse] Handler was null, not added to callbacks');
	}

	log(`[Langfuse] Returning config with ${callbacks.length} callback(s)`);

	return {
		runName: `[${context.getWorkflow().name}] ${context.getNode().name}`,
		metadata: {
			execution_id: context.getExecutionId(),
			workflow: context.getWorkflow(),
			node: context.getNode().name,
			...(config.additionalMetadata ?? {}),
		},
		callbacks: callbacks.length > 0 ? callbacks : undefined,
	};
}
