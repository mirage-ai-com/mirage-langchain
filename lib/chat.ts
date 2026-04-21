/*
 * mirage-langchain
 *
 * Copyright 2025, Mirage AI
 * Author: Mirage AI
 */

/**************************************************************************
 * IMPORTS
 ***************************************************************************/

// NPM
import {
  AIMessage,
  AIMessageChunk,
  BaseMessage,
  HumanMessage,
  ToolMessage,
  UsageMetadata
} from "@langchain/core/messages";

import { coerceMessageLikeToMessage } from "@langchain/core/messages";

import { ToolCall, ToolCallChunk } from "@langchain/core/messages/tool";

import { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";

import {
  type StructuredOutputMethodOptions
} from "@langchain/core/language_models/base";

import {
  BaseChatModel,
  BaseChatModelParams,
  BindToolsInput,
  BaseChatModelCallOptions
} from "@langchain/core/language_models/chat_models";

import {
  type BaseLanguageModelInput
} from "@langchain/core/language_models/base";

import { ChatGenerationChunk } from "@langchain/core/outputs";
import {
  RunnableConfig,
  RunnablePassthrough,
  RunnableSequence
} from "@langchain/core/runnables";

import { convertToOpenAITool } from "@langchain/core/utils/function_calling";
import {
  JsonOutputParser,
  StructuredOutputParser
} from "@langchain/core/output_parsers";

import { isInteropZodSchema } from "@langchain/core/utils/types";
import { toJsonSchema } from "@langchain/core/utils/json_schema";

import {
  Mirage,
  AnswerChatRequest,
  AnswerChatRequestContextConversationMessage,
  AnswerChatResponseStreamable,
  AnswerChatResponseToolCall,
  AnswerChatResponseChunkAnswer,
  AnswerChatRequestTool,
  AnswerChatResponseLogprob
} from "mirage-api";

import {
  Runnable
} from "@langchain/core/runnables";

interface ChatMirageCallOptions extends BaseChatModelCallOptions {
  tools?: BindToolsInput[];
  tool_choice?: ChatMirageToolChoice;
  timeout?: number;
}

/**************************************************************************
 * TYPES
 ***************************************************************************/

export type ChatMirageToolChoice = "auto" | "any" | "required";
export type ChatMirageModel = "small" | "medium" | "large";

interface ChatMirageInput extends BaseChatModelParams {
  model?: ChatMirageModel;
  userId?: string;
  tools?: BindToolsInput[];
  tool_choice?: ChatMirageToolChoice;
  secretKey?: string;
  headers?: unknown;
  streaming?: boolean;
  format?: string;
  temperature?: number;
  maxTokens?: number;
  topP?: number;
  frequencyPenalty?: number;
  presencePenalty?: number;
  seed?: number;
  logprobs?: boolean;
  topLogprobs?: number;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  structured_output?: Record<string, any> | undefined;
}

/**************************************************************************
 * LANGCHAIN > CHAT MIRAGE
 ***************************************************************************/

/**
 * ChatMirage langchain adapter
 */
class ChatMirage<
  CallOptions extends ChatMirageCallOptions = ChatMirageCallOptions
  > extends BaseChatModel<CallOptions, AIMessageChunk>
  implements ChatMirageInput {
  model: ChatMirageModel;
  userId: string;
  secretKey: string;
  temperature?: number;
  maxTokens?: number;
  format?: string;
  tools?: unknown[];
  tool_choice?: ChatMirageToolChoice;
  logprobs?: boolean;
  topLogprobs?: number;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  structured_output?: Record<string, any> | undefined;
  client: Mirage;

  fields: ChatMirageInput;

  // eslint-disable-next-line jsdoc/require-jsdoc, crisp/no-snake-case
  static lc_name() {
    return "ChatMirage";
  }

  /**
   * Constructor for ChatMirage
   */
  constructor(fields: ChatMirageInput) {
    super(fields);

    this.fields = fields;

    this.model = fields.model;
    this.userId = fields.userId;
    this.secretKey = fields.secretKey;
    this.temperature = fields.temperature;
    this.maxTokens = fields.maxTokens;
    this.format = fields.format;
    this.tools = fields.tools;
    this.tool_choice = fields.tool_choice;
    this.logprobs = fields.logprobs;
    this.topLogprobs = fields.topLogprobs;
    this.structured_output = fields.structured_output;

    if (!this.userId || !this.secretKey) {
      throw new Error("ChatMirage requires userId and secretKey");
    }

    // Initialize Mirage client
    this.client = new Mirage(this.userId, this.secretKey);
  }

  /**
   * Get the LLM type
   */
  _llmType() {
    return "mirage";
  }

  /**
   * Convert LangChain messages to Mirage format
   */
  static convertMessagesToMirage(
    messages: BaseMessage[]
  ): AnswerChatRequestContextConversationMessage[] {
    return messages.map((baseMessage) => {
      // @ts-expect-error: currentMessage is typed on BaseMessage
      let currentMessage: HumanMessage|AIMessage|ToolMessage = baseMessage;

      let role;

      const coercedMessage = coerceMessageLikeToMessage(baseMessage);

      if (coercedMessage.getType() === "human") {
        role = "user";

        currentMessage = coercedMessage as HumanMessage;
      } else if (coercedMessage.getType() === "ai") {
        role = "agent";

        currentMessage = coercedMessage as AIMessage;
      } else if (coercedMessage.getType() === "system") {
        role = "system";
      } else if (coercedMessage.getType() === "tool") {
        role = "tool";

        currentMessage = coercedMessage as ToolMessage;
      } else {
        throw new Error(
          `Unsupported message type for Mirage: ${coercedMessage.getType()}`
        );
      }

      let content = "";

      if (typeof currentMessage.content === "string") {
        content = currentMessage.content;
      } else if (Array.isArray(currentMessage.content)) {
        for (const contentPart of currentMessage.content) {
          if (contentPart.type === "text") {
            content = `${content}\n${contentPart.text as string}`;
          } else if (contentPart.type === "image_url") {
            content = "Image upload from user but unsupported";
          } else {
            throw new Error(
              `Unsupported message content type. ${contentPart.type}`
            );
          }
        }
      } else {
        // Fallback for other content types
        content = JSON.stringify(currentMessage.content);
      }

      const mirageMessage: AnswerChatRequestContextConversationMessage = {
        from: role,
        text: content
      };

      if (role === "tool") {
        // @ts-expect-error: Tool call ID is not typed on langchain
        mirageMessage.tool_call_id = currentMessage.tool_call_id as string;
      }

      const toolCalls: ToolCall[] = (
        currentMessage as AIMessage
      ).tool_calls || [];

      // Add tool calls if present in AI messages
      if (toolCalls.length > 0) {
        mirageMessage.tool_calls = toolCalls.map((toolCall) => {
          return {
            id: toolCall.id,
            function: {
              name: toolCall.name,
              arguments: (typeof (toolCall.args) === "string")
                ? JSON.parse(toolCall.args)
                : toolCall.args
            }
          };
        });
      }

      return mirageMessage;
    });
  }

  /**
   * Bind tools to the model
   */
  override bindTools(tools : BindToolsInput[]) {
    // @ts-expect-error: Tools have different types on langchain
    return this.withConfig({
      tools: tools.map((tool) => {
        return convertToOpenAITool(tool);
      })
    });
  }

  /**
   * Get LangSmith parameters
   */
  // @ts-expect-error: ParsedCallOptions is not typed on langchain
  getLsParams(options: this["ParsedCallOptions"]) {
    return {
      ls_provider: "mirage",
      ls_model_name: this.model,
      ls_model_type: "chat",
      ls_temperature: this.temperature,
      ls_max_tokens: this.maxTokens,
      ls_stop: options.stop
    };
  }

  /**
   * Get invocation parameters for Mirage API
   */
  invocationParams(_options : this["ParsedCallOptions"]) {
    const params: {
      model: ChatMirageModel;
      tools: AnswerChatRequestTool[];
      tool_choice?: { mode: string };
      answer?: {
        temperature?: number;
        max_tokens?: number;
        logprobs?: boolean;
        top_logprobs?: number;
      };
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      schema?: Record<string, any> | undefined;
    } = {
      model: this.model,
      tools: this.tools as AnswerChatRequestTool[] || [],
      tool_choice: (
        this.tool_choice === "any" || this.tool_choice === "required"
      )
        ? { mode: "required" }
        : (this.tool_choice === "auto")
            ? { mode: "auto" }
            : undefined
    };

    // Add model parameters if they exist
    const answerParams = {
      temperature: this.temperature,
      max_tokens: this.maxTokens,
      logprobs: this.logprobs,
      top_logprobs: this.topLogprobs
    };

    if (Object.keys(answerParams).length > 0) {
      params.answer = answerParams;
    }

    // Add schema from structured_output if present
    if (this.structured_output) {
      params.schema = this.structured_output;
    }

    return params;
  }

  /**
   * Generate chat completion
   */
  async _generate(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager: CallbackManagerForLLMRun
  ) {
    let fullContent = "";
    let lastMetadata = {};

    let usageMetadata: UsageMetadata;

    const allToolCalls = [];
    const allLogprobs: AnswerChatResponseLogprob[] = [];

    // Collect all chunks and build the full response
    for await (const chunk of this._streamResponseChunks(messages, options, runManager)) {
      if (chunk.text) {
        fullContent += chunk.text;
      }

      if (chunk.message?.response_metadata) {
        lastMetadata = chunk.message.response_metadata;

        // Collect logprobs from chunks
        // @ts-expect-error: response_metadata is typed on AIMessage
        if (chunk.message.response_metadata.logprobs) {
          allLogprobs.push(
            // @ts-expect-error: response_metadata is typed on AIMessage
            chunk.message.response_metadata.logprobs as AnswerChatResponseLogprob
          );
        }
      }

      // @ts-expect-error: usage_metadata is typed on AIMessage
      usageMetadata = (chunk.message as AIMessageChunk).usage_metadata;

      const toolCalls = (chunk.message as AIMessageChunk).tool_calls || [];

      // Collect all tool calls from all chunks
      if (toolCalls.length > 0) {
        allToolCalls.push(...toolCalls);
      }
    }

    // Merge all logprobs into the final metadata
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const finalMetadata: any = { ...lastMetadata };

    if (allLogprobs.length > 0) {
      finalMetadata.logprobs = {
        content: allLogprobs
      };
    }

    // Create the final AIMessage
    const nonChunkMessage = new AIMessage({
      content: fullContent,
      tool_calls: allToolCalls,
      response_metadata: finalMetadata,
      usage_metadata: usageMetadata
    });

    return {
      generations: [
        {
          text: fullContent,
          message: nonChunkMessage
        }
      ]
    };
  }

  /**
   * Stream chat completion chunks
   */
  async *_streamResponseChunks(
    messages : BaseMessage[],
    options : this["ParsedCallOptions"],
    runManager: CallbackManagerForLLMRun
  ) {
    const params = this.invocationParams(options);
    const mirageMessages = ChatMirage.convertMessagesToMirage(messages);

    const usageMetadata = {
      input_tokens: 0,
      output_tokens: 0,
      total_tokens: 0
    };

    try {
      const answerPayload: AnswerChatRequest = {
        model: params.model,
        tools: params.tools,
        context: {
          conversation: {
            messages: mirageMessages
          }
        }
      };

      // Add answer params (logprobs, temperature, etc.) if present
      if (params.answer) {
        answerPayload.answer = params.answer;
      }

      if (params.tool_choice) {
        if (mirageMessages.length > 0 &&
        mirageMessages[mirageMessages.length - 1].from === "user") {
          // @ts-expect-error: Payload is hardcoded
          answerPayload.tool_choice = params.tool_choice;
        }
      }

      if (params.schema) {
        answerPayload.schema = params.schema;
      }

      const stream = await this.client.Task.AnswerChat(
        answerPayload,
        {
          stream: true
        }
      ) as AnswerChatResponseStreamable;

      // Attach listeners IMMEDIATELY to avoid missing early events
      const chunks: (
        | AnswerChatResponseToolCall
        | AnswerChatResponseChunkAnswer
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        | { logprobs: any }
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        | { tool_calls: any }
      )[] = [];

      let isComplete = false;
      let error: Error | null = null;
      let hasData = false;

      // Listen to Mirage-specific events
      stream.on("answer", (chunk: { chunk: string }) => {
        hasData = true;

        // Extract the actual text content from the chunk object
        const text = chunk.chunk;

        chunks.push({ chunk: text });
      });

      stream.on("logprobs", (chunk: unknown) => {
        hasData = true;

        chunks.push({ logprobs: chunk });
      });

      stream.on("tool_calls", (chunk: unknown) => {
        hasData = true;

        chunks.push({ tool_calls: chunk });
      });

      // @ts-expect-error: Mirage is not handling done event
      stream.on("done", () => {
        isComplete = true;
      });

      stream.on("end", () => {
        isComplete = true;
      });

      stream.on("error", (err: Error) => {
        error = err;
      });

      const signal = options.signal;

      // Create iterator that reads from the already-attached listeners
      const createStreamIterator = async function* () {
        // Wait for chunks to arrive, completion, or abort
        while (!isComplete && !error) {
          if (signal?.aborted === true) {
            break;
          }

          if (chunks.length > 0) {
            yield chunks.shift();

            continue;
          }

          // Race the polling tick against the abort signal
          await new Promise<void>((resolve) => {
            const timer = setTimeout(() => {
              if (signal !== undefined) {
                signal.removeEventListener("abort", onAbort);
              }

              resolve();
            }, 10);

            const onAbort = () => {
              clearTimeout(timer);

              resolve();
            };

            if (signal !== undefined) {
              signal.addEventListener("abort", onAbort, { once: true });
            }
          });
        }

        // Yield remaining chunks
        while (chunks.length > 0) {
          yield chunks.shift();
        }

        // Surface abort as a terminal error
        if (signal?.aborted === true && !isComplete && !error) {
          throw new Error("Mirage stream aborted");
        }

        // If no data was received and no error, this might be due to invalid credentials
        if (!hasData && !error) {
          throw new Error(
              "No data received from Mirage API. This may be due to invalid credentials" +
              " or API configuration."
          );
        }

        if (error) {
          throw new Error(error.message);
        }
      };

      // Use the separate iterator
      const mirageChunks = createStreamIterator();

      for await (const chunk of mirageChunks) {
        if (options.signal?.aborted) {
          break;
        }

        let content = "";

        let toolCalls: ToolCallChunk[] = [];
        let logprobs: AnswerChatResponseLogprob | null = null;

        if ((chunk as AnswerChatResponseChunkAnswer)?.chunk) {
          content = (chunk as AnswerChatResponseChunkAnswer).chunk;
        }

        // @ts-expect-error: For better compatibility, tool calls are passed as an array
        if (chunk.tool_calls) {
          // @ts-expect-error: Same problem
          toolCalls = chunk.tool_calls as AnswerChatResponseToolCall[];
        }

        if ("logprobs" in chunk && chunk?.logprobs) {
          logprobs = chunk.logprobs as AnswerChatResponseLogprob;
        }

        // Update usage metadata if available

        const responseMetadata = logprobs ? { logprobs } : {};

        const generationChunk = new ChatGenerationChunk({
          text: content,
          message: new AIMessageChunk({
            content,
            tool_call_chunks: toolCalls.map(
              (toolCall: AnswerChatResponseToolCall) => {
                return {
                  id: toolCall.id,
                  name: toolCall.function?.name,
                  args: JSON.stringify(toolCall.function?.arguments || []),
                  type: "tool_call_chunk"
                };
              }
            ),
            response_metadata: responseMetadata
          }),
          generationInfo: usageMetadata
        });

        await runManager?.handleLLMNewToken(
            content,
            undefined,
            undefined,
            undefined,
            undefined,
            { chunk: generationChunk }
        );

        yield generationChunk;
      }
    } catch (error) {
      const errorMessage = (error instanceof Error)
        ? `Error: ${error.message}`
        : "Error: Unknown error";

      yield new ChatGenerationChunk({
        text: errorMessage,
        message: new AIMessageChunk({
          content: errorMessage
        })
      });

      await runManager?.handleLLMNewToken(errorMessage);
    }
  }

  /**
   * Add structured output support
   */
  withStructuredOutput(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    outputSchema: Record<string, any>, config : StructuredOutputMethodOptions<boolean> & {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      schema?: Record<string, any> | undefined
    } = {}
  ) {
    const outputSchemaIsZod = isInteropZodSchema(outputSchema);

    // Use schema from config if provided, otherwise use outputSchema
    const schemaToUse = (config.schema !== undefined) ? config.schema : outputSchema;

    // Convert to JSON schema format (if Zod, convert it; if already JSON schema, use as-is)
    const schemaToUseIsZod = isInteropZodSchema(schemaToUse);
    const jsonSchema = schemaToUseIsZod ? toJsonSchema(schemaToUse) : schemaToUse;

    const llm = this.withConfig({
      // @ts-expect-error: Format is not typed on langchain
      format: "json",
      structured_output: jsonSchema
    });

    const outputParser = outputSchemaIsZod
      ? StructuredOutputParser.fromZodSchema(outputSchema)
      : new JsonOutputParser();

    if (!config.includeRaw) {
      return llm.pipe(outputParser);
    }

    const parserAssign = RunnablePassthrough.assign({
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      parsed: (input: any, config: any) => {
        return outputParser.invoke(
          (input as { raw: string }).raw, config as RunnableConfig
        );
      }
    });

    const parserNone = RunnablePassthrough.assign({
      parsed: (): null => {
        return null;
      }
    });

    const parsedWithFallback = parserAssign.withFallbacks({
      fallbacks: [parserNone]
    });

    return RunnableSequence.from([
      {
        raw: llm
      },
      parsedWithFallback
    ]);
  }

  /**
   * Override withConfig to create a new ChatMirage instance with the updated fields
   */
  override withConfig(
    config: Partial<CallOptions>
  ): Runnable<BaseLanguageModelInput, AIMessageChunk, CallOptions> {
    return new ChatMirage<CallOptions>({ ...this.fields, ...config });
  }
}

/**************************************************************************
 * EXPORTS
 ***************************************************************************/

export { ChatMirage, ChatMirageCallOptions, ChatMirageInput };
