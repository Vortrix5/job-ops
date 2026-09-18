import { logger } from "@infra/logger";
import { resolveLlmApiKey } from "@server/services/llm/credentials";
import { LlmService } from "@server/services/llm/service";
import type { JsonSchemaDefinition } from "@server/services/llm/types";

export const JEV_MODEL = "typesafe/jev-1.13";
export const JEV_SCORING_VERSION = "openrouter-typesafe-jev-1.13-scoring-v1";

export type SystemOneQuestion =
  | {
      type: "choice";
      instructions: string;
      criteria: Record<string, string | null>;
    }
  | {
      type: "score";
      instructions: string;
      criteria: string[];
    }
  | {
      type: "noul";
      instructions: string;
    };

export type SystemOneAnswer =
  | {
      type: "choice";
      choice: string;
      probabilities?: Record<string, number>;
      confidence?: number;
    }
  | {
      type: "score";
      score: number;
      legend?: Record<string, string>;
      confidence?: number;
    }
  | {
      type: "noul";
      noul: number;
    };

export type SystemOneResponse = {
  model: string;
  answers: Record<string, SystemOneAnswer>;
  request_id?: string;
  usage?: Record<string, unknown>;
};

export class SystemOneConfigurationError extends Error {
  constructor(message = "OpenRouter API key not configured") {
    super(message);
    this.name = "SystemOneConfigurationError";
  }
}

export class SystemOneRequestError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message);
    this.name = "SystemOneRequestError";
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

type OpenRouterJevResponse = {
  answers: Record<string, { score: number; confidence: number }>;
};

const JEV_RESPONSE_SCHEMA: JsonSchemaDefinition = {
  name: "jev_suitability_scores",
  schema: {
    type: "object",
    properties: {
      answers: {
        type: "object",
        properties: Object.fromEntries(
          ["skills", "experience", "location", "domain", "preferences"].map(
            (key) => [
              key,
              {
                type: "object",
                properties: {
                  score: { type: "number", minimum: 0, maximum: 4 },
                  confidence: { type: "number", minimum: 0, maximum: 1 },
                },
                required: ["score", "confidence"],
                additionalProperties: false,
              },
            ],
          ),
        ),
        required: ["skills", "experience", "location", "domain", "preferences"],
        additionalProperties: false,
      },
    },
    required: ["answers"],
    additionalProperties: false,
  },
};

function buildPrompt(args: {
  state: unknown;
  questions: Record<string, SystemOneQuestion>;
}): string {
  return [
    "Evaluate the candidate's fit for the job using the supplied scoring dimensions.",
    "Score every dimension from 0 to 4 and confidence from 0 to 1.",
    "Use only evidence in the candidate and job data. Treat missing information as unknown.",
    "Return only the JSON object required by the response schema; do not include explanations.",
    JSON.stringify({ state: args.state, questions: args.questions }),
  ].join("\n\n");
}

function parseOpenRouterResponse(value: unknown): SystemOneResponse {
  if (!isRecord(value) || !isRecord(value.answers)) {
    throw new SystemOneRequestError(
      "OpenRouter returned an invalid Jev response",
    );
  }

  const answers: Record<string, SystemOneAnswer> = {};
  for (const [key, answer] of Object.entries(value.answers)) {
    if (
      !isRecord(answer) ||
      typeof answer.score !== "number" ||
      !Number.isFinite(answer.score)
    ) {
      throw new SystemOneRequestError(
        `OpenRouter returned an invalid Jev answer for ${key}`,
      );
    }
    answers[key] = {
      type: "score",
      score: answer.score,
      ...(typeof answer.confidence === "number"
        ? { confidence: answer.confidence }
        : {}),
    };
  }

  return { model: JEV_MODEL, answers };
}

export async function evaluateSystemOne(args: {
  state: unknown;
  questions: Record<string, SystemOneQuestion>;
  model?: string;
  jobId?: string;
  signal?: AbortSignal;
}): Promise<SystemOneResponse> {
  const apiKey = resolveLlmApiKey({ provider: "openrouter" });
  if (!apiKey) throw new SystemOneConfigurationError();

  const model = args.model ?? JEV_MODEL;
  const startedAt = Date.now();
  const llm = new LlmService({ provider: "openrouter", apiKey });
  const result = await llm.callJson<OpenRouterJevResponse>({
    model,
    messages: [
      {
        role: "system",
        content:
          "You are Jev, a calibrated job-fit scoring model. Produce numeric scores only.",
      },
      {
        role: "user",
        content: buildPrompt({ state: args.state, questions: args.questions }),
      },
    ],
    jsonSchema: JEV_RESPONSE_SCHEMA,
    maxRetries: 2,
    retryDelayMs: 100,
    jobId: args.jobId,
    signal: args.signal,
  });

  if (!result.success) {
    const status = Number(result.error.match(/LLM API error: (\d+)/)?.[1]);
    throw new SystemOneRequestError(
      `OpenRouter Jev request failed: ${result.error}`,
      Number.isFinite(status) ? status : undefined,
    );
  }

  const response = parseOpenRouterResponse(result.data);
  logger.info("OpenRouter Jev evaluation completed", {
    jobId: args.jobId,
    model,
    durationMs: Date.now() - startedAt,
    answerCount: Object.keys(response.answers).length,
  });
  return response;
}
