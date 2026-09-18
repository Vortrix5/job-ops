import { logger } from "@infra/logger";
import { buildHeaders, joinUrl } from "@server/services/llm/utils/http";
import { resolveLlmRuntimeSettings } from "@server/services/modelSelection";

export const JEV_MODEL = "typesafe/jev-1.13";
export const JEV_SCORING_VERSION = "openrouter-typesafe-jev-1.13-decisions-v1";
const DECISIONS_PATH = "/api/alpha/decisions";
const DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai";

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

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function parseDecisionsResponse(value: unknown): SystemOneResponse {
  if (!isRecord(value) || !isRecord(value.answers)) {
    throw new SystemOneRequestError(
      "OpenRouter returned an invalid Jev decisions response",
    );
  }

  const answers: Record<string, SystemOneAnswer> = {};
  for (const [key, answer] of Object.entries(value.answers)) {
    if (
      !isRecord(answer) ||
      answer.type !== "score" ||
      typeof answer.score !== "number" ||
      !Number.isFinite(answer.score)
    ) {
      throw new SystemOneRequestError(
        `OpenRouter returned an invalid Jev decisions answer for ${key}`,
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

  return {
    model: typeof value.model === "string" ? value.model : JEV_MODEL,
    answers,
    ...(typeof value.id === "string" ? { request_id: value.id } : {}),
    ...(isRecord(value.usage) ? { usage: value.usage } : {}),
  };
}

function isRetryableStatus(status: number): boolean {
  return status === 429 || status >= 500;
}

export async function evaluateSystemOne(args: {
  state: unknown;
  questions: Record<string, SystemOneQuestion>;
  model?: string;
  jobId?: string;
  signal?: AbortSignal;
}): Promise<SystemOneResponse> {
  const runtime = await resolveLlmRuntimeSettings("scoring");
  if (runtime.provider !== "openrouter") {
    throw new SystemOneConfigurationError(
      `Jev scoring requires OpenRouter; scoring is configured for ${runtime.provider ?? "no provider"}`,
    );
  }
  if (!runtime.apiKey) throw new SystemOneConfigurationError();

  const model = args.model ?? JEV_MODEL;
  const startedAt = Date.now();
  const url = joinUrl(
    runtime.baseUrl || DEFAULT_OPENROUTER_BASE_URL,
    DECISIONS_PATH,
  );

  for (let attempt = 0; attempt < 3; attempt += 1) {
    try {
      const response = await fetch(url, {
        method: "POST",
        headers: buildHeaders({
          apiKey: runtime.apiKey,
          provider: "openrouter",
        }),
        body: JSON.stringify({
          model,
          state: args.state,
          questions: args.questions,
        }),
        signal: args.signal,
      });

      if (response.ok) {
        const result = parseDecisionsResponse(await response.json());
        logger.info("OpenRouter Jev decisions evaluation completed", {
          jobId: args.jobId,
          model,
          durationMs: Date.now() - startedAt,
          answerCount: Object.keys(result.answers).length,
        });
        return result;
      }

      if (!isRetryableStatus(response.status) || attempt === 2) {
        throw new SystemOneRequestError(
          `OpenRouter Jev decisions request failed with status ${response.status}`,
          response.status,
        );
      }
    } catch (error) {
      if (error instanceof SystemOneRequestError) {
        if (!isRetryableStatus(error.status ?? 0) || attempt === 2) {
          throw error;
        }
      } else if (attempt === 2) {
        throw new SystemOneRequestError(
          `OpenRouter Jev decisions request failed: ${error instanceof Error ? error.message : "unknown error"}`,
        );
      }
    }

    await wait(100 * 2 ** attempt);
  }

  throw new SystemOneRequestError("OpenRouter Jev decisions request failed");
}
