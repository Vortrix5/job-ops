import { logger } from "@infra/logger";
import { getOriginalEnvValue } from "@server/services/envSettings";

export const JEV_MODEL = "jev-1.13";
export const JEV_SCORING_VERSION = "jev-1.13-scoring-v1";
const SYSTEM_ONE_URL = "https://api.typesafe.ai/v1/systemone";

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
  constructor(message = "TypeSafe API key not configured") {
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

function parseResponse(value: unknown): SystemOneResponse {
  if (!isRecord(value) || !isRecord(value.answers)) {
    throw new SystemOneRequestError("TypeSafe returned an invalid response");
  }

  for (const answer of Object.values(value.answers)) {
    if (!isRecord(answer) || typeof answer.type !== "string") {
      throw new SystemOneRequestError("TypeSafe returned an invalid answer");
    }
  }

  return value as unknown as SystemOneResponse;
}

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function evaluateSystemOne(args: {
  state: unknown;
  questions: Record<string, SystemOneQuestion>;
  model?: string;
  jobId?: string;
  signal?: AbortSignal;
}): Promise<SystemOneResponse> {
  const apiKey = getOriginalEnvValue("TYPESAFE_API_KEY")?.trim();
  if (!apiKey) throw new SystemOneConfigurationError();

  const model = args.model ?? JEV_MODEL;
  const startedAt = Date.now();
  for (let attempt = 0; attempt < 3; attempt += 1) {
    try {
      const response = await fetch(SYSTEM_ONE_URL, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${apiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          state: args.state,
          model,
          questions: args.questions,
        }),
        signal: args.signal,
      });

      if (response.ok) {
        const result = parseResponse(await response.json());
        logger.info("TypeSafe System One evaluation completed", {
          jobId: args.jobId,
          model,
          durationMs: Date.now() - startedAt,
          answerCount: Object.keys(result.answers).length,
        });
        return result;
      }

      const retryable = response.status === 429 || response.status === 529;
      if (!retryable || attempt === 2) {
        throw new SystemOneRequestError(
          `TypeSafe API request failed with status ${response.status}`,
          response.status,
        );
      }
    } catch (error) {
      if (
        error instanceof SystemOneRequestError &&
        error.status !== 429 &&
        error.status !== 529
      ) {
        throw error;
      }
      if (attempt === 2) {
        if (error instanceof SystemOneRequestError) throw error;
        throw new SystemOneRequestError(
          `TypeSafe API request failed: ${error instanceof Error ? error.message : "unknown error"}`,
        );
      }
    }

    await wait(100 * 2 ** attempt);
  }

  throw new SystemOneRequestError("TypeSafe API request failed");
}
