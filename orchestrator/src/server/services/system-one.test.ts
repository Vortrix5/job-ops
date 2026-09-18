import { afterEach, describe, expect, it, vi } from "vitest";
import {
  evaluateSystemOne,
  SystemOneConfigurationError,
  type SystemOneRequestError,
} from "./system-one";

const originalFetch = global.fetch;
const originalOpenRouterApiKey = process.env.OPENROUTER_API_KEY;
const originalLlmApiKey = process.env.LLM_API_KEY;

afterEach(() => {
  global.fetch = originalFetch;
  if (originalOpenRouterApiKey === undefined)
    delete process.env.OPENROUTER_API_KEY;
  else process.env.OPENROUTER_API_KEY = originalOpenRouterApiKey;
  if (originalLlmApiKey === undefined) delete process.env.LLM_API_KEY;
  else process.env.LLM_API_KEY = originalLlmApiKey;
  vi.restoreAllMocks();
});

describe("OpenRouter Jev client", () => {
  it("sends a structured OpenRouter request and returns typed answers", async () => {
    process.env.OPENROUTER_API_KEY = "test-openrouter-key";
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        choices: [
          {
            message: {
              content: JSON.stringify({
                answers: {
                  skills: { score: 3.5, confidence: 0.9 },
                },
              }),
            },
          },
        ],
      }),
    }) as typeof fetch;

    const result = await evaluateSystemOne({
      state: { candidate: {}, job: {} },
      questions: {
        skills: {
          type: "score",
          instructions: "How strong is the fit?",
          criteria: ["poor", "good"],
        },
      },
      jobId: "job-1",
    });

    expect(result.answers.skills).toMatchObject({ score: 3.5 });
    expect(global.fetch).toHaveBeenCalledWith(
      "https://openrouter.ai/api/v1/chat/completions",
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({
          Authorization: "Bearer test-openrouter-key",
        }),
        body: expect.stringContaining('"model":"typesafe/jev-1.13"'),
      }),
    );
  });

  it("retries rate limits and gateway overloads", async () => {
    process.env.OPENROUTER_API_KEY = "test-openrouter-key";
    global.fetch = vi
      .fn()
      .mockResolvedValueOnce({
        ok: false,
        status: 429,
        text: async () => "rate limited",
        headers: { get: () => null },
      })
      .mockResolvedValueOnce({
        ok: false,
        status: 529,
        text: async () => "overloaded",
        headers: { get: () => null },
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          choices: [
            {
              message: {
                content: JSON.stringify({ answers: {} }),
              },
            },
          ],
        }),
      }) as typeof fetch;

    await expect(
      evaluateSystemOne({ state: {}, questions: {} }),
    ).resolves.toMatchObject({ answers: {} });
    expect(global.fetch).toHaveBeenCalledTimes(3);
  });

  it("fails before making a request without credentials", async () => {
    process.env.OPENROUTER_API_KEY = "";
    process.env.LLM_API_KEY = "";
    global.fetch = vi.fn() as typeof fetch;

    await expect(
      evaluateSystemOne({ state: {}, questions: {} }),
    ).rejects.toBeInstanceOf(SystemOneConfigurationError);
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it("does not retry non-transient API errors", async () => {
    process.env.OPENROUTER_API_KEY = "test-openrouter-key";
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 400,
      text: async () => "bad request",
      headers: { get: () => null },
    }) as typeof fetch;

    await expect(
      evaluateSystemOne({ state: {}, questions: {} }),
    ).rejects.toEqual(
      expect.objectContaining({
        name: "SystemOneRequestError",
        status: 400,
      } satisfies Partial<SystemOneRequestError>),
    );
    expect(global.fetch).toHaveBeenCalledTimes(1);
  });
});
