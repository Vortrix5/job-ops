import { afterEach, describe, expect, it, vi } from "vitest";
import {
  evaluateSystemOne,
  JEV_MODEL,
  SystemOneConfigurationError,
  type SystemOneRequestError,
} from "./system-one";

const originalFetch = global.fetch;
const originalApiKey = process.env.TYPESAFE_API_KEY;

afterEach(() => {
  global.fetch = originalFetch;
  if (originalApiKey === undefined) delete process.env.TYPESAFE_API_KEY;
  else process.env.TYPESAFE_API_KEY = originalApiKey;
  vi.restoreAllMocks();
});

describe("TypeSafe System One client", () => {
  it("sends the native Jev request and returns typed answers", async () => {
    process.env.TYPESAFE_API_KEY = "test-typesafe-key";
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        model: JEV_MODEL,
        answers: {
          skills: { type: "score", score: 3.5, confidence: 0.9 },
        },
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
      "https://api.typesafe.ai/v1/systemone",
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({
          Authorization: "Bearer test-typesafe-key",
        }),
        body: expect.stringContaining('"model":"jev-1.13"'),
      }),
    );
  });

  it("retries rate limits and gateway overloads", async () => {
    process.env.TYPESAFE_API_KEY = "test-typesafe-key";
    global.fetch = vi
      .fn()
      .mockResolvedValueOnce({ ok: false, status: 429 })
      .mockResolvedValueOnce({ ok: false, status: 529 })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ model: JEV_MODEL, answers: {} }),
      }) as typeof fetch;

    await expect(
      evaluateSystemOne({ state: {}, questions: {} }),
    ).resolves.toMatchObject({ answers: {} });
    expect(global.fetch).toHaveBeenCalledTimes(3);
  });

  it("fails before making a request without credentials", async () => {
    delete process.env.TYPESAFE_API_KEY;
    global.fetch = vi.fn() as typeof fetch;

    await expect(
      evaluateSystemOne({ state: {}, questions: {} }),
    ).rejects.toBeInstanceOf(SystemOneConfigurationError);
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it("does not retry non-transient API errors", async () => {
    process.env.TYPESAFE_API_KEY = "test-typesafe-key";
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 400,
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
