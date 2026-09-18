import { createJob } from "@shared/testing/factories";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { evaluateSystemOneMock, getEffectiveSettingsMock } = vi.hoisted(() => ({
  evaluateSystemOneMock: vi.fn(),
  getEffectiveSettingsMock: vi.fn(),
}));

vi.mock("./system-one", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./system-one")>();
  return { ...actual, evaluateSystemOne: evaluateSystemOneMock };
});

vi.mock("./settings", () => ({
  getEffectiveSettings: getEffectiveSettingsMock,
}));

vi.mock("./resumeProjects", () => ({
  filterProfileProjectsForAi: vi.fn(async (profile) => profile),
}));

vi.mock("./hosted-usage", () => ({
  withHostedUsageReservation: vi.fn(
    async (_args, work) => (await work()).result,
  ),
}));

import {
  composeJevScore,
  JEV_SCORING_VERSION,
  scoreJobSuitability,
} from "./scorer";
import type { SystemOneAnswer } from "./system-one";

function answers(
  scores: Record<string, number>,
  confidence = 0.8,
): Record<string, SystemOneAnswer> {
  return Object.fromEntries(
    Object.entries(scores).map(([key, score]) => [
      key,
      { type: "score", score, confidence } satisfies SystemOneAnswer,
    ]),
  ) as Record<string, SystemOneAnswer>;
}

describe("Jev scoring", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    getEffectiveSettingsMock.mockResolvedValue({
      penalizeMissingSalary: { value: false },
      missingSalaryPenalty: { value: 10 },
    });
  });

  it("combines Jev dimensions using the existing weighted 0-100 score", () => {
    const result = composeJevScore(
      answers({
        skills: 4,
        experience: 3,
        location: 2,
        domain: 1,
        preferences: 0,
      }),
    );

    expect(result.score).toBe(60);
    expect(result.reason).toBe(
      "Skills: excellent; Experience: strong; Location: moderate; Domain: weak; Preferences: poor",
    );
    expect(result.confidence).toBe(0.8);
    expect(JSON.parse(result.breakdown)).toMatchObject({
      model: "typesafe/jev-1.13",
      version: JEV_SCORING_VERSION,
      weightedScore: 60,
    });
  });

  it("sends one Jev request with sanitized state and five score questions", async () => {
    evaluateSystemOneMock.mockResolvedValue({
      model: "typesafe/jev-1.13",
      answers: answers({
        skills: 4,
        experience: 4,
        location: 4,
        domain: 4,
        preferences: 4,
      }),
    });

    const result = await scoreJobSuitability(
      createJob({ jobDescription: "<p>Build <strong>APIs</strong>.</p>" }),
      {
        basics: { summary: "Backend engineer", email: "private@example.com" },
      },
      {
        skipHostedUsage: true,
        scoringInstructions: "Prefer platform engineering roles.",
      },
    );

    expect(evaluateSystemOneMock).toHaveBeenCalledTimes(1);
    const request = evaluateSystemOneMock.mock.calls[0][0];
    expect(request.model).toBe("typesafe/jev-1.13");
    expect(request.questions).toEqual(
      expect.objectContaining({
        skills: expect.objectContaining({ type: "score" }),
        experience: expect.objectContaining({ type: "score" }),
        location: expect.objectContaining({ type: "score" }),
        domain: expect.objectContaining({ type: "score" }),
        preferences: expect.objectContaining({ type: "score" }),
      }),
    );
    expect(request.state).toMatchObject({
      candidate: {
        basics: { summary: "Backend engineer" },
        skills: [],
        experience: [],
        projects: [],
        education: [],
        languages: [],
        awards: [],
        certifications: [],
        publications: [],
        volunteer: [],
        interests: [],
      },
      job: { jobDescription: "Build APIs ." },
      scoringInstructions: "Prefer platform engineering roles.",
    });
    expect(JSON.stringify(request.state)).not.toContain("private@example.com");
    expect(result.score).toBe(100);
    expect(result.jobBrief).toBeNull();
    expect(result.suitabilityScoringVersion).toBe(JEV_SCORING_VERSION);
  });

  it("keeps the deterministic missing-salary penalty", async () => {
    getEffectiveSettingsMock.mockResolvedValue({
      penalizeMissingSalary: { value: true },
      missingSalaryPenalty: { value: 10 },
    });
    evaluateSystemOneMock.mockResolvedValue({
      model: "typesafe/jev-1.13",
      answers: answers({
        skills: 4,
        experience: 4,
        location: 4,
        domain: 4,
        preferences: 4,
      }),
    });

    const result = await scoreJobSuitability(
      createJob({ salary: null, salaryMinAmount: null, salaryMaxAmount: null }),
      {},
      { skipHostedUsage: true },
    );

    expect(result.score).toBe(90);
    expect(result.reason).toContain("missing salary information");
  });
});
