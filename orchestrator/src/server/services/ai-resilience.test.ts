import { createJob } from "@shared/testing/factories";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as settingsRepo from "../repositories/settings";
import { pickProjectIdsForJob } from "./projectSelection";
import { scoreJobSuitability } from "./scorer";

// --- Mocks ---
const settingsMocks = vi.hoisted(() => ({
  getSetting: vi.fn().mockResolvedValue(null),
  getAllSettings: vi.fn().mockResolvedValue({}),
  getEffectiveSettings: vi.fn(),
}));
const systemOneMock = vi.hoisted(() => vi.fn());

vi.mock("../repositories/settings", () => settingsMocks);
vi.mock("@server/repositories/settings", () => settingsMocks);
vi.mock("@server/services/settings", () => ({
  getEffectiveSettings: settingsMocks.getEffectiveSettings,
}));
vi.mock("./system-one", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./system-one")>();
  return { ...actual, evaluateSystemOne: systemOneMock };
});

function effectiveSettings(raw: Record<string, unknown>) {
  return {
    model: { value: raw.model ?? "gpt-4o-mini" },
    llmProvider: { value: raw.llmProvider ?? "openrouter" },
    llmBaseUrl: { value: raw.llmBaseUrl ?? null },
    llmPurposeOverrides: { value: {} },
    modelScorer: { value: null },
    modelProjectSelection: { value: null },
    scoringInstructions: { value: "" },
    scoringPromptTemplate: { value: null },
    penalizeMissingSalary: { value: false },
    missingSalaryPenalty: { value: 0 },
  };
}

// We need to mock 'fetch' globally for these tests
const globalFetch = global.fetch;

const mockJob = createJob({
  id: "test-job",
  source: "gradcracker",
  title: "Senior Engineer",
  employer: "Test Corp",
  jobDescription: "Looking for a TypeScript and React expert.",
  status: "discovered",
  suitabilityScore: null,
  suitabilityReason: null,
});

const mockProfile = { name: "Test User" };

describe("AI Service Resilience", () => {
  beforeEach(() => {
    global.fetch = vi.fn();
    delete process.env.LLM_API_KEY;
    process.env.OPENROUTER_API_KEY = "mock-key"; // Ensure logic tries to call API
    vi.mocked(settingsRepo.getAllSettings).mockResolvedValue({
      llmProvider: "openrouter",
      llmApiKey: "mock-key",
    });
    settingsMocks.getEffectiveSettings.mockImplementation(async () =>
      effectiveSettings(await settingsMocks.getAllSettings()),
    );
  });

  afterEach(() => {
    global.fetch = globalFetch;
    delete process.env.LLM_API_KEY;
    delete process.env.OPENROUTER_API_KEY;
    vi.restoreAllMocks();
  });

  describe("scoreJobSuitability (Jev)", () => {
    it("returns the weighted score from native Jev answers", async () => {
      systemOneMock.mockResolvedValue({
        model: "jev-1.13",
        answers: Object.fromEntries(
          ["skills", "experience", "location", "domain", "preferences"].map(
            (key) => [key, { type: "score", score: 3.4, confidence: 0.9 }],
          ),
        ),
      });

      const result = await scoreJobSuitability(mockJob, mockProfile);

      expect(result.score).toBe(85);
      expect(result.suitabilityConfidence).toBe(0.9);
      expect(global.fetch).not.toHaveBeenCalled();
    });

    it("turns native provider failures into retryable scoring failures", async () => {
      systemOneMock.mockRejectedValue(new Error("upstream unavailable"));

      await expect(scoreJobSuitability(mockJob, mockProfile)).rejects.toThrow(
        "Jev scoring failed: upstream unavailable",
      );
    });
  });

  describe("pickProjectIdsForJob (Project Selection)", () => {
    const mockProjects = [
      {
        id: "p1",
        name: "React App",
        description: "Used React",
        date: "2022",
        summaryText: "React stuff",
        isVisibleInBase: true,
      },
      {
        id: "p2",
        name: "Python Script",
        description: "Used Python",
        date: "2023",
        summaryText: "Python stuff",
        isVisibleInBase: true,
      },
    ];

    it("should return projects selected by AI", async () => {
      const mockResponse = {
        ok: true,
        json: async () => ({
          choices: [
            {
              message: {
                content: JSON.stringify({ selectedProjectIds: ["p1"] }),
              },
            },
          ],
        }),
      };
      vi.mocked(global.fetch).mockResolvedValue(mockResponse as any);

      const result = await pickProjectIdsForJob({
        jobDescription: "React dev",
        eligibleProjects: mockProjects,
        desiredCount: 1,
      });

      expect(result).toEqual(["p1"]);
    });

    it("should fallback if API fails", async () => {
      vi.mocked(global.fetch).mockRejectedValue(new Error("Network error"));

      const result = await pickProjectIdsForJob({
        jobDescription: "React dev", // Should match p1 due to keyword 'React'
        eligibleProjects: mockProjects,
        desiredCount: 1,
      });

      // It should fall back to keyword matching
      // p1 has 'React', p2 has 'Python'. 'React dev' matches p1.
      expect(result).toEqual(["p1"]);
    });

    it("should fallback if AI returns garbage", async () => {
      const mockResponse = {
        ok: true,
        json: async () => ({
          choices: [{ message: { content: "No valid JSON here" } }],
        }),
      };
      vi.mocked(global.fetch).mockResolvedValue(mockResponse as any);

      const result = await pickProjectIdsForJob({
        jobDescription: "Python dev", // Should match p2
        eligibleProjects: mockProjects,
        desiredCount: 1,
      });

      expect(result).toEqual(["p2"]);
    });

    it("should validate returned IDs exist in eligible list", async () => {
      // AI returns an ID that doesn't exist ('p999')
      const mockResponse = {
        ok: true,
        json: async () => ({
          choices: [
            {
              message: {
                content: JSON.stringify({ selectedProjectIds: ["p999", "p1"] }),
              },
            },
          ],
        }),
      };
      vi.mocked(global.fetch).mockResolvedValue(mockResponse as any);

      const result = await pickProjectIdsForJob({
        jobDescription: "stuff",
        eligibleProjects: mockProjects,
        desiredCount: 2,
      });

      // Invalid IDs are removed and deterministic fallback fills the slot.
      expect(result).toEqual(["p1", "p2"]);
    });

    it("tops up a short AI ranking deterministically", async () => {
      vi.mocked(global.fetch).mockResolvedValue({
        ok: true,
        json: async () => ({
          choices: [
            {
              message: {
                content: JSON.stringify({ selectedProjectIds: ["p2"] }),
              },
            },
          ],
        }),
      } as any);

      const result = await pickProjectIdsForJob({
        jobDescription: "React dev",
        eligibleProjects: [
          ...mockProjects,
          {
            id: "p3",
            name: "Project 3",
            description: "Other",
            summaryText: "Other",
            date: "2024",
            isVisibleInBase: false,
          },
        ],
        desiredCount: 3,
      });

      expect(result).toEqual(["p2", "p1", "p3"]);
    });
  });
});
