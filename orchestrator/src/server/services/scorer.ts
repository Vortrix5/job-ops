/**
 * Service for scoring job suitability with TypeSafe Jev through OpenRouter.
 */

import { logger } from "@infra/logger";
import type { Job, UpdateJobInput } from "@shared/types";
import { stripHtmlTags } from "@shared/utils/string";
import { withHostedUsageReservation } from "./hosted-usage";
import { filterProfileProjectsForAi } from "./resumeProjects";
import { getEffectiveSettings } from "./settings";
import {
  evaluateSystemOne,
  JEV_MODEL,
  JEV_SCORING_VERSION,
  type SystemOneAnswer,
  SystemOneConfigurationError,
  type SystemOneQuestion,
  SystemOneRequestError,
} from "./system-one";

export { JEV_SCORING_VERSION } from "./system-one";

export class LlmNotConfiguredError extends Error {
  constructor(message?: string) {
    super(message ?? "OpenRouter API key not configured");
    this.name = "LlmNotConfiguredError";
  }
}

export class ScoringUnavailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ScoringUnavailableError";
  }
}

export interface SuitabilityResult {
  score: number | null;
  reason: string;
  jobBrief: string | null;
  jobUpdates?: UpdateJobInput;
  suitabilityConfidence?: number | null;
  suitabilityBreakdown?: string | null;
  suitabilityScoringVersion?: string;
}

type ProfileRecord = Record<string, unknown>;

const DIMENSIONS = [
  {
    key: "skills",
    label: "Skills",
    weight: 30,
    instructions:
      "How well do the candidate's demonstrated skills match the required skills and responsibilities?",
    criteria: [
      "No relevant skills or evidence",
      "Few relevant skills; major gaps",
      "Some relevant skills; meaningful gaps remain",
      "Most required skills are demonstrated",
      "Strong direct match across the required skills",
    ],
  },
  {
    key: "experience",
    label: "Experience",
    weight: 25,
    instructions:
      "How well does the candidate's demonstrated experience match the role's seniority, scope, and responsibilities?",
    criteria: [
      "No relevant experience or clear seniority mismatch",
      "Limited relevant experience; major scope mismatch",
      "Partially aligned experience or seniority",
      "Experience and seniority are mostly aligned",
      "Strong direct match in experience, scope, and seniority",
    ],
  },
  {
    key: "location",
    label: "Location",
    weight: 15,
    instructions:
      "How well do the job's location and work pattern fit the candidate's stated location and preferences? Treat unstated information as unknown, not as a rejection.",
    criteria: [
      "Known hard conflict with location or work pattern",
      "Likely conflict with limited evidence of compatibility",
      "Unclear or partly compatible location/work pattern",
      "Mostly compatible location/work pattern",
      "Clearly compatible location/work pattern or remote role",
    ],
  },
  {
    key: "domain",
    label: "Domain",
    weight: 15,
    instructions:
      "How well does the candidate's industry, product, and technical domain experience fit the job?",
    criteria: [
      "No relevant domain evidence",
      "Little relevant domain evidence",
      "Some transferable domain experience",
      "Relevant domain experience is evident",
      "Strong direct domain match",
    ],
  },
  {
    key: "preferences",
    label: "Preferences",
    weight: 15,
    instructions:
      "How well does the job fit the candidate's stated career direction and preferences, including role type and practical constraints?",
    criteria: [
      "Conflicts with stated career direction or hard preferences",
      "Several preference conflicts",
      "Mixed or mostly unknown preference fit",
      "Mostly aligned with stated preferences",
      "Strongly aligned with stated career direction and preferences",
    ],
  },
] as const;

function isSalaryMissing(job: Job): boolean {
  return (
    !job.salary?.trim() &&
    job.salaryMinAmount == null &&
    job.salaryMaxAmount == null
  );
}

function applySalaryPenalty(
  job: Job,
  originalScore: number,
  originalReason: string,
  settings: { penalizeMissingSalary: boolean; missingSalaryPenalty: number },
): { score: number; reason: string } {
  if (!settings.penalizeMissingSalary || !isSalaryMissing(job)) {
    return { score: originalScore, reason: originalReason };
  }

  const penalty = settings.missingSalaryPenalty;
  const adjustedScore = Math.max(0, originalScore - penalty);
  const reason = `${originalReason} Score reduced by ${penalty} points due to missing salary information.`;

  logger.info("Applied salary penalty", {
    jobId: job.id,
    originalScore,
    penalty,
    finalScore: adjustedScore,
  });

  return { score: adjustedScore, reason };
}

function buildJobState(job: Job): Record<string, unknown> {
  return {
    title: job.title,
    employer: job.employer,
    location: job.location,
    salary: job.salary,
    disciplines: job.disciplines,
    degreeRequired: job.degreeRequired,
    starting: job.starting,
    jobDescription: stripHtmlTags(job.jobDescription ?? "") || null,
    jobType: job.jobType,
    jobLevel: job.jobLevel,
    jobFunction: job.jobFunction,
    skills: job.skills,
    experienceRange: job.experienceRange,
    companyIndustry: job.companyIndustry,
    isRemote: job.isRemote,
    workFromHomeType: job.workFromHomeType,
  };
}

function buildQuestions(): Record<string, SystemOneQuestion> {
  return Object.fromEntries(
    DIMENSIONS.map((dimension) => [
      dimension.key,
      {
        type: "score",
        instructions: dimension.instructions,
        criteria: [...dimension.criteria],
      },
    ]),
  ) as Record<string, SystemOneQuestion>;
}

function scoreAnswer(
  answers: Record<string, SystemOneAnswer>,
  key: string,
): { score: number; confidence: number | null } {
  const answer = answers[key];
  if (!answer || answer.type !== "score" || !Number.isFinite(answer.score)) {
    throw new ScoringUnavailableError(
      `Jev returned an invalid ${key} suitability score`,
    );
  }

  return {
    score: Math.min(4, Math.max(0, answer.score)),
    confidence:
      typeof answer.confidence === "number" &&
      Number.isFinite(answer.confidence)
        ? Math.min(1, Math.max(0, answer.confidence))
        : null,
  };
}

export function composeJevScore(answers: Record<string, SystemOneAnswer>): {
  score: number;
  reason: string;
  confidence: number | null;
  breakdown: string;
} {
  const dimensions = DIMENSIONS.map((dimension) => ({
    ...dimension,
    answer: scoreAnswer(answers, dimension.key),
  }));
  const weightedScore = dimensions.reduce(
    (total, dimension) =>
      total + (dimension.weight * dimension.answer.score) / 4,
    0,
  );
  const confidences = dimensions
    .map((dimension) => dimension.answer.confidence)
    .filter((confidence): confidence is number => confidence !== null);
  const confidence =
    confidences.length > 0
      ? confidences.reduce((total, value) => total + value, 0) /
        confidences.length
      : null;

  const reason = dimensions
    .map(
      (dimension) => `${dimension.label}: ${fitLabel(dimension.answer.score)}`,
    )
    .join("; ");
  const breakdown = JSON.stringify({
    model: JEV_MODEL,
    version: JEV_SCORING_VERSION,
    weightedScore,
    confidence,
    dimensions: Object.fromEntries(
      dimensions.map((dimension) => [dimension.key, dimension.answer]),
    ),
  });

  return {
    score: Math.min(100, Math.max(0, Math.round(weightedScore))),
    reason,
    confidence,
    breakdown,
  };
}

function fitLabel(score: number): string {
  if (score >= 3.5) return "excellent";
  if (score >= 2.5) return "strong";
  if (score >= 1.5) return "moderate";
  if (score >= 0.5) return "weak";
  return "poor";
}

/** Score a job's suitability using one Jev call with five atomic dimensions. */
export async function scoreJobSuitability(
  job: Job,
  profile: Record<string, unknown>,
  options: { scoringInstructions?: string; skipHostedUsage?: boolean } = {},
): Promise<SuitabilityResult> {
  if (!options.skipHostedUsage) {
    return withHostedUsageReservation({ action: "tailoring" }, async () => ({
      result: await scoreJobSuitability(job, profile, {
        ...options,
        skipHostedUsage: true,
      }),
      usedUnits: 1,
    }));
  }

  const [settings, aiProfile] = await Promise.all([
    getEffectiveSettings(),
    filterProfileProjectsForAi(profile),
  ]);

  let response: Awaited<ReturnType<typeof evaluateSystemOne>>;
  try {
    response = await evaluateSystemOne({
      state: {
        candidate: sanitizeProfileForPrompt(aiProfile),
        job: buildJobState(job),
        scoringInstructions: options.scoringInstructions?.trim() || null,
      },
      questions: buildQuestions(),
      model: JEV_MODEL,
      jobId: job.id,
    });
  } catch (error) {
    if (
      error instanceof SystemOneConfigurationError ||
      (error instanceof SystemOneRequestError &&
        (error.status === 401 || error.status === 403))
    ) {
      logger.warn("Jev scoring unavailable — pausing pipeline", {
        jobId: job.id,
        error: error.message,
      });
      throw new LlmNotConfiguredError(
        `Jev scoring failed: ${error.message}. Set OPENROUTER_API_KEY or LLM_API_KEY, then resume scoring.`,
      );
    }

    logger.warn("Jev scoring failed", {
      jobId: job.id,
      error: error instanceof Error ? error.message : "unknown error",
    });
    throw new ScoringUnavailableError(
      `Jev scoring failed: ${error instanceof Error ? error.message : "unknown error"}`,
    );
  }

  const composed = composeJevScore(response.answers);
  const penaltyResult = applySalaryPenalty(
    job,
    composed.score,
    composed.reason,
    {
      penalizeMissingSalary: settings.penalizeMissingSalary.value,
      missingSalaryPenalty: settings.missingSalaryPenalty.value,
    },
  );

  return {
    score: penaltyResult.score,
    reason: penaltyResult.reason,
    jobBrief: null,
    jobUpdates: {},
    suitabilityConfidence: composed.confidence,
    suitabilityBreakdown: composed.breakdown,
    suitabilityScoringVersion: JEV_SCORING_VERSION,
  };
}

function sanitizeProfileForPrompt(
  profile: Record<string, unknown>,
): Record<string, unknown> {
  return {
    basics: sanitizeBasics(profile.basics),
    skills: sanitizeItems(profile, "skills", [
      "name",
      "description",
      "level",
      "proficiency",
      "keywords",
    ]),
    experience: sanitizeItems(profile, "experience", [
      "company",
      "position",
      "location",
      "date",
      "period",
      "summary",
      "description",
    ]),
    projects: sanitizeItems(profile, "projects", [
      "name",
      "description",
      "date",
      "period",
      "summary",
      "keywords",
    ]),
    education: sanitizeItems(profile, "education", [
      "school",
      "institution",
      "degree",
      "area",
      "grade",
      "location",
      "date",
      "period",
      "summary",
      "description",
    ]),
    languages: sanitizeItems(profile, "languages", [
      "language",
      "fluency",
      "level",
    ]),
    awards: sanitizeItems(profile, "awards", [
      "title",
      "awarder",
      "date",
      "summary",
      "description",
    ]),
    certifications: sanitizeItems(profile, "certifications", [
      "title",
      "issuer",
      "date",
      "summary",
      "description",
    ]),
    publications: sanitizeItems(profile, "publications", [
      "title",
      "publisher",
      "date",
      "summary",
      "description",
    ]),
    volunteer: sanitizeItems(profile, "volunteer", [
      "organization",
      "position",
      "location",
      "date",
      "period",
      "summary",
      "description",
    ]),
    interests: sanitizeItems(profile, "interests", [
      "name",
      "summary",
      "description",
      "keywords",
    ]),
  };
}

function sanitizeBasics(value: unknown): ProfileRecord {
  if (!isRecord(value)) return {};
  return pickDefined(value, ["label", "headline", "summary", "location"]);
}

function sanitizeItems(
  profile: ProfileRecord,
  sectionKey: string,
  allowedKeys: string[],
): ProfileRecord[] {
  return collectSectionItems(profile, sectionKey)
    .filter(isVisibleCvItem)
    .map((item) => sanitizeCvItem(item, allowedKeys))
    .filter((item) => Object.keys(item).length > 0);
}

function collectSectionItems(
  profile: ProfileRecord,
  sectionKey: string,
): ProfileRecord[] {
  const sections = isRecord(profile.sections) ? profile.sections : {};
  const section = sections[sectionKey];

  if (isRecord(section)) {
    if (!isVisibleCvItem(section)) return [];
    if (Array.isArray(section.items)) return section.items.filter(isRecord);
  }

  const topLevelSection = profile[sectionKey];
  if (Array.isArray(topLevelSection)) return topLevelSection.filter(isRecord);
  if (isRecord(topLevelSection)) {
    if (!isVisibleCvItem(topLevelSection)) return [];
    if (Array.isArray(topLevelSection.items)) {
      return topLevelSection.items.filter(isRecord);
    }
  }

  return [];
}

function sanitizeCvItem(
  item: ProfileRecord,
  allowedKeys: string[],
): ProfileRecord {
  const sanitized = pickDefined(item, allowedKeys);
  if (Array.isArray(item.roles)) {
    const roles = item.roles
      .filter(isRecord)
      .filter(isVisibleCvItem)
      .map((role) =>
        pickDefined(role, ["position", "period", "summary", "description"]),
      )
      .filter((role) => Object.keys(role).length > 0);
    if (roles.length > 0) sanitized.roles = roles;
  }
  return sanitized;
}

function pickDefined(source: ProfileRecord, keys: string[]): ProfileRecord {
  const result: ProfileRecord = {};
  for (const key of keys) {
    const value = source[key];
    if (value !== undefined && value !== null && value !== "") {
      result[key] = value;
    }
  }
  return result;
}

function isVisibleCvItem(item: ProfileRecord): boolean {
  return item.hidden !== true && item.visible !== false;
}

function isRecord(value: unknown): value is ProfileRecord {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

export async function scoreAndRankJobs(
  jobs: Job[],
  profile: Record<string, unknown>,
): Promise<
  Array<Job & { suitabilityScore: number | null; suitabilityReason: string }>
> {
  const scoredJobs = await Promise.all(
    jobs.map(async (job) => {
      const {
        score,
        reason,
        jobUpdates = {},
      } = await scoreJobSuitability(job, profile);
      return {
        ...job,
        ...jobUpdates,
        suitabilityScore: score,
        suitabilityReason: reason,
      };
    }),
  );

  return scoredJobs.sort((a, b) => {
    if (a.suitabilityScore == null && b.suitabilityScore == null) return 0;
    if (a.suitabilityScore == null) return 1;
    if (b.suitabilityScore == null) return -1;
    return b.suitabilityScore - a.suitabilityScore;
  });
}
