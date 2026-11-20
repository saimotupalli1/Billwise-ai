import "dotenv/config";
import express, { Request, Response } from "express";
import path from "path";
import { fileURLToPath } from "url";
import { GoogleGenAI, ThinkingLevel, Type } from "@google/genai";
import { createServer as createViteServer } from "vite";
import { extractTextItems, getDocumentProxy, inspectPages } from "unpdf";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT) || 3000;
const MODEL_NAME = "gemini-4-hydrogen";
const FALLBACK_MODEL_NAME = "gemini-3.1-flash-lite";

const ALLOWED_MIME_TYPES = new Set([
  "application/pdf",
  "image/png",
  "image/jpeg",
  "image/jpg",
  "image/webp",
]);

function getAiClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error("GEMINI_API_KEY is not configured in environment secrets.");
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        "User-Agent": "aistudio-build",
      },
    },
  });
}

async function withRetry<T>(fn: () => Promise<T>, maxRetries = 2): Promise<T> {
  let lastError: unknown;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (err: unknown) {
      lastError = err;
      const msg = err instanceof Error ? err.message : String(err);
      const isTransient =
        msg.includes("503") ||
        msg.includes("500") ||
        msg.includes("UNAVAILABLE") ||
        msg.includes("DEADLINE_EXCEEDED");
      if (!isTransient || attempt === maxRetries) {
        throw err;
      }
      await new Promise((resolve) => setTimeout(resolve, 600 * (attempt + 1)));
    }
  }
  throw lastError;
}

function isQuotaError(err: unknown): boolean {
  const msg = err instanceof Error ? err.message : String(err);
  return msg.includes("RESOURCE_EXHAUSTED") || msg.includes("429");
}

async function generateWithModelFallback(
  ai: GoogleGenAI,
  request: {
    contents: unknown;
    systemInstruction: string;
    responseSchema: Record<string, unknown>;
  }
) {
  try {
    return await withRetry(() =>
      ai.models.generateContent({
        model: MODEL_NAME,
        contents: request.contents as never,
        config: {
          systemInstruction: request.systemInstruction,
          thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
          responseMimeType: "application/json",
          responseSchema: request.responseSchema as never,
        },
      })
    );
  } catch (err) {
    if (!isQuotaError(err)) {
      throw err;
    }
    console.warn(
      `Primary model ${MODEL_NAME} quota exhausted (429); falling back to ${FALLBACK_MODEL_NAME}.`
    );
    return await withRetry(() =>
      ai.models.generateContent({
        model: FALLBACK_MODEL_NAME,
        contents: request.contents as never,
        config: {
          systemInstruction: request.systemInstruction,
          thinkingConfig: { thinkingLevel: ThinkingLevel.MINIMAL },
          responseMimeType: "application/json",
          responseSchema: request.responseSchema as never,
        },
      })
    );
  }
}

function formatApiError(err: unknown): { status: number; message: string } {
  const raw = err instanceof Error ? err.message : "Unexpected server error.";
  if (raw.includes("GEMINI_API_KEY is not configured")) {
    return {
      status: 500,
      message: "Gemini API key is missing. Please verify it in Settings > Secrets.",
    };
  }
  if (raw.includes("API_KEY_INVALID") || raw.includes("PERMISSION_DENIED") || raw.includes("403")) {
    return {
      status: 403,
      message: "API key permission error. Please check your Gemini API key in Settings > Secrets.",
    };
  }
  if (raw.includes("RESOURCE_EXHAUSTED") || raw.includes("429")) {
    const isDailyLimit =
      raw.includes("PerDay") || raw.includes("per day") || raw.includes("limit: 20");
    return {
      status: 429,
      message: isDailyLimit
        ? "Daily free-tier Gemini API quota exceeded. Please wait for the quota reset or select a billing-enabled API key in Settings > Secrets."
        : "Rate limit reached. Please wait a moment and try again.",
    };
  }
  if (raw.includes("400") || raw.includes("INVALID_ARGUMENT")) {
    return {
      status: 400,
      message: "The uploaded file could not be processed. Try a clearer PDF or image (PNG/JPG/WEBP).",
    };
  }
  return { status: 500, message: raw };
}

function languageInstructionText(lang: string): string {
  switch (lang) {
    case "te":
      return "Write the summary in natural, clear Telugu script (తెలుగు). Keep numbers, consumer IDs, and currency amounts accurate.";
    case "hi":
      return "Write the summary in natural, clear Hindi using Devanagari script (हिन्दी). Keep numbers, consumer IDs, and currency amounts accurate.";
    case "tenglish":
      return "Write the summary in natural Tenglish (conversational Telugu written in English/Latin letters, e.g., 'Mee electricity bill total ₹1,450 undi, due date 24 Nov.'). Keep numbers and labels accurate.";
    case "en":
    case "auto":
    default:
      return "Write the summary in clear, everyday English.";
  }
}

async function extractEmbeddedPdfText(pdfBuffer: Buffer): Promise<{
  isTextBased: boolean;
  text: string;
  totalPages: number;
}> {
  let pdf: Awaited<ReturnType<typeof getDocumentProxy>> | null = null;
  try {
    pdf = await getDocumentProxy(new Uint8Array(pdfBuffer));
    const [inspection, itemsResult] = await Promise.all([
      inspectPages(pdf),
      extractTextItems(pdf),
    ]);

    const pageTexts: string[] = [];
    for (let p = 0; p < itemsResult.items.length; p++) {
      const pageItems = itemsResult.items[p].filter((it) => it.str.trim().length > 0);
      if (pageItems.length === 0) continue;

      // Sort top-to-bottom (y descending in PDF coordinates) then left-to-right (x ascending)
      pageItems.sort((a, b) => {
        const yDiff = b.y - a.y;
        const tol = Math.max(4, Math.min(a.height || 10, b.height || 10) * 0.45);
        if (Math.abs(yDiff) > tol) return yDiff;
        return a.x - b.x;
      });

      const lines: string[] = [];
      let currentRowY: number | null = null;
      let currentTol = 5;
      let rowTokens: Array<{ x: number; text: string }> = [];

      const flushRow = () => {
        if (rowTokens.length === 0) return;
        rowTokens.sort((a, b) => a.x - b.x);
        lines.push(rowTokens.map((t) => t.text).join("  "));
        rowTokens = [];
      };

      for (const item of pageItems) {
        const tol = Math.max(4, (item.height || item.fontSize || 10) * 0.45);
        if (currentRowY === null || Math.abs(item.y - currentRowY) <= Math.max(currentTol, tol)) {
          if (currentRowY === null) {
            currentRowY = item.y;
            currentTol = tol;
          }
          rowTokens.push({ x: item.x, text: item.str.trim() });
        } else {
          flushRow();
          currentRowY = item.y;
          currentTol = tol;
          rowTokens.push({ x: item.x, text: item.str.trim() });
        }
      }
      flushRow();

      pageTexts.push(`--- Page ${p + 1} ---\n${lines.join("\n")}`);
    }

    const mergedText = pageTexts.join("\n\n").trim();
    const visibleTextLength = inspection.pages.reduce(
      (sum, pg) => sum + Math.max(0, pg.textLength - pg.invisibleTextLength),
      0
    );
    const alnumCount = (mergedText.match(/[\p{L}\p{N}]/gu) || []).length;
    const replacementCount = (mergedText.match(/\uFFFD/g) || []).length;
    const garbledRatio = mergedText.length > 0 ? replacementCount / mergedText.length : 1;

    const isTextBased =
      alnumCount >= 60 && visibleTextLength >= 60 && garbledRatio < 0.08;

    return {
      isTextBased,
      text: mergedText,
      totalPages: itemsResult.totalPages,
    };
  } catch (err) {
    console.warn("Embedded PDF text inspection failed, falling back to PDF OCR:", err);
    return { isTextBased: false, text: "", totalPages: 0 };
  } finally {
    if (pdf && "cleanup" in pdf && typeof pdf.cleanup === "function") {
      try {
        await pdf.cleanup();
      } catch {
        // Ignore cleanup errors
      }
    }
  }
}

interface AnalyzeBillBody {
  fileName?: string;
  mimeType?: string;
  originalMimeType?: string;
  base64Data?: string;
  slotId?: "bill1" | "bill2";
  slotLabel?: string;
  languagePreference?: string;
  preprocessingApplied?: string[];
}

type EditableBillFieldKey =
  | "utilityProvider"
  | "consumerName"
  | "consumerNumber"
  | "serviceCategory"
  | "billNumber"
  | "billDate"
  | "billingPeriod"
  | "dueDate"
  | "disconnectionDate"
  | "previousReading"
  | "currentReading"
  | "unitsConsumed"
  | "fixedCharges"
  | "energyCharges"
  | "taxesAndAdditionalCharges"
  | "totalAmountPayable";

interface ChatBillInput {
  slotId: string;
  slotLabel: string;
  fileName: string;
  extractionMethod?: string;
  ocrConfidence?: "High" | "Medium" | "Low" | null;
  documentConfidenceNote?: string | null;
  isElectricityBill: boolean;
  utilityProvider: string | null;
  consumerName: string | null;
  consumerNumber: string | null;
  serviceCategory: string | null;
  billNumber: string | null;
  billDate: string | null;
  billingPeriod: string | null;
  dueDate: string | null;
  disconnectionDate: string | null;
  previousReading: string | null;
  currentReading: string | null;
  unitsConsumed: string | null;
  fixedCharges: string | null;
  energyCharges: string | null;
  taxesAndAdditionalCharges: string | null;
  totalAmountPayable: string | null;
  chargesBreakdown: Array<{
    name: string;
    amount: string;
    simpleExplanation: string;
  }>;
  unavailableFields: string[];
  verifiedValues?: Partial<Record<EditableBillFieldKey, string | null>>;
  userCorrectedFields?: string[];
  rawOcrText: string;
}

interface ChatBody {
  message?: string;
  languagePreference?: "auto" | "en" | "te" | "hi" | "tenglish";
  bills?: ChatBillInput[];
  history?: Array<{
    role: "user" | "assistant";
    content: string;
  }>;
}

function isFieldVerified(b: ChatBillInput, key: EditableBillFieldKey): boolean {
  return Boolean(
    b.verifiedValues && Object.prototype.hasOwnProperty.call(b.verifiedValues, key)
  );
}

function getEffectiveField(b: ChatBillInput, key: EditableBillFieldKey): string | null {
  if (isFieldVerified(b, key)) {
    return b.verifiedValues?.[key] ?? null;
  }
  return b[key] ?? null;
}

function formatFieldWithProvenance(
  b: ChatBillInput,
  key: EditableBillFieldKey
): string {
  const original = b[key] ?? null;
  if (isFieldVerified(b, key)) {
    const verified = b.verifiedValues?.[key] ?? null;
    return `${verified ?? "UNAVAILABLE"} [USER-VERIFIED | Original OCR was: ${original ?? "UNAVAILABLE"}]`;
  }
  return original ?? "UNAVAILABLE";
}

function parseNumericValue(raw: string | null | undefined): number | null {
  if (!raw) return null;
  const cleaned = raw.replace(/,/g, "");
  const matches = cleaned.match(/-?\d+(?:\.\d+)?/g);
  if (!matches || matches.length !== 1) return null;
  const num = Number(matches[0]);
  return Number.isFinite(num) ? num : null;
}

function round2(n: number): number {
  return Math.round(n * 100) / 100;
}

function tryParseBillDateMs(b: ChatBillInput): number | null {
  const candidates = [
    getEffectiveField(b, "billDate"),
    getEffectiveField(b, "dueDate"),
    getEffectiveField(b, "billingPeriod"),
  ];
  for (const raw of candidates) {
    if (!raw) continue;
    const d1 = Date.parse(raw);
    if (!Number.isNaN(d1)) return d1;
    // Try extracting end date from a range like "01 Aug 2026 to 31 Aug 2026"
    const parts = raw.split(/\bto\b|\s*[-–—]\s*/i);
    const tail = parts[parts.length - 1]?.trim();
    if (tail) {
      const d2 = Date.parse(tail);
      if (!Number.isNaN(d2)) return d2;
    }
  }
  return null;
}

function buildMissingInfoAudit(b: ChatBillInput): string {
  const checks: Array<{ key: EditableBillFieldKey; label: string; impact: string }> = [
    {
      key: "utilityProvider",
      label: "Electricity Provider",
      impact: "Cannot verify which utility board issued the bill.",
    },
    {
      key: "consumerNumber",
      label: "Consumer Number",
      impact: "Cannot confirm the account/service connection ID.",
    },
    {
      key: "billingPeriod",
      label: "Billing Period",
      impact: "Cannot verify exact billing cycle start and end dates.",
    },
    {
      key: "dueDate",
      label: "Due Date",
      impact: "Cannot answer when payment is due without late fees.",
    },
    {
      key: "disconnectionDate",
      label: "Disconnection Date",
      impact: "Cannot state the disconnection deadline.",
    },
    {
      key: "previousReading",
      label: "Previous Meter Reading",
      impact: "Cannot independently verify units consumed from meter readings.",
    },
    {
      key: "currentReading",
      label: "Current Meter Reading",
      impact: "Cannot independently verify units consumed from meter readings.",
    },
    {
      key: "unitsConsumed",
      label: "Units Consumed",
      impact: "Cannot report total kWh usage, average cost per unit, or consumption % change.",
    },
    {
      key: "fixedCharges",
      label: "Fixed Charges",
      impact: "Cannot separate fixed monthly fees from usage charges.",
    },
    {
      key: "energyCharges",
      label: "Energy Charges",
      impact: "Cannot isolate unit-consumption charges or energy cost per unit.",
    },
    {
      key: "taxesAndAdditionalCharges",
      label: "Taxes & Additional Charges",
      impact: "Cannot isolate taxes, duties, or surcharges.",
    },
    {
      key: "totalAmountPayable",
      label: "Total Amount Payable",
      impact: "Cannot state the final bill total or verify charge sum math.",
    },
  ];

  const missingLines: string[] = [];
  for (const c of checks) {
    if (!getEffectiveField(b, c.key)) {
      missingLines.push(`- Missing ${c.label}: ${c.impact}`);
    }
  }

  return missingLines.length > 0
    ? missingLines.join("\n")
    : "- All 12 core bill fields are available.";
}

function buildSingleBillMathAudit(b: ChatBillInput): string {
  const notes: string[] = [];

  const prev = parseNumericValue(getEffectiveField(b, "previousReading"));
  const curr = parseNumericValue(getEffectiveField(b, "currentReading"));
  const units = parseNumericValue(getEffectiveField(b, "unitsConsumed"));
  if (prev !== null && curr !== null) {
    const readingDiff = round2(curr - prev);
    if (units !== null) {
      const unitDelta = round2(units - readingDiff);
      if (Math.abs(unitDelta) < 0.01) {
        notes.push(
          `Meter Reading Check [Calculated]: Current (${curr}) - Previous (${prev}) = ${readingDiff} units, matching Units Consumed (${units}) exactly.`
        );
      } else {
        notes.push(
          `Meter Reading Check [Calculated]: Current (${curr}) - Previous (${prev}) = ${readingDiff}, whereas Units Consumed is ${units} (difference of ${unitDelta}; could reflect a meter multiplier/CT factor or OCR reading difference if not printed).`
        );
      }
    } else {
      notes.push(
        `Meter Reading Check [Calculated]: Current (${curr}) - Previous (${prev}) = ${readingDiff}, while Units Consumed field is unavailable.`
      );
    }
  } else {
    notes.push("Meter Reading Check: Cannot verify (previous or current reading unavailable).");
  }

  const total = parseNumericValue(getEffectiveField(b, "totalAmountPayable"));
  const fixed = parseNumericValue(getEffectiveField(b, "fixedCharges"));
  const energy = parseNumericValue(getEffectiveField(b, "energyCharges"));
  const taxes = parseNumericValue(getEffectiveField(b, "taxesAndAdditionalCharges"));

  if (units !== null && units > 0) {
    const perUnitParts: string[] = [];
    if (total !== null) {
      perUnitParts.push(`Overall avg cost/unit = ${round2(total / units)} per unit`);
    }
    if (energy !== null) {
      perUnitParts.push(`Energy-only avg rate/unit = ${round2(energy / units)} per unit`);
    }
    if (perUnitParts.length > 0) {
      notes.push(`Unit Cost [Calculated]: ${perUnitParts.join("; ")}.`);
    }
  }

  if (total !== null && total > 0) {
    const shares: string[] = [];
    if (energy !== null) shares.push(`Energy Charges: ${round2((energy / total) * 100)}%`);
    if (fixed !== null) shares.push(`Fixed Charges: ${round2((fixed / total) * 100)}%`);
    if (taxes !== null)
      shares.push(`Taxes & Additional Charges: ${round2((taxes / total) * 100)}%`);
    if (shares.length > 0) {
      notes.push(`Charge Shares of Total (${total}) [Calculated]: ${shares.join(", ")}.`);
    }
  }

  const breakdown = Array.isArray(b.chargesBreakdown) ? b.chargesBreakdown : [];
  const itemNumbers = breakdown
    .map((c) => ({ name: c.name, val: parseNumericValue(c.amount) }))
    .filter((c): c is { name: string; val: number } => c.val !== null);

  const hasExplicitRoundItem = itemNumbers.some((item) => /round/i.test(item.name));

  if (total !== null && itemNumbers.length > 0 && itemNumbers.length === breakdown.length) {
    const breakdownSum = round2(itemNumbers.reduce((s, x) => s + x.val, 0));
    const breakdownExpr = itemNumbers.map((x) => `${x.name} (${x.val})`).join(" + ");
    const diff = round2(total - breakdownSum);
    if (Math.abs(diff) < 0.01) {
      notes.push(
        `Itemized Charges Sum Verification [Calculated]: ${breakdownExpr} = ${breakdownSum}, which matches Total Amount Payable (${total}) exactly${hasExplicitRoundItem ? " (includes explicit rounding line item)" : ""}.`
      );
    } else if (Math.abs(diff) <= 1.5) {
      notes.push(
        `Itemized Charges Sum Verification [Calculated]: ${breakdownExpr} = ${breakdownSum}, compared to Total Amount Payable (${total}) — small rounding difference of ${diff >= 0 ? "+" : ""}${diff}.`
      );
    } else {
      notes.push(
        `Itemized Charges Sum Verification [Calculated]: ${breakdownExpr} = ${breakdownSum}, compared to Total Amount Payable (${total}) — difference of ${diff >= 0 ? "+" : ""}${diff} (may reflect arrears, credits, subsidies, or unextracted line items on the bill).`
      );
    }
  }

  if (total !== null && fixed !== null && energy !== null && taxes !== null) {
    const coreSum = round2(fixed + energy + taxes);
    const diff = round2(total - coreSum);
    if (Math.abs(diff) < 0.01) {
      notes.push(
        `Core Charges Sum Verification [Calculated]: Fixed (${fixed}) + Energy (${energy}) + Taxes & Additional (${taxes}) = ${coreSum}, matching Total Amount Payable (${total}) exactly.`
      );
    } else if (Math.abs(diff) <= 1.5) {
      notes.push(
        `Core Charges Sum Verification [Calculated]: Fixed (${fixed}) + Energy (${energy}) + Taxes & Additional (${taxes}) = ${coreSum} vs Total Amount Payable (${total}) — rounding difference of ${diff >= 0 ? "+" : ""}${diff}.`
      );
    } else {
      notes.push(
        `Core Charges Sum Verification [Calculated]: Fixed (${fixed}) + Energy (${energy}) + Taxes & Additional (${taxes}) = ${coreSum} vs Total Amount Payable (${total}) — difference of ${diff >= 0 ? "+" : ""}${diff} (check itemized breakdown for arrears, rebates, or additional fees).`
      );
    }
  } else if (
    !(total !== null && itemNumbers.length > 0 && itemNumbers.length === breakdown.length)
  ) {
    notes.push(
      "Charge Sum Verification: Cannot perform full sum verification because Total Amount Payable or one or more individual charge components are missing."
    );
  }

  return notes.join("\n");
}

function buildMultiBillComparisonAudit(bills: ChatBillInput[]): string {
  if (bills.length < 2) return "";
  const [b1, b2] = bills;

  const d1 = tryParseBillDateMs(b1);
  const d2 = tryParseBillDateMs(b2);
  let older = b1;
  let newer = b2;
  let orderNote =
    "Date Order: Could not parse unambiguous bill dates; comparing Bill 1 -> Bill 2 directly.";
  if (d1 !== null && d2 !== null && d1 !== d2) {
    if (d1 < d2) {
      older = b1;
      newer = b2;
      orderNote = `Chronological Order [Inferred from printed dates]: ${b1.slotLabel} is earlier -> ${b2.slotLabel} is newer.`;
    } else {
      older = b2;
      newer = b1;
      orderNote = `Chronological Order [Inferred from printed dates]: ${b2.slotLabel} is earlier -> ${b1.slotLabel} is newer.`;
    }
  }

  const lines: string[] = [
    `=== DETERMINISTIC COMPARISON ANALYTICS (${older.slotLabel} -> ${newer.slotLabel}) ===`,
    orderNote,
  ];
  const missing: string[] = [];

  const compareMetric = (
    label: string,
    rawOlder: string | null,
    rawNewer: string | null
  ): { v1: number | null; v2: number | null; delta: number | null } => {
    const v1 = parseNumericValue(rawOlder);
    const v2 = parseNumericValue(rawNewer);
    if (v1 === null || v2 === null) {
      missing.push(
        `${label} (${older.slotLabel}: ${rawOlder ?? "missing"}, ${newer.slotLabel}: ${rawNewer ?? "missing"})`
      );
      return { v1, v2, delta: null };
    }
    const delta = round2(v2 - v1);
    const pct = v1 !== 0 ? round2((delta / v1) * 100) : null;
    const direction = delta > 0 ? "increased" : delta < 0 ? "decreased" : "unchanged";
    const pctStr = pct !== null ? ` (${pct >= 0 ? "+" : ""}${pct}%)` : "";
    lines.push(
      `- ${label} [Calculated]: ${older.slotLabel} = ${v1} -> ${newer.slotLabel} = ${v2} | ${direction} by ${delta >= 0 ? "+" : ""}${delta}${pctStr}`
    );
    return { v1, v2, delta };
  };

  const unitsCmp = compareMetric(
    "Units Consumed",
    getEffectiveField(older, "unitsConsumed"),
    getEffectiveField(newer, "unitsConsumed")
  );
  const totalCmp = compareMetric(
    "Total Bill Amount",
    getEffectiveField(older, "totalAmountPayable"),
    getEffectiveField(newer, "totalAmountPayable")
  );
  const energyCmp = compareMetric(
    "Energy Charges",
    getEffectiveField(older, "energyCharges"),
    getEffectiveField(newer, "energyCharges")
  );
  const fixedCmp = compareMetric(
    "Fixed Charges",
    getEffectiveField(older, "fixedCharges"),
    getEffectiveField(newer, "fixedCharges")
  );
  const taxesCmp = compareMetric(
    "Taxes & Additional Charges",
    getEffectiveField(older, "taxesAndAdditionalCharges"),
    getEffectiveField(newer, "taxesAndAdditionalCharges")
  );

  // Per-unit effective cost comparison
  if (
    unitsCmp.v1 !== null &&
    unitsCmp.v1 > 0 &&
    unitsCmp.v2 !== null &&
    unitsCmp.v2 > 0 &&
    totalCmp.v1 !== null &&
    totalCmp.v2 !== null
  ) {
    const cpu1 = round2(totalCmp.v1 / unitsCmp.v1);
    const cpu2 = round2(totalCmp.v2 / unitsCmp.v2);
    const cpuDelta = round2(cpu2 - cpu1);
    const cpuPct = cpu1 !== 0 ? round2((cpuDelta / cpu1) * 100) : null;
    lines.push(
      `- Overall Avg Cost per Unit [Calculated]: ${older.slotLabel} = ${cpu1}/unit -> ${newer.slotLabel} = ${cpu2}/unit (${cpuDelta >= 0 ? "+" : ""}${cpuDelta}${cpuPct !== null ? `, ${cpuPct >= 0 ? "+" : ""}${cpuPct}%` : ""})`
    );
  }

  // Contribution share of total bill delta
  if (totalCmp.delta !== null && Math.abs(totalCmp.delta) > 0.01) {
    const contributors: Array<{ name: string; delta: number; sharePct: number }> = [];
    for (const entry of [
      { name: "Energy Charges", delta: energyCmp.delta },
      { name: "Fixed Charges", delta: fixedCmp.delta },
      { name: "Taxes & Additional Charges", delta: taxesCmp.delta },
    ]) {
      if (entry.delta !== null && Math.abs(entry.delta) > 0.01) {
        contributors.push({
          name: entry.name,
          delta: entry.delta,
          sharePct: round2((entry.delta / totalCmp.delta) * 100),
        });
      }
    }
    if (contributors.length > 0) {
      lines.push(
        "Contribution to Total Bill Change [Calculated]: " +
          contributors
            .map(
              (c) =>
                `${c.name}: ${c.delta >= 0 ? "+" : ""}${c.delta} (${c.sharePct}% of total change)`
            )
            .join("; ")
      );
    }
  }

  // Match itemized breakdown charges where names align
  const breakdownOlder = Array.isArray(older.chargesBreakdown)
    ? older.chargesBreakdown
    : [];
  const breakdownNewer = Array.isArray(newer.chargesBreakdown)
    ? newer.chargesBreakdown
    : [];

  const mapOlder = new Map<string, { name: string; val: number }>();
  for (const item of breakdownOlder) {
    const val = parseNumericValue(item.amount);
    if (val !== null) {
      mapOlder.set(item.name.toLowerCase().trim(), { name: item.name, val });
    }
  }
  const itemDeltas: Array<{
    name: string;
    v1: number;
    v2: number;
    delta: number;
    pct: number | null;
  }> = [];
  for (const item2 of breakdownNewer) {
    const key = item2.name.toLowerCase().trim();
    const match1 = mapOlder.get(key);
    const v2 = parseNumericValue(item2.amount);
    if (match1 && v2 !== null) {
      const delta = round2(v2 - match1.val);
      const pct = match1.val !== 0 ? round2((delta / match1.val) * 100) : null;
      itemDeltas.push({ name: item2.name, v1: match1.val, v2, delta, pct });
    }
  }
  if (itemDeltas.length > 0) {
    itemDeltas.sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));
    lines.push(
      "Matched Itemized Charge Changes [Calculated] (sorted by magnitude): " +
        itemDeltas
          .map(
            (d) =>
              `${d.name}: ${d.v1} -> ${d.v2} (${d.delta >= 0 ? "+" : ""}${d.delta}${d.pct !== null ? `, ${d.pct >= 0 ? "+" : ""}${d.pct}%` : ""})`
          )
          .join("; ")
    );
  }

  if (missing.length > 0) {
    lines.push(`Missing Comparison Data (cannot compute difference): ${missing.join("; ")}`);
  }

  return lines.join("\n");
}

async function startServer() {
  const app = express();
  app.use(express.json({ limit: "30mb" }));

  app.get("/api/health", (_req: Request, res: Response) => {
    res.json({ status: "ok" });
  });

  app.post("/api/bills/analyze", async (req: Request, res: Response) => {
    try {
      const body = req.body as AnalyzeBillBody;
      const {
        fileName,
        mimeType,
        originalMimeType,
        base64Data,
        slotId,
        slotLabel,
        languagePreference,
        preprocessingApplied,
      } = body;

      if (!fileName || !mimeType || !base64Data) {
        res.status(400).json({ error: "Missing fileName, mimeType, or file data." });
        return;
      }

      const normalizedMime = mimeType === "image/jpg" ? "image/jpeg" : mimeType.toLowerCase();
      if (!ALLOWED_MIME_TYPES.has(normalizedMime)) {
        res.status(400).json({
          error: "Unsupported file format. Please upload a PDF, PNG, JPG, or WEBP electricity bill.",
        });
        return;
      }

      const ai = getAiClient();
      const langNote = languageInstructionText(languagePreference || "auto");

      let extractionMethod = "Image OCR";
      let directPdfText: string | null = null;

      if (normalizedMime === "application/pdf") {
        const pdfResult = await extractEmbeddedPdfText(Buffer.from(base64Data, "base64"));
        if (pdfResult.isTextBased) {
          directPdfText = pdfResult.text;
          extractionMethod = "Embedded PDF Text";
        } else {
          extractionMethod = "Scanned PDF OCR";
        }
      } else {
        extractionMethod =
          Array.isArray(preprocessingApplied) && preprocessingApplied.length > 0
            ? "Preprocessed Image OCR"
            : "Image OCR";
      }

      const extractionInstructions = `You are BillWise AI's OCR and NLP Electricity Bill Extraction Engine.
You support diverse electricity bill layouts across utility providers (tabular multi-column PDFs, regional Indian utility slips like TSSPDCL/APSPDCL/APEPDCL/BESCOM/MSEDCL/TNEB/UPPCL/BSES/Tata Power, thermal POS rolls, bilingual English/Telugu/Hindi bills, and international utility statements).

Your task is to:
1. ${
        directPdfText
          ? "Read the embedded PDF text supplied below and preserve it faithfully in 'rawOcrText'."
          : "Perform faithful, complete Optical Character Recognition (OCR) on the uploaded document (handling rotated/sideways scans, tables, thermal prints, and multi-column layouts) and transcribe all visible text, numbers, headers, tables, Telugu/Hindi/English labels, and dates into 'rawOcrText'."
      }
2. Determine if the document is actually an electricity/power utility bill ('isElectricityBill').
3. Extract all structured electricity bill entities strictly from what is visibly present in the document:
   - Electricity provider name ('utilityProvider')
   - Consumer name ('consumerName')
   - Consumer number / USC No / Service No / CA No ('consumerNumber')
   - Service / tariff category ('serviceCategory')
   - Bill number ('billNumber')
   - Bill date ('billDate')
   - Billing period ('billingPeriod')
   - Due date ('dueDate')
   - Disconnection date ('disconnectionDate')
   - Previous meter reading ('previousReading')
   - Current meter reading ('currentReading')
   - Units consumed ('unitsConsumed')
   - Fixed charges ('fixedCharges' — fixed/demand/minimum monthly charge)
   - Energy charges ('energyCharges' — variable consumption/unit slab charges)
   - Taxes and additional charges ('taxesAndAdditionalCharges' — electricity duty, GST/tax, customer charges, FAC/FPPCA, surcharge, cess, or combined total of taxes & extra fees if listed separately; include label details cleanly)
   - Total amount payable ('totalAmountPayable')
   - CRITICAL RULE: NEVER invent, estimate, guess, or fabricate missing values. If a field is not visible, illegible, or absent on the bill, set it to null and list its human-readable name in 'unavailableFields'.
   - For 'chargesBreakdown', list every visible charge, tax, fee, credit, or subsidy line item exactly as printed with its amount and a simple, beginner-friendly 1-sentence explanation.
   - Assess 'ocrConfidence' honestly ("High", "Medium", or "Low") based on document readability, sharpness, and completeness (use "High" for clear digital text, "Medium" if slightly blurry/skewed/faded, "Low" if heavily degraded or partially cut off), and note any factual visibility issues in 'documentConfidenceNote'.
4. Write 'initialSummary': a detailed, easy-to-understand structured bill breakdown in Markdown (NEVER write a paragraph!). Use ONLY headings, bullet points, and a Markdown table formatted like this:
   ### Quick Bill Overview
   - **Provider:** <value or "Not found in uploaded bill">
   - **Consumer Name & Number:** <values or "Not found in uploaded bill">
   - **Billing Period:** <value or "Not found in uploaded bill">
   - **Total Amount Payable:** <value or "Not found in uploaded bill">
   - **Due Date:** <value or "Not found in uploaded bill">
   - **Disconnection Date:** <value or "Not found in uploaded bill">

   ### Meter & Units Used
   - **Previous Reading:** <value or "Not found in uploaded bill">
   - **Current Reading:** <value or "Not found in uploaded bill">
   - **Total Units Consumed:** <value or "Not found in uploaded bill">

   ### Charges Breakdown (Simple Meaning)
   | Charge | Amount | Simple Explanation |
   | :--- | :--- | :--- |
   (include every extracted charge, tax, fee, or fixed/energy charge with its plain-language meaning)

   ### Missing Information
   - List any important details not found in the uploaded bill (or state "- All key bill details were found.").
5. Generate 'suggestedQuestions': 3–4 short, helpful questions a user can ask next based ONLY on the fields actually extracted from this bill (e.g., if dueDate exists, suggest asking about due date; if energyCharges/chargesBreakdown exist, suggest explaining charges; if unitsConsumed exists, suggest asking about units & cost per unit; if fields are missing, suggest asking what is missing). Write them in the target summary language.
   ${langNote}`;

      const contents = directPdfText
        ? `Extract structured electricity bill fields from the embedded text of "${fileName}" (${slotLabel || "Bill"}):\n\n${directPdfText.slice(0, 30000)}`
        : {
            parts: [
              {
                inlineData: {
                  mimeType: normalizedMime,
                  data: base64Data,
                },
              },
              {
                text: `Extract the full OCR text and structured electricity bill fields from "${fileName}" (${slotLabel || "Bill"}).`,
              },
            ],
          };

      const response = await generateWithModelFallback(ai, {
        contents,
        systemInstruction: extractionInstructions,
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            isElectricityBill: {
              type: Type.BOOLEAN,
              description: "True if the document is an electricity/power utility bill, false otherwise.",
            },
            utilityProvider: {
              type: Type.STRING,
              nullable: true,
              description: "Electricity utility board or company name.",
            },
            consumerName: {
              type: Type.STRING,
              nullable: true,
            },
            consumerNumber: {
              type: Type.STRING,
              nullable: true,
              description: "Consumer No, USC No, Service No, Account ID, or CA Number.",
            },
            serviceCategory: {
              type: Type.STRING,
              nullable: true,
              description: "Tariff category / domestic / commercial / LT-1 etc.",
            },
            billNumber: {
              type: Type.STRING,
              nullable: true,
            },
            billDate: {
              type: Type.STRING,
              nullable: true,
            },
            billingPeriod: {
              type: Type.STRING,
              nullable: true,
            },
            dueDate: {
              type: Type.STRING,
              nullable: true,
            },
            disconnectionDate: {
              type: Type.STRING,
              nullable: true,
            },
            previousReading: {
              type: Type.STRING,
              nullable: true,
            },
            currentReading: {
              type: Type.STRING,
              nullable: true,
            },
            unitsConsumed: {
              type: Type.STRING,
              nullable: true,
              description: "Total kWh / units billed with unit label.",
            },
            fixedCharges: {
              type: Type.STRING,
              nullable: true,
              description: "Fixed charges / demand charges / monthly base charges as printed.",
            },
            energyCharges: {
              type: Type.STRING,
              nullable: true,
              description: "Energy / electricity consumption charges as printed.",
            },
            taxesAndAdditionalCharges: {
              type: Type.STRING,
              nullable: true,
              description: "Electricity duty, taxes, customer charges, FAC/FPPCA, surcharges, or additional fees as printed.",
            },
            totalAmountPayable: {
              type: Type.STRING,
              nullable: true,
              description: "Total amount due with currency symbol as printed.",
            },
            chargesBreakdown: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  name: { type: Type.STRING },
                  amount: { type: Type.STRING },
                  simpleExplanation: { type: Type.STRING },
                },
                required: ["name", "amount", "simpleExplanation"],
              },
            },
            unavailableFields: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description: "List of important standard bill fields not present or unreadable in this document.",
            },
            rawOcrText: {
              type: Type.STRING,
              description: "Complete verbatim OCR or embedded PDF text from the uploaded bill.",
            },
            ocrConfidence: {
              type: Type.STRING,
              enum: ["High", "Medium", "Low"],
              nullable: true,
              description: "Qualitative OCR readability confidence based on text clarity.",
            },
            documentConfidenceNote: {
              type: Type.STRING,
              nullable: true,
              description: "Any factual note about image blur, cropping, or partial pages, or null if clear.",
            },
            initialSummary: {
              type: Type.STRING,
              description:
                "Detailed, easy-to-understand structured Markdown breakdown using headings, bullet points, and a charges table (never dense paragraphs).",
            },
            suggestedQuestions: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description:
                "3-4 short, relevant follow-up questions grounded strictly in fields actually extracted from this bill.",
            },
          },
          required: [
            "isElectricityBill",
            "chargesBreakdown",
            "unavailableFields",
            "rawOcrText",
            "initialSummary",
            "suggestedQuestions",
          ],
        },
      });

      const parsed = JSON.parse((response.text ?? "{}").trim());

      res.json({
        bill: {
          slotId: slotId === "bill2" ? "bill2" : "bill1",
          slotLabel: slotLabel || (slotId === "bill2" ? "Bill 2" : "Bill 1"),
          fileName,
          mimeType: (originalMimeType || normalizedMime).toLowerCase(),
          extractionMethod,
          preprocessingApplied: Array.isArray(preprocessingApplied) ? preprocessingApplied : [],
          isElectricityBill: Boolean(parsed.isElectricityBill),
          utilityProvider: parsed.utilityProvider ?? null,
          consumerName: parsed.consumerName ?? null,
          consumerNumber: parsed.consumerNumber ?? null,
          serviceCategory: parsed.serviceCategory ?? null,
          billNumber: parsed.billNumber ?? null,
          billDate: parsed.billDate ?? null,
          billingPeriod: parsed.billingPeriod ?? null,
          dueDate: parsed.dueDate ?? null,
          disconnectionDate: parsed.disconnectionDate ?? null,
          previousReading: parsed.previousReading ?? null,
          currentReading: parsed.currentReading ?? null,
          unitsConsumed: parsed.unitsConsumed ?? null,
          fixedCharges: parsed.fixedCharges ?? null,
          energyCharges: parsed.energyCharges ?? null,
          taxesAndAdditionalCharges: parsed.taxesAndAdditionalCharges ?? null,
          totalAmountPayable: parsed.totalAmountPayable ?? null,
          chargesBreakdown: Array.isArray(parsed.chargesBreakdown) ? parsed.chargesBreakdown : [],
          unavailableFields: Array.isArray(parsed.unavailableFields) ? parsed.unavailableFields : [],
          rawOcrText: directPdfText || parsed.rawOcrText || "",
          ocrConfidence:
            extractionMethod === "Embedded PDF Text"
              ? "High"
              : parsed.ocrConfidence ?? null,
          documentConfidenceNote: parsed.documentConfidenceNote ?? null,
          initialSummary: parsed.initialSummary || "Bill analyzed.",
          suggestedQuestions: Array.isArray(parsed.suggestedQuestions)
            ? parsed.suggestedQuestions.slice(0, 4)
            : [],
        },
      });
    } catch (err) {
      console.error("Analyze bill error:", err);
      const { status, message } = formatApiError(err);
      res.status(status).json({ error: message });
    }
  });

  app.post("/api/chat", async (req: Request, res: Response) => {
    try {
      const body = req.body as ChatBody;
      const message = (body.message || "").trim();
      const languagePreference = body.languagePreference || "auto";
      const bills = Array.isArray(body.bills) ? body.bills : [];
      const history = Array.isArray(body.history) ? body.history : [];

      if (!message) {
        res.status(400).json({ error: "Message cannot be empty." });
        return;
      }

      const ai = getAiClient();

      const billsContext =
        bills.length === 0
          ? "NO ELECTRICITY BILLS ARE CURRENTLY UPLOADED."
          : [
              ...bills.map(
                (b) => `=== ${b.slotLabel} (${b.fileName}) ===
Extraction Method: ${b.extractionMethod ?? "OCR"}
OCR Confidence: ${b.ocrConfidence ?? "Unknown"}${b.documentConfidenceNote ? ` (${b.documentConfidenceNote})` : ""}
User-Verified Fields: ${
                  b.verifiedValues && Object.keys(b.verifiedValues).length > 0
                    ? Object.keys(b.verifiedValues).join(", ")
                    : Array.isArray(b.userCorrectedFields) && b.userCorrectedFields.length > 0
                      ? b.userCorrectedFields.join(", ")
                      : "None"
                }
Is Electricity Bill: ${b.isElectricityBill}
Utility Provider: ${formatFieldWithProvenance(b, "utilityProvider")}
Consumer Name: ${formatFieldWithProvenance(b, "consumerName")}
Consumer Number: ${formatFieldWithProvenance(b, "consumerNumber")}
Service Category: ${formatFieldWithProvenance(b, "serviceCategory")}
Bill Number: ${formatFieldWithProvenance(b, "billNumber")}
Bill Date: ${formatFieldWithProvenance(b, "billDate")}
Billing Period: ${formatFieldWithProvenance(b, "billingPeriod")}
Due Date: ${formatFieldWithProvenance(b, "dueDate")}
Disconnection Date: ${formatFieldWithProvenance(b, "disconnectionDate")}
Previous Meter Reading: ${formatFieldWithProvenance(b, "previousReading")}
Current Meter Reading: ${formatFieldWithProvenance(b, "currentReading")}
Units Consumed: ${formatFieldWithProvenance(b, "unitsConsumed")}
Fixed Charges: ${formatFieldWithProvenance(b, "fixedCharges")}
Energy Charges: ${formatFieldWithProvenance(b, "energyCharges")}
Taxes & Additional Charges: ${formatFieldWithProvenance(b, "taxesAndAdditionalCharges")}
Total Amount Payable: ${formatFieldWithProvenance(b, "totalAmountPayable")}
Charges Breakdown: ${JSON.stringify(b.chargesBreakdown ?? [])}
Unavailable Fields: ${JSON.stringify(b.unavailableFields ?? [])}
Missing Information Impact:
${buildMissingInfoAudit(b)}
Deterministic Numerical Audit (computed using User-Verified values where present, else Original OCR):
${buildSingleBillMathAudit(b)}
Raw OCR / Extracted Text:
${(b.rawOcrText ?? "").slice(0, 12000)}`
              ),
              buildMultiBillComparisonAudit(bills),
            ]
              .filter(Boolean)
              .join("\n\n");

      const recentHistory = history.slice(-18);
      const historyText =
        recentHistory.length === 0
          ? "No prior conversation turns."
          : recentHistory
              .map((m) => `${m.role === "user" ? "User" : "BillWise AI"}: ${m.content}`)
              .join("\n");

      let langDirective = "";
      if (languagePreference === "en") {
        langDirective =
          "The user selected English mode. Reply strictly in clear, simple English (including headings, explanations, evidence notes, clarification options, and suggestedQuestions).";
      } else if (languagePreference === "te") {
        langDirective =
          "The user selected Telugu mode. Reply strictly in natural, easy-to-read Telugu script (తెలుగు) — including headings, explanations, clarification options, and suggestedQuestions — while keeping bill numbers, dates, and IDs accurate.";
      } else if (languagePreference === "hi") {
        langDirective =
          "The user selected Hindi mode. Reply strictly in natural, easy-to-read Hindi using Devanagari script (हिन्दी) — including headings, explanations, clarification options, and suggestedQuestions — while keeping bill numbers, units, dates, and consumer IDs accurate.";
      } else if (languagePreference === "tenglish") {
        langDirective =
          "The user selected Tenglish mode. Reply strictly in natural Tenglish (conversational Telugu written in English/Latin alphabet, e.g., 'Mee bill lo energy charges ₹940 unnayi, fixed charges ₹60 unnayi.') — including headings, explanations, clarification options, and suggestedQuestions.";
      } else {
        langDirective =
          "Auto-detect the language of the user's latest message among English, Telugu, Hindi, and Tenglish: if written in Telugu script (తెలుగు), reply (including headings and suggestedQuestions) in Telugu; if written in Hindi (Devanagari script हिन्दी, or clear Hindi phrasing in Roman letters like 'mera bijli bill kitna hai' / 'due date kab hai'), reply in Hindi using Devanagari script (हिन्दी); if written in Tenglish (Telugu words using English letters like 'bill entha', 'due date eppudu', 'enduku ekkuva vachindi', 'cheppu', 'units enni'), reply (including headings and suggestedQuestions) in natural Tenglish; otherwise reply in English. Always honor explicit language-switch requests in the user's message (e.g., 'Explain in Hindi', 'Hindi mein batao', 'Telugu lo cheppu') while keeping prior conversation context intact.";
      }

      const systemInstruction = `You are BillWise AI, an advanced Evidence-Based Electricity Bill Understanding Chatbot.

CORE EVIDENCE-BASED ANSWERING & NLP CAPABILITIES:
1. Strict Evidence Grounding:
   - Answer EVERY electricity-bill question strictly using the extracted/user-verified data and raw OCR text of the uploaded bill(s).
   - Connect every claim in your reply to its exact source bill ('Bill 1' / 'Bill 2'), supporting field name(s), and value(s).
   - Populating 'evidence' is MANDATORY for every bill-related answer: list each supporting item with 'billLabel', 'fieldName', 'value', and 'valueSource' ("Extracted", "User-Verified", or "Calculated").
   - In 'reply', always include a clear **Source & Evidence** section/table distinguishing **Extracted from Bill**, **User-Verified**, and **Calculated** values.
2. Question-to-Field Grounding Rules:
   - Total Amount & Charge Questions: MUST list each individual extracted charge (Fixed Charges, Energy Charges, Taxes & Additional Charges, and itemized 'chargesBreakdown' lines), show their arithmetic sum step-by-step, and verify that sum against 'Total Amount Payable' (noting exact match, rounding difference, or unexplained gap/arrears from the Deterministic Numerical Audit).
   - Due-Date Questions: MUST cite the extracted 'Due Date', plus 'Bill Date', 'Billing Period', and 'Disconnection Date' when present (explicitly noting any missing date as not found in the uploaded bill).
   - Consumption & Meter-Reading Questions: MUST cite 'Previous Meter Reading', 'Current Meter Reading', and 'Units Consumed', and show the calculated reading difference ('Current - Previous') check where both readings exist.
3. Smart Question Understanding & Follow-Up Resolution:
   - Recognize semantic intent across diverse phrasings in English, Telugu, Hindi, and Tenglish (e.g., "service no / USC no / CA no" = Consumer Number; "last date / fine-free date" = Due Date; "kWh / power used" = Units Consumed; "does my bill add up?" = Charge Sum Verification).
   - Resolve short or elliptical follow-ups ("Why?", "Explain this amount.", "Does that match the total?", "What about last month?", "How much did it increase?", "Which charge increased?") using prior conversation turns and uploaded bill data.
4. Ambiguity Detection (Clarify Before Answering When Truly Ambiguous):
   - If a user's question genuinely has multiple distinct meanings in context, set 'needsClarification' to true, set 'detectedIntent' to "Clarification Needed", provide 2–4 concise 'clarificationOptions' (in the reply language), and ask a polite, structured clarification question BEFORE guessing.
   - Trigger clarification when 2 bills are loaded and user asks "What is the due date?", "How much is the total?", "Explain the charges", or "Correct the amount to ₹1,200" WITHOUT specifying Bill 1 vs Bill 2 (unless recent turns already uniquely focused on one bill), or when asking "What is the date?" / "How much is the charge?" ambiguously.
   - Do NOT trigger clarification unnecessarily when intent is clear.
5. Smart Bill Reasoning (Why did the bill increase/change?):
   - Compare earlier/previous and newer/current bills using Deterministic Comparison Analytics: units consumed (% change), average cost/unit, and individual charges (with share of total change).
   - NEVER assume or invent unprinted causes (such as AC usage, weather, appliance faults, or tariff slab changes not printed on the bill). If only 1 bill is uploaded, state clearly that a second bill is needed to compare changes.
6. Consumption Pattern Intelligence:
   - Compare units consumed across bills, identify increases/decreases, report exact unit & percentage changes and average cost/unit changes, and explicitly highlight missing comparison data.
7. Smart OCR Correction (Separate Verified Values):
   - Always treat USER-VERIFIED values as authoritative in all answers and calculations, while preserving and acknowledging original OCR values.
   - If the user corrects an OCR value in chat unambiguously, set 'detectedIntent' to "Smart OCR Correction", populate 'ocrCorrections', and confirm both original OCR and new User-Verified values.
8. Missing & Unclear Information Intelligence:
   - If any required value is missing or unclear (e.g., "Not found in uploaded bill", or 'ocrConfidence' is Medium/Low with 'documentConfidenceNote'), state clearly what is missing/uncertain and which part of the question cannot be answered from the bill.
   - NEVER guess, extrapolate, or fabricate missing values.
9. Answer Confidence & Reliability (NO FAKE PERCENTAGES):
   - Set 'answerReliability' from ["High", "Medium", "Low", "Uncertain", "Cannot Determine"] with a short factual 'reliabilityReason' based ONLY on extraction method, 'ocrConfidence', user verification, and field completeness. Never invent numeric confidence percentages.
10. Smart Suggested Next Questions:
   - Populate 'suggestedQuestions' with 2–3 concise follow-up questions in the SAME language as 'reply', grounded strictly in fields present in the uploaded bill(s) (or [] if no bill is uploaded).

RESPONSE FORMAT RULES (STRICT — NO PARAGRAPHS):
- Keep explanations simple, beginner-friendly, and free of dense paragraphs.
- Use short section headings (### Heading), concise bullet points (- **Label:** Value — simple meaning), numbered calculation steps, and Markdown tables (| Field / Charge | Value | Basis / Meaning |).

STRICT GUARDRAILS (NON-NEGOTIABLE):
- NEVER invent missing bill information or assume reasons for increased charges not established by the uploaded bill(s).
- If NO bill has been uploaded yet and the user asks bill-specific questions, politely ask them to upload a PDF or image of their electricity bill first (in the reply language) and set 'answerReliability' to "Cannot Determine".
- Scope restriction: Keep BillWise AI focused strictly on electricity bill understanding (no payment features, admin dashboards, or unrelated tools).

LANGUAGE DIRECTIVE:
${langDirective}`;

      const prompt = `UPLOADED BILL CONTEXT, PROVENANCE & DETERMINISTIC ANALYTICS:
${billsContext}

CONVERSATION HISTORY:
${historyText}

LATEST USER QUESTION:
${message}`;

      const response = await generateWithModelFallback(ai, {
        contents: prompt,
        systemInstruction,
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            detectedIntent: {
              type: Type.STRING,
              description:
                "Concise 2-5 word label describing the user's NLP intent (e.g., 'Follow-up: Increase Reason', 'Charge Relationship & Math', 'Consumption Trend Analysis', 'Smart OCR Correction', 'Clarification Needed', 'Bill Summary', 'Due Date Check').",
            },
            detectedLanguage: {
              type: Type.STRING,
              enum: ["English", "Telugu", "Hindi", "Tenglish"],
              description: "Detected language of the user's question.",
            },
            needsClarification: {
              type: Type.BOOLEAN,
              description:
                "True ONLY when the user's question is genuinely ambiguous and requires a clarification question before answering.",
            },
            clarificationOptions: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description:
                "2-4 concise clickable clarification choices in the reply language when needsClarification is true; empty array otherwise.",
            },
            answerReliability: {
              type: Type.STRING,
              enum: ["High", "Medium", "Low", "Uncertain", "Cannot Determine"],
              description:
                "Qualitative reliability grounded in OCR/PDF extraction confidence, user verification, and data completeness. Never use numeric percentages.",
            },
            reliabilityReason: {
              type: Type.STRING,
              description:
                "Short factual explanation of answerReliability (e.g., 'Based on High-confidence Embedded PDF Text' or 'Previous reading missing from uploaded bill').",
            },
            evidence: {
              type: Type.ARRAY,
              description:
                "Supporting bill fields and calculated metrics used in the answer, distinguishing Extracted, User-Verified, and Calculated sources.",
              items: {
                type: Type.OBJECT,
                properties: {
                  billLabel: { type: Type.STRING },
                  fieldName: { type: Type.STRING },
                  value: { type: Type.STRING },
                  valueSource: {
                    type: Type.STRING,
                    enum: ["Extracted", "User-Verified", "Calculated"],
                  },
                },
                required: ["billLabel", "fieldName", "value", "valueSource"],
              },
            },
            ocrCorrections: {
              type: Type.ARRAY,
              description:
                "Populated ONLY when the user explicitly corrects an extracted OCR field value unambiguously; otherwise empty array.",
              items: {
                type: Type.OBJECT,
                properties: {
                  slotId: {
                    type: Type.STRING,
                    enum: ["bill1", "bill2"],
                  },
                  field: {
                    type: Type.STRING,
                    enum: [
                      "utilityProvider",
                      "consumerName",
                      "consumerNumber",
                      "serviceCategory",
                      "billNumber",
                      "billDate",
                      "billingPeriod",
                      "dueDate",
                      "disconnectionDate",
                      "previousReading",
                      "currentReading",
                      "unitsConsumed",
                      "fixedCharges",
                      "energyCharges",
                      "taxesAndAdditionalCharges",
                      "totalAmountPayable",
                    ],
                  },
                  correctedValue: {
                    type: Type.STRING,
                    nullable: true,
                    description: "The corrected value, or null if user says the field is not on the bill.",
                  },
                },
                required: ["slotId", "field", "correctedValue"],
              },
            },
            suggestedQuestions: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description:
                "2-3 short, relevant follow-up questions in the reply language grounded in the uploaded bill's extracted fields.",
            },
            reply: {
              type: Type.STRING,
              description:
                "Detailed, structured answer in Markdown format (headings, bullet lists, tables; never dense paragraphs), including a clear Source & Evidence section when citing bill data.",
            },
          },
          required: [
            "detectedIntent",
            "detectedLanguage",
            "needsClarification",
            "clarificationOptions",
            "answerReliability",
            "reliabilityReason",
            "evidence",
            "ocrCorrections",
            "suggestedQuestions",
            "reply",
          ],
        },
      });

      const parsed = JSON.parse((response.text ?? "{}").trim());

      res.json({
        detectedIntent: parsed.detectedIntent || "General Bill Q&A",
        detectedLanguage: parsed.detectedLanguage || "English",
        needsClarification: Boolean(parsed.needsClarification),
        clarificationOptions: Array.isArray(parsed.clarificationOptions)
          ? parsed.clarificationOptions.slice(0, 4)
          : [],
        answerReliability: parsed.answerReliability || "High",
        reliabilityReason: parsed.reliabilityReason || "",
        evidence: Array.isArray(parsed.evidence) ? parsed.evidence : [],
        ocrCorrections: Array.isArray(parsed.ocrCorrections) ? parsed.ocrCorrections : [],
        suggestedQuestions: Array.isArray(parsed.suggestedQuestions)
          ? parsed.suggestedQuestions.slice(0, 4)
          : [],
        reply:
          parsed.reply ||
          "I could not generate an answer from the uploaded bill. Please try rephrasing your question.",
      });
    } catch (err) {
      console.error("Chat error:", err);
      const { status, message } = formatApiError(err);
      res.status(status).json({ error: message });
    }
  });

  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(__dirname, "dist");
    app.use(express.static(distDir));
    app.get("*", (_req: Request, res: Response) => {
      res.sendFile(path.join(distDir, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`BillWise AI server running on http://localhost:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error("Failed to start server:", err);
  process.exit(1);
});
