---
name: data-extraction-verification
description: Mandatory verification protocol for numeric data extraction from Treasury Bulletins and tabular sources. USE THIS SKILL during Phase 5 (Execute) of brainstorming whenever extracting numeric values. Prevents row/column misalignment, wrong metric selection, and wrong data series errors.
---

# Data Extraction Verification Protocol

Mandatory verification workflow for numeric extraction from Treasury Bulletins or tabular sources.

## When to Apply

Apply DURING data extraction (Phase 5 of brainstorming). Every numeric value must pass these checks before use.

## Protocol Steps

### 1. Source Context Declaration

BEFORE using any number, state:
- Exact file path
- Table name/identifier
- Row label AND column header
- Units (millions, billions, percentage, etc.)

```
Source: treasury_bulletins_parsed/2023/table_cm_ii_2.csv
Table: CM-II-2 (Foreign Holdings)
Row: "United Kingdom" | Column: "December 2023"
Units: Millions of dollars
```

### 2. Value Confirmation

- Re-read source passage containing the value
- Verify adjacent cells (above/below, left/right) to confirm alignment
- For large tables: read 2-3 surrounding values to confirm position

### 3. Metric Disambiguation

BEFORE extraction, explicitly identify:
- What metric does the question ask for?
- What metric does the column/row label describe?
- Are these the same?

Watch for:
- "amount outstanding" vs "sales" vs "redemptions"
- "average" vs raw values requiring computation
- "end of period" vs "during period"
- "total" vs specific subcategory

**Stop condition:** If your column header does not contain the exact words from the question, verify it's the correct metric before proceeding.

### 4. Magnitude Sanity Check

- Estimate expected order of magnitude from context
- Compare extracted value against expectation
- If off by >10x: STOP and re-verify source selection
- For comparisons: verify both values use same metric type

### 5. Multi-Source Triangulation (high-stakes extractions)

When same data appears in multiple locations:
- Check at least one alternative source
- Flag discrepancies before proceeding

## Quick Checklist

Before using an extracted value:

- [ ] Stated exact file, table, row, column, units
- [ ] Re-read source to confirm value
- [ ] Verified adjacent cells for alignment
- [ ] Confirmed metric matches question's ask
- [ ] Checked magnitude is reasonable

## Error Patterns Prevented

| Error Type | Example | Prevention |
|------------|---------|------------|
| Row/column misalignment | 103,235 vs 103,375 | Step 2: Verify adjacent cells |
| Wrong metric | "amount outstanding" instead of asked metric | Step 3: Metric disambiguation |
| Wrong data series | Wrong year or category | Step 1: Source declaration |
| Magnitude error | Value 260x too large | Step 4: Sanity check |
