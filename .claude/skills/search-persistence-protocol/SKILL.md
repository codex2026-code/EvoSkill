---
name: search-persistence-protocol
description: Enforces exhaustive search strategies before concluding factual answers. Use this skill when answering questions requiring web research, questions with potentially ambiguous terms, or enumeration/counting questions. Prevents premature search termination by requiring term interpretation expansion, multi-source verification, and completeness checks.
---

# Search Persistence Protocol

Enforces thorough search coverage before concluding. The agent's most common failure mode is stopping searches too early.

## Rule 1: Term Interpretation Expansion

Before searching ambiguous terms, list ALL reasonable interpretations:

**Example expansions:**
- "Agency" → government agencies, space agencies, news agencies, regulatory agencies, intelligence agencies
- "Most widely used" → by active users, by market share, by downloads, by revenue
- "Popular" → most followers, most engagement, most referenced
- "Best" → highest rated, most sales, expert recommended

**Protocol:** Execute searches for EACH interpretation before narrowing. Do not assume the first interpretation is correct.

## Rule 2: Three-Source Minimum

Before concluding ANY factual answer:
1. Check at least THREE independent sources
2. If only 1-2 sources found, search with different query formulations
3. For enumeration questions, cross-reference list against 2+ additional sources

**Trigger:** Before stating any factual claim, verify: "Have I checked 3+ sources?"

## Rule 3: "Unable to Find" Protocol

Before reporting inability to find data:

1. **Try 3+ query formulations** - Rephrase, use synonyms, try different term orders
2. **Try related searches** - Information may exist in adjacent contexts
3. **Attempt derivation** - Can the answer be calculated from related data?
   - If 96% have X, then ~4% don't have X
   - If list shows 11 items but "top 12" is mentioned, one is missing

**Rule 3a: Data Source Follow-Through** - If you identify a specific data source that should contain the answer (database URL, API endpoint, indicator code, dataset ID), you MUST attempt to fetch from it before concluding "unable to find."

- ❌ "The ITU DataHub has indicator 100095 for 2G coverage. To get the exact figure, query the DataHub directly." → WRONG: identified source but didn't fetch
- ✓ "The ITU DataHub has indicator 100095. Let me query it directly..." [proceeds to fetch] → CORRECT

Anti-pattern: Telling the user "you would need to query X" when you could query X yourself.

Only report "unable to find" after exhausting these steps.

## Rule 4: Enumeration Completeness Check

When counting items in a category:

1. Search "[category] complete list" and "[category] all time"
2. Cross-reference count with 2+ ranking/list sources
3. Before finalizing: "Could I have missed any?"
4. If sources disagree on count, investigate the discrepancy

## Execution

For each factual question:

```
1. EXPAND: List all interpretations of ambiguous terms
2. SEARCH: Query each interpretation separately
3. VERIFY: Confirm 3+ independent sources checked
4. COMPLETE: For enumerations, cross-check for missing items
5. DERIVE: If direct answer unavailable, attempt calculation from related data
6. CONCLUDE: Only after steps 1-5 are satisfied
```
