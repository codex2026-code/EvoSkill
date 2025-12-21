---
name: treasury-fiscal-data
description: >
  Query the U.S. Treasury Fiscal Data API (api.fiscaldata.treasury.gov) to retrieve
  official government financial data including receipts, outlays, and deficit/surplus
  figures. Use this skill when you need to get accurate federal budget data for fiscal
  years, retrieve Monthly Treasury Statement (MTS) data for analysis, access national
  debt figures, or perform economic analysis requiring official Treasury data. This
  skill knows the correct API endpoints, field names, and filter syntax that the raw
  API requires.
---

# Treasury Fiscal Data API

Query official U.S. Treasury financial data with correct endpoint paths and field names.

## Quick Start

For fiscal year summary data (total receipts, outlays, deficit/surplus):

```bash
python scripts/treasury_api.py --dataset mts_summary --fiscal-year 2023
```

Output:
```json
{
  "fiscal_years": {
    "2023": {
      "total_receipts": 4439283739920.54,
      "total_outlays": 6134432040451.31,
      "deficit_surplus": -1695148300530.77,
      "total_receipts_millions": 4439283.74,
      "total_outlays_millions": 6134432.04,
      "deficit_surplus_millions": -1695148.30
    }
  }
}
```

## Common Use Cases

### Get Multiple Fiscal Years (for time series analysis)
```bash
python scripts/treasury_api.py --dataset mts_summary \
  --fiscal-year 2019 --fiscal-year 2020 --fiscal-year 2021 \
  --fiscal-year 2022 --fiscal-year 2023
```

### Query Raw API Endpoint
```bash
python scripts/treasury_api.py --endpoint mts_table_9 --fiscal-year 2023 --month 9
```

### List Available Endpoints
```bash
python scripts/treasury_api.py --list-endpoints
```

## Key Information

### Fiscal Year Calendar
- Runs **October 1 to September 30**
- FY2023 = October 1, 2022 through September 30, 2023
- September (month 09) contains fiscal year-end totals

### Monetary Values
- All amounts in **DOLLARS** (not millions)
- Divide by 1,000,000 for millions
- The `*_millions` fields are pre-calculated

### MTS Table Reference

| Table | Purpose |
|-------|---------|
| `mts_table_1` | Monthly breakdown with Year-to-Date totals |
| `mts_table_4` | Receipts by source (taxes, etc.) |
| `mts_table_5` | Outlays by agency |
| `mts_table_9` | **Best for summary totals** |

### API Filter Syntax
```
filter=field_name:eq:value
filter=record_fiscal_year:eq:2023,record_calendar_month:eq:09
filter=record_fiscal_year:in:(2020,2021,2022)
```

## Detailed API Reference

See [references/api_reference.md](references/api_reference.md) for complete field names, operators, and endpoint details.
