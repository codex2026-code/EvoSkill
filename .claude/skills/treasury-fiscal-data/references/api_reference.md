# Treasury Fiscal Data API Reference

## Overview

The U.S. Treasury Fiscal Data API provides access to official government financial data. Base URL:
```
https://api.fiscaldata.treasury.gov/services/api/fiscal_service
```

## Monthly Treasury Statement (MTS) Endpoints

### mts_table_1 - Summary of Receipts and Outlays
**Path**: `/v1/accounting/mts/mts_table_1`

Contains monthly breakdown of receipts, outlays, and deficit/surplus with year-to-date totals.

**Key Fields**:
| Field | Description |
|-------|-------------|
| `record_date` | Date of the record (YYYY-MM-DD) |
| `record_fiscal_year` | Fiscal year (e.g., 2023) |
| `record_calendar_month` | Calendar month (01-12) |
| `classification_desc` | Category description (e.g., "October", "Year-to-Date") |
| `current_month_gross_rcpt_amt` | Receipts for the current month |
| `current_month_gross_outly_amt` | Outlays for the current month |
| `current_month_dfct_sur_amt` | Deficit/surplus for the current month |

**Notes**: Look for rows where `classification_desc` = "Year-to-Date" to get cumulative fiscal year totals.

---

### mts_table_4 - Receipts by Source Category
**Path**: `/v1/accounting/mts/mts_table_4`

Detailed breakdown of receipts by source (income taxes, employment taxes, excise taxes, etc.).

**Key Fields**:
| Field | Description |
|-------|-------------|
| `classification_desc` | Receipt category (e.g., "Individual Income Taxes") |
| `current_fytd_gross_rcpt_amt` | Gross receipts for fiscal year-to-date |
| `current_fytd_refund_amt` | Refunds for fiscal year-to-date |
| `current_fytd_net_rcpt_amt` | Net receipts (gross - refunds) |

---

### mts_table_5 - Outlays by Agency
**Path**: `/v1/accounting/mts/mts_table_5`

Outlays organized by government agency and budget subfunction.

**Key Fields**:
| Field | Description |
|-------|-------------|
| `classification_desc` | Agency/department name |
| `current_fytd_gross_outly_amt` | Gross outlays for fiscal year-to-date |
| `current_fytd_net_outly_amt` | Net outlays |

---

### mts_table_9 - Summary by Category (Recommended for Totals)
**Path**: `/v1/accounting/mts/mts_table_9`

**Best endpoint for total receipts, outlays, and deficit/surplus by fiscal year.**

**Key Fields**:
| Field | Description |
|-------|-------------|
| `classification_desc` | Category name |
| `parent_id` | Parent classification ID |
| `classification_id` | Unique ID for this row |
| `current_fytd_rcpt_outly_amt` | Amount (receipts or outlays depending on section) |

**Structure**:
- Rows with `classification_desc = "Receipts"` and `parent_id = "null"` are section headers
- Rows with `classification_desc = "Total"` contain summary totals
- Use `parent_id` to determine if a "Total" is for receipts or outlays

---

## Other Useful Endpoints

### debt_to_penny - National Debt
**Path**: `/v2/accounting/od/debt_to_penny`

Daily total public debt outstanding.

**Key Fields**: `tot_pub_debt_out_amt`, `debt_held_public_amt`, `intragov_hold_amt`

---

## Query Parameters

### Filter Syntax
```
filter=field_name:operator:value
```

**Operators**:
| Operator | Meaning | Example |
|----------|---------|---------|
| `eq` | Equals | `record_fiscal_year:eq:2023` |
| `lt` | Less than | `tot_pub_debt_out_amt:lt:30000000000000` |
| `lte` | Less than or equal | |
| `gt` | Greater than | |
| `gte` | Greater than or equal | |
| `in` | In list | `record_fiscal_year:in:(2020,2021,2022)` |

**Combining filters** (comma-separated):
```
filter=record_fiscal_year:eq:2023,record_calendar_month:eq:09
```

### Pagination
```
page[size]=1000    # Records per page (max 10000)
page[number]=2     # Page number
```

### Sorting
```
sort=record_date        # Ascending
sort=-record_date       # Descending (prefix with -)
```

### Format
```
format=json    # Response format
```

---

## Important Notes

### Fiscal Year Calendar
- Fiscal years run **October 1 to September 30**
- FY2023 = October 1, 2022 through September 30, 2023
- September (month 09) contains the full fiscal year totals

### Monetary Values
- All amounts are in **DOLLARS** (not millions)
- To convert to millions: divide by 1,000,000
- Values are stored as strings in the API (e.g., "4439283739920.54")
- "null" string indicates no value

### Common Gotchas
1. Field names vary between tables - always check the specific endpoint
2. Month is stored as zero-padded string: "09" not "9"
3. Filter values are case-sensitive
4. Some rows have "null" as a string value, not JSON null
5. Parent-child relationships use `parent_id` and `classification_id` fields

---

## Example Queries

### Get FY2023 Summary (September data)
```
/v1/accounting/mts/mts_table_9?filter=record_fiscal_year:eq:2023,record_calendar_month:eq:09&format=json
```

### Get Multiple Fiscal Years
```
/v1/accounting/mts/mts_table_9?filter=record_fiscal_year:in:(2020,2021,2022,2023),record_calendar_month:eq:09&format=json
```

### Get Current National Debt
```
/v2/accounting/od/debt_to_penny?sort=-record_date&page[size]=1&format=json
```
