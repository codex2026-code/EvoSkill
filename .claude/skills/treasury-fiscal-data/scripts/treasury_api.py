#!/usr/bin/env python3
"""
Treasury Fiscal Data API Client

Queries the U.S. Treasury's Fiscal Data API (api.fiscaldata.treasury.gov) to retrieve
official financial data including receipts, outlays, and deficit/surplus figures.

Usage:
    python treasury_api.py --dataset mts_summary --fiscal-year 2023
    python treasury_api.py --dataset mts_summary --fiscal-year 2020 --fiscal-year 2021 --fiscal-year 2022
    python treasury_api.py --endpoint mts_table_9 --fiscal-year 2023 --month 9
    python treasury_api.py --list-endpoints

Examples:
    # Get fiscal year-end summary (September data) for FY2023
    python treasury_api.py --dataset mts_summary --fiscal-year 2023

    # Get multiple fiscal years for time series analysis
    python treasury_api.py --dataset mts_summary --fiscal-year 2015 --fiscal-year 2016 --fiscal-year 2017

    # Query a specific MTS table directly
    python treasury_api.py --endpoint mts_table_9 --fiscal-year 2023 --month 9

Output:
    All monetary amounts are in DOLLARS (not millions).
    To convert to millions, divide by 1,000,000.
"""

import argparse
import json
import sys
from typing import Optional

# Try to use requests if available (handles SSL better), otherwise fall back to urllib
try:
    import requests
    USE_REQUESTS = True
except ImportError:
    USE_REQUESTS = False
    import ssl
    from urllib.request import urlopen, Request
    from urllib.parse import urlencode
    from urllib.error import HTTPError, URLError

    SSL_CONTEXT = ssl.create_default_context()
    try:
        import certifi
        SSL_CONTEXT.load_verify_locations(certifi.where())
    except ImportError:
        pass


BASE_URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"

# Key Monthly Treasury Statement (MTS) endpoints with CORRECT field names
MTS_ENDPOINTS = {
    "mts_table_1": {
        "path": "/v1/accounting/mts/mts_table_1",
        "description": "Summary of Receipts and Outlays - Monthly breakdown with Year-to-Date totals",
        "key_fields": ["record_date", "record_fiscal_year", "record_calendar_month",
                       "classification_desc", "current_month_gross_rcpt_amt",
                       "current_month_gross_outly_amt", "current_month_dfct_sur_amt"]
    },
    "mts_table_4": {
        "path": "/v1/accounting/mts/mts_table_4",
        "description": "Receipts by Source Category - Total receipts breakdown",
        "key_fields": ["record_date", "record_fiscal_year", "record_calendar_month",
                       "classification_desc", "current_fytd_net_rcpt_amt"]
    },
    "mts_table_5": {
        "path": "/v1/accounting/mts/mts_table_5",
        "description": "Outlays by Agency and Budget Subfunction",
        "key_fields": ["record_date", "record_fiscal_year", "record_calendar_month",
                       "classification_desc", "current_fytd_net_outly_amt"]
    },
    "mts_table_9": {
        "path": "/v1/accounting/mts/mts_table_9",
        "description": "Summary of Receipts, Outlays - Fiscal Year Totals by Category",
        "key_fields": ["record_date", "record_fiscal_year", "record_calendar_month",
                       "classification_desc", "parent_id", "current_fytd_rcpt_outly_amt"]
    }
}

# Additional commonly used datasets
OTHER_ENDPOINTS = {
    "debt_to_penny": {
        "path": "/v2/accounting/od/debt_to_penny",
        "description": "Total Public Debt Outstanding (daily)",
        "key_fields": ["record_date", "tot_pub_debt_out_amt", "intragov_hold_amt",
                       "debt_held_public_amt"]
    },
    "avg_interest_rates": {
        "path": "/v2/accounting/od/avg_interest_rates",
        "description": "Average Interest Rates on U.S. Treasury Securities",
        "key_fields": ["record_date", "security_desc", "avg_interest_rate_amt"]
    }
}


def build_filter_string(fiscal_year: Optional[int] = None,
                         month: Optional[int] = None,
                         fiscal_years: Optional[list] = None) -> str:
    """
    Build the filter query string for the Treasury API.

    The API uses a specific filter syntax:
    - Single value: filter=field_name:eq:value
    - Multiple values: filter=field_name:in:(value1,value2,value3)
    - Combined filters: filter=field1:eq:value1,field2:eq:value2

    Args:
        fiscal_year: Single fiscal year to filter (e.g., 2023)
        month: Calendar month number (1-12, where 9 = September = fiscal year end)
        fiscal_years: List of fiscal years to filter

    Returns:
        Filter string for the API query
    """
    filters = []

    if fiscal_years and len(fiscal_years) > 1:
        years_str = ",".join(str(y) for y in fiscal_years)
        filters.append(f"record_fiscal_year:in:({years_str})")
    elif fiscal_year:
        filters.append(f"record_fiscal_year:eq:{fiscal_year}")
    elif fiscal_years and len(fiscal_years) == 1:
        filters.append(f"record_fiscal_year:eq:{fiscal_years[0]}")

    if month:
        filters.append(f"record_calendar_month:eq:{month:02d}")

    return ",".join(filters) if filters else ""


def query_treasury_api(endpoint_path: str,
                       filter_str: Optional[str] = None,
                       sort: Optional[str] = None,
                       page_size: int = 1000) -> dict:
    """
    Query the Treasury Fiscal Data API.

    Args:
        endpoint_path: API endpoint path (e.g., "/v1/accounting/mts/mts_table_9")
        filter_str: Filter query string
        sort: Sort specification (e.g., "-record_date" for descending)
        page_size: Number of records per page (max 10000)

    Returns:
        JSON response from the API
    """
    url = f"{BASE_URL}{endpoint_path}"

    params = {
        "page[size]": page_size,
        "format": "json"
    }

    if filter_str:
        params["filter"] = filter_str
    if sort:
        params["sort"] = sort

    if USE_REQUESTS:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    else:
        from urllib.parse import urlencode
        query_string = urlencode(params)
        full_url = f"{url}?{query_string}"
        request = Request(full_url, headers={"Accept": "application/json"})
        with urlopen(request, timeout=30, context=SSL_CONTEXT) as response:
            return json.loads(response.read().decode("utf-8"))


def get_fiscal_year_summary(fiscal_years: list) -> dict:
    """
    Get fiscal year-end summary data (receipts, outlays, deficit/surplus).

    Uses MTS Table 9 which contains the summary by category.
    For end-of-fiscal-year data, filters to September (month 09).

    Args:
        fiscal_years: List of fiscal years to retrieve

    Returns:
        Dictionary with fiscal year summary data including:
        - total_receipts: Total government receipts in dollars
        - total_outlays: Total government outlays in dollars
        - deficit_surplus: Calculated as receipts - outlays (negative = deficit)
    """
    result = {
        "source": "U.S. Treasury Fiscal Data API",
        "endpoint": "mts_table_9",
        "unit": "dollars",
        "note": "Divide by 1,000,000 to convert to millions of dollars",
        "fiscal_years": {},
        "errors": []
    }

    for fy in fiscal_years:
        filter_str = f"record_fiscal_year:eq:{fy},record_calendar_month:eq:09"

        try:
            api_response = query_treasury_api(
                endpoint_path=MTS_ENDPOINTS["mts_table_9"]["path"],
                filter_str=filter_str
            )

            data = api_response.get("data", [])

            if not data:
                result["errors"].append(f"No data found for FY{fy}")
                continue

            # Find the parent IDs for Receipts and Net Outlays sections
            receipts_parent_id = None
            outlays_parent_id = None

            for row in data:
                desc = row.get("classification_desc", "")
                parent_id = row.get("parent_id", "")

                if desc == "Receipts" and parent_id == "null":
                    receipts_parent_id = row.get("classification_id")
                elif desc == "Net Outlays" and parent_id == "null":
                    outlays_parent_id = row.get("classification_id")

            # Now find the Total rows under each parent
            total_receipts = None
            total_outlays = None
            record_date = None

            for row in data:
                desc = row.get("classification_desc", "")
                parent_id = row.get("parent_id", "")
                amt = row.get("current_fytd_rcpt_outly_amt")

                if not record_date:
                    record_date = row.get("record_date")

                if desc == "Total":
                    if parent_id == receipts_parent_id:
                        total_receipts = float(amt) if amt and amt != "null" else None
                    elif parent_id == outlays_parent_id:
                        total_outlays = float(amt) if amt and amt != "null" else None

            # Calculate deficit/surplus
            deficit_surplus = None
            if total_receipts is not None and total_outlays is not None:
                deficit_surplus = total_receipts - total_outlays

            result["fiscal_years"][str(fy)] = {
                "record_date": record_date,
                "total_receipts": total_receipts,
                "total_outlays": total_outlays,
                "deficit_surplus": deficit_surplus,
                "total_receipts_millions": total_receipts / 1_000_000 if total_receipts else None,
                "total_outlays_millions": total_outlays / 1_000_000 if total_outlays else None,
                "deficit_surplus_millions": deficit_surplus / 1_000_000 if deficit_surplus else None
            }

        except Exception as e:
            result["errors"].append(f"Error fetching FY{fy}: {str(e)}")

    if not result["errors"]:
        del result["errors"]

    return result


def query_raw_endpoint(endpoint_key: str,
                       fiscal_year: Optional[int] = None,
                       month: Optional[int] = None,
                       custom_filter: Optional[str] = None) -> dict:
    """
    Query a specific endpoint with optional filters.

    Args:
        endpoint_key: Key from MTS_ENDPOINTS or OTHER_ENDPOINTS
        fiscal_year: Optional fiscal year filter
        month: Optional month filter (1-12)
        custom_filter: Optional custom filter string (overrides other filters)

    Returns:
        Raw API response
    """
    if endpoint_key in MTS_ENDPOINTS:
        endpoint = MTS_ENDPOINTS[endpoint_key]
    elif endpoint_key in OTHER_ENDPOINTS:
        endpoint = OTHER_ENDPOINTS[endpoint_key]
    else:
        raise ValueError(f"Unknown endpoint: {endpoint_key}. Use --list-endpoints to see available endpoints.")

    filter_str = custom_filter or build_filter_string(fiscal_year=fiscal_year, month=month)

    return query_treasury_api(
        endpoint_path=endpoint["path"],
        filter_str=filter_str,
        sort="-record_date"
    )


def list_endpoints():
    """Print available endpoints with descriptions."""
    print("Monthly Treasury Statement (MTS) Endpoints:")
    print("-" * 60)
    for key, info in MTS_ENDPOINTS.items():
        print(f"\n{key}:")
        print(f"  Path: {info['path']}")
        print(f"  Description: {info['description']}")
        print(f"  Key Fields: {', '.join(info['key_fields'][:4])}...")

    print("\n\nOther Fiscal Data Endpoints:")
    print("-" * 60)
    for key, info in OTHER_ENDPOINTS.items():
        print(f"\n{key}:")
        print(f"  Path: {info['path']}")
        print(f"  Description: {info['description']}")


def main():
    parser = argparse.ArgumentParser(
        description="Query the U.S. Treasury Fiscal Data API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get fiscal year summary for FY2023
  python treasury_api.py --dataset mts_summary --fiscal-year 2023

  # Get multiple fiscal years for analysis
  python treasury_api.py --dataset mts_summary --fiscal-year 2018 --fiscal-year 2019 --fiscal-year 2020

  # Query a specific MTS table
  python treasury_api.py --endpoint mts_table_9 --fiscal-year 2023 --month 9

  # List all available endpoints
  python treasury_api.py --list-endpoints

Notes:
  - All monetary values are in DOLLARS (not millions)
  - For fiscal year-end data, September (month 09) contains the full year totals
  - Fiscal years run from October 1 to September 30
  - Example: FY2023 = October 1, 2022 through September 30, 2023
        """
    )

    parser.add_argument("--dataset", choices=["mts_summary"],
                        help="High-level dataset to query")
    parser.add_argument("--endpoint", type=str,
                        help="Specific API endpoint to query (e.g., mts_table_9)")
    parser.add_argument("--fiscal-year", type=int, action="append", dest="fiscal_years",
                        help="Fiscal year(s) to query (can be specified multiple times)")
    parser.add_argument("--month", type=int, choices=range(1, 13), metavar="1-12",
                        help="Calendar month to filter (9 = September = fiscal year end)")
    parser.add_argument("--filter", type=str, dest="custom_filter",
                        help="Custom filter string (advanced usage)")
    parser.add_argument("--list-endpoints", action="store_true",
                        help="List all available endpoints")
    parser.add_argument("--pretty", action="store_true", default=True,
                        help="Pretty-print JSON output (default: true)")
    parser.add_argument("--compact", action="store_true",
                        help="Compact JSON output (no pretty printing)")

    args = parser.parse_args()

    if args.list_endpoints:
        list_endpoints()
        return

    if not args.dataset and not args.endpoint:
        parser.error("Either --dataset or --endpoint is required (or use --list-endpoints)")

    try:
        if args.dataset == "mts_summary":
            fiscal_years = args.fiscal_years or [2023]
            result = get_fiscal_year_summary(fiscal_years=fiscal_years)
        elif args.endpoint:
            fiscal_year = args.fiscal_years[0] if args.fiscal_years else None
            result = query_raw_endpoint(
                endpoint_key=args.endpoint,
                fiscal_year=fiscal_year,
                month=args.month,
                custom_filter=args.custom_filter
            )
        else:
            parser.error("Specify --dataset or --endpoint")
            return

        indent = None if args.compact else 2
        print(json.dumps(result, indent=indent))

    except requests.exceptions.HTTPError as e:
        print(f"API Error: {e.response.status_code} - {e.response.text}", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"Network Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
