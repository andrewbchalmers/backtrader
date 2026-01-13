"""
Fetch NASDAQ tickers for a specific year and save to CSV
Usage: python get_nasdaq_tickers.py --year 2024 --output nasdaq_2024.csv
       python get_nasdaq_tickers.py --index NASDAQ100 --output nasdaq100_2024.csv
"""

import pandas as pd
import argparse
from datetime import datetime
import requests
from bs4 import BeautifulSoup


def get_nasdaq_composite_tickers():
    """Get NASDAQ Composite tickers (all NASDAQ-listed stocks)"""
    try:
        print("Fetching NASDAQ Composite tickers from NASDAQ FTP...")

        # NASDAQ provides a free CSV of all listed stocks
        url = "ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqlisted.txt"

        df = pd.read_csv(url, sep='|')

        # Filter out the trailer row and test symbols
        df = df[df['Symbol'].notna()]
        df = df[~df['Symbol'].str.contains('File Creation Time', na=False)]
        df = df[df['Test Issue'] == 'N']  # Exclude test issues

        tickers = df['Symbol'].tolist()
        tickers = [ticker.strip() for ticker in tickers if ticker.strip()]

        print(f"✓ Found {len(tickers)} NASDAQ Composite tickers")
        return sorted(tickers)

    except Exception as e:
        print(f"Error fetching NASDAQ Composite tickers: {e}")
        return None


def get_nasdaq100_tickers():
    """Get NASDAQ-100 tickers from Wikipedia"""
    url = 'https://en.wikipedia.org/wiki/Nasdaq-100'

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        tables = pd.read_html(url, storage_options={'User-Agent': headers['User-Agent']})

        # The NASDAQ-100 components table is usually the first or second table
        df = None
        for table in tables:
            if 'Ticker' in table.columns or 'Symbol' in table.columns:
                df = table
                break

        if df is None:
            raise Exception("Could not find ticker column in tables")

        # Extract ticker column (could be 'Ticker' or 'Symbol')
        ticker_col = 'Ticker' if 'Ticker' in df.columns else 'Symbol'
        tickers = df[ticker_col].tolist()

        # Clean up tickers
        tickers = [str(ticker).replace('.', '-').strip() for ticker in tickers if pd.notna(ticker)]
        tickers = [t for t in tickers if t and not t.startswith('File')]

        print(f"✓ Found {len(tickers)} NASDAQ-100 tickers")
        return sorted(tickers)

    except Exception as e:
        print(f"Error fetching NASDAQ-100 tickers: {e}")
        return get_nasdaq100_alternative()


def get_nasdaq100_alternative():
    """Alternative method for NASDAQ-100 using BeautifulSoup"""
    url = 'https://en.wikipedia.org/wiki/Nasdaq-100'

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the components table
        tables = soup.find_all('table', class_='wikitable')

        tickers = []
        for table in tables:
            rows = table.find_all('tr')[1:]  # Skip header
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 2:  # Make sure we have enough cells
                    # Ticker is usually in first or second column
                    ticker = cells[0].text.strip() if cells[0].text.strip() else cells[1].text.strip()
                    if ticker and len(ticker) <= 5:  # Basic validation
                        ticker = ticker.replace('.', '-')
                        tickers.append(ticker)

        if tickers:
            print(f"✓ Found {len(tickers)} NASDAQ-100 tickers (alternative method)")
            return sorted(list(set(tickers)))
        else:
            raise Exception("No tickers found")

    except Exception as e:
        print(f"Alternative method also failed: {e}")
        return None


def get_nasdaq_tickers_historical(year, index_type='COMPOSITE'):
    """
    Get historical NASDAQ tickers for a specific year.

    Args:
        year: Year to fetch tickers for
        index_type: 'COMPOSITE' for all NASDAQ stocks, 'NASDAQ100' for NASDAQ-100
    """
    current_year = datetime.now().year

    if year == current_year:
        print(f"Fetching current {index_type} tickers for {year}...")
        if index_type == 'NASDAQ100':
            return get_nasdaq100_tickers()
        else:
            return get_nasdaq_composite_tickers()

    elif year < current_year:
        print(f"⚠️  Warning: Fetching historical tickers for {year}")
        print(f"    This will return current {index_type} constituents.")
        print(f"    For accurate historical constituents, consider using:")
        print(f"    - A financial data API (Bloomberg, FactSet, etc.)")
        print(f"    - Internet Archive snapshots")
        print(f"    - Historical data providers")
        print()

        # Try to get from Internet Archive for NASDAQ-100
        if index_type == 'NASDAQ100':
            print("Attempting to fetch from Internet Archive...")
            archive_tickers = get_nasdaq_from_archive(year)

            if archive_tickers:
                return archive_tickers
            else:
                print("Falling back to current tickers...\n")
                return get_nasdaq100_tickers()
        else:
            # NASDAQ Composite is too large for archive
            print("Fetching current NASDAQ Composite tickers...\n")
            return get_nasdaq_composite_tickers()

    else:
        print(f"❌ Error: Year {year} is in the future!")
        return None


def get_nasdaq_from_archive(year):
    """
    Attempt to get NASDAQ-100 tickers from Internet Archive's Wayback Machine
    """
    url = f"https://web.archive.org/web/{year}0701/https://en.wikipedia.org/wiki/Nasdaq-100"

    try:
        print(f"Trying archive URL: {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        tables = pd.read_html(url, storage_options={'User-Agent': headers['User-Agent']})

        df = None
        for table in tables:
            if 'Ticker' in table.columns or 'Symbol' in table.columns:
                df = table
                break

        if df is None:
            raise Exception("Could not find ticker column")

        ticker_col = 'Ticker' if 'Ticker' in df.columns else 'Symbol'
        tickers = df[ticker_col].tolist()
        tickers = [str(ticker).replace('.', '-').strip() for ticker in tickers if pd.notna(ticker)]

        print(f"✓ Successfully retrieved {len(tickers)} tickers from archive!")
        return sorted(tickers)

    except Exception as e:
        print(f"Could not fetch from archive: {e}")
        return None


def save_tickers_to_csv(tickers, filename):
    """Save tickers to CSV file, one per line"""
    if not tickers:
        print("No tickers to save!")
        return False

    try:
        with open(filename, 'w') as f:
            for ticker in tickers:
                f.write(f"{ticker}\n")

        print(f"✓ Saved {len(tickers)} tickers to {filename}")
        return True

    except Exception as e:
        print(f"Error saving to CSV: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Fetch NASDAQ tickers for a specific year and save to CSV'
    )
    parser.add_argument(
        '--year',
        type=int,
        default=datetime.now().year,
        help='Year to fetch tickers for (default: current year)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='nasdaq_tickers.csv',
        help='Output CSV filename (default: nasdaq_tickers.csv)'
    )
    parser.add_argument(
        '--index',
        type=str,
        choices=['COMPOSITE', 'NASDAQ100'],
        default='NASDAQ100',
        help='Which index to fetch: COMPOSITE (all NASDAQ stocks) or NASDAQ100 (default: NASDAQ100)'
    )

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"NASDAQ Ticker Fetcher")
    print(f"{'='*60}\n")

    # Get tickers
    tickers = get_nasdaq_tickers_historical(args.year, args.index)

    if tickers:
        # Save to CSV
        save_tickers_to_csv(tickers, args.output)

        # Print summary
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        print(f"Index:         {args.index}")
        print(f"Year:          {args.year}")
        print(f"Total Tickers: {len(tickers)}")
        print(f"Output File:   {args.output}")
        print(f"\nFirst 10 tickers: {', '.join(tickers[:10])}")
        print(f"Last 10 tickers:  {', '.join(tickers[-10:])}")
        print(f"{'='*60}\n")
    else:
        print("Failed to fetch tickers.")


if __name__ == "__main__":
    main()