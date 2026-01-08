"""
Fetch S&P 500 tickers for a specific year and save to CSV
Usage: python get_sp500_tickers.py --year 2020 --output sp500_2020.csv
python get_sp500_tickers.py --year 2024 --output sp500_2024.csv
"""

import pandas as pd
import argparse
from datetime import datetime
import requests
from bs4 import BeautifulSoup


def get_sp500_tickers_current():
    """Get current S&P 500 tickers from Wikipedia"""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

    try:
        # Add headers to avoid 403 Forbidden
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        # Read the table from Wikipedia with headers
        tables = pd.read_html(url, storage_options={'User-Agent': headers['User-Agent']})
        df = tables[0]

        # Extract ticker symbols
        tickers = df['Symbol'].tolist()

        # Clean up any special characters (some tickers have dots that need to be replaced)
        tickers = [ticker.replace('.', '-') for ticker in tickers]

        return sorted(tickers)

    except Exception as e:
        print(f"Error fetching S&P 500 tickers: {e}")
        print("Trying alternative method...")
        return get_sp500_alternative()


def get_sp500_alternative():
    """Alternative method using requests + BeautifulSoup"""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'id': 'constituents'})

        tickers = []
        for row in table.find_all('tr')[1:]:  # Skip header
            cells = row.find_all('td')
            if cells:
                ticker = cells[0].text.strip()
                ticker = ticker.replace('.', '-')
                tickers.append(ticker)

        return sorted(tickers)

    except Exception as e:
        print(f"Alternative method also failed: {e}")
        return None


def get_sp500_tickers_historical(year):
    """
    Get historical S&P 500 tickers for a specific year.
    Note: This gets current tickers as a baseline. For true historical accuracy,
    you'd need a paid data source or the Internet Archive.
    """
    current_year = datetime.now().year

    if year == current_year:
        print(f"Fetching current S&P 500 tickers for {year}...")
        return get_sp500_tickers_current()

    elif year < current_year:
        print(f"⚠️  Warning: Fetching historical tickers for {year}")
        print(f"    This will return current S&P 500 constituents.")
        print(f"    For accurate historical constituents, consider using:")
        print(f"    - A financial data API (Bloomberg, FactSet, etc.)")
        print(f"    - Internet Archive snapshots of Wikipedia")
        print(f"    - Historical data providers")
        print()

        # Try to get from Internet Archive
        print("Attempting to fetch from Internet Archive...")
        archive_tickers = get_sp500_from_archive(year)

        if archive_tickers:
            return archive_tickers
        else:
            print("Falling back to current tickers...\n")
            return get_sp500_tickers_current()

    else:
        print(f"❌ Error: Year {year} is in the future!")
        return None


def get_sp500_from_archive(year):
    """
    Attempt to get S&P 500 tickers from Internet Archive's Wayback Machine
    This tries to find a snapshot from the specified year
    """
    # Try to get a snapshot from July of the specified year
    url = f"https://web.archive.org/web/{year}0701/https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    try:
        print(f"Trying archive URL: {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        tables = pd.read_html(url, storage_options={'User-Agent': headers['User-Agent']})
        df = tables[0]
        tickers = df['Symbol'].tolist()
        tickers = [ticker.replace('.', '-') for ticker in tickers]
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
        description='Fetch S&P 500 tickers for a specific year and save to CSV'
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
        default='sp500_tickers.csv',
        help='Output CSV filename (default: sp500_tickers.csv)'
    )

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"S&P 500 Ticker Fetcher")
    print(f"{'='*60}\n")

    # Get tickers
    tickers = get_sp500_tickers_historical(args.year)

    if tickers:
        # Save to CSV
        save_tickers_to_csv(tickers, args.output)

        # Print summary
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        print(f"Year:          {args.year}")
        print(f"Total Tickers: {len(tickers)}")
        print(f"Output File:   {args.output}")
        print(f"\nFirst 10 tickers: {', '.join(tickers[:10])}")
        print(f"{'='*60}\n")
    else:
        print("Failed to fetch tickers.")


if __name__ == "__main__":
    main()