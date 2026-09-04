#!/usr/bin/env python
"""
CAC 40 News Extraction Utility
Extracts financial news articles from Les Echos mentioning CAC 40 companies

Usage:
    python run_all_spiders.py              # Run standard extraction
    python run_all_spiders.py --debug      # Run with debug logging
    python run_all_spiders.py --no-cache   # Run without using cached pages

Requirements:
    - Scrapy must be installed (pip install -r requirements.txt)
    - Virtual environment should be activated
"""

import os
import sys
import subprocess
import json
from datetime import datetime

# Color codes for terminal output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'


def print_header():
    """Print project header"""
    print(f"\n{BLUE}{'='*60}")
    print(f"CAC 40 News Extraction - Les Echos")
    print(f"{'='*60}{RESET}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target: Financial news mentioning CAC 40 companies\n")


def setup_directories():
    """Create necessary output directories"""
    os.makedirs('data', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    print(f"{GREEN}✓{RESET} Output directories ready")
    print(f"  ├─ data/      (CSV/JSON output)")
    print(f"  └─ logs/      (Execution logs)\n")


def run_news_spider(debug=False, no_cache=False):
    """
    Execute the news spider
    
    Args:
        debug (bool): Enable debug logging
        no_cache (bool): Disable HTTP caching
    
    Returns:
        dict: Execution result with status and details
    """
    print(f"{BLUE}{'─'*60}")
    print(f"Running: les_echos_news spider")
    print(f"{'─'*60}{RESET}\n")
    
    try:
        # Build command
        cmd = ["scrapy", "crawl", "les_echos_news"]
        
        # Add debug flag if requested
        if debug:
            cmd.extend(["-a", "LOG_LEVEL=DEBUG"])
        
        # Disable cache if requested
        if no_cache:
            cmd.extend(["-a", "HTTPCACHE_ENABLED=False"])
        
        # Run spider with error output
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        status = "SUCCESS"
        print(f"{GREEN}✅ Spider completed successfully{RESET}")
        return {
            'status': status,
            'code': 0,
            'message': 'News extraction completed',
            'timestamp': datetime.now().isoformat()
        }
        
    except subprocess.CalledProcessError as e:
        status = "FAILED"
        error_msg = e.stderr or f"Exit code: {e.returncode}"
        print(f"{RED}❌ Spider failed{RESET}")
        print(f"Error: {error_msg}")
        return {
            'status': status,
            'code': e.returncode,
            'message': str(error_msg),
            'timestamp': datetime.now().isoformat()
        }
        
    except subprocess.TimeoutExpired:
        print(f"{RED}❌ Spider timeout (exceeded 1 hour){RESET}")
        return {
            'status': 'TIMEOUT',
            'code': -1,
            'message': 'Extraction exceeded 1 hour timeout',
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"{RED}❌ Unexpected error{RESET}")
        print(f"Error: {str(e)}")
        return {
            'status': 'ERROR',
            'code': -1,
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }


def check_output_files():
    """Check and report on generated output files"""
    print(f"\n{BLUE}{'─'*60}")
    print("Output Files:")
    print(f"{'─'*60}{RESET}\n")
    
    output_files = {
        'data/news_articles.csv': 'Articles (CSV)',
        'data/news_articles.json': 'Articles (JSON)',
    }
    
    found_files = {}
    total_size = 0
    
    for file_path, description in output_files.items():
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            total_size += size
            found_files[file_path] = size
            size_kb = size / 1024
            print(f"{GREEN}✅{RESET} {file_path}")
            print(f"   └─ {description} ({size_kb:.2f} KB)")
        else:
            print(f"{YELLOW}⚠️  {file_path} (not generated){RESET}")
    
    print(f"\nTotal data size: {total_size / 1024:.2f} KB")
    return len(found_files) > 0


def count_articles():
    """Count articles in JSON output"""
    try:
        json_file = 'data/news_articles.json'
        if os.path.exists(json_file):
            with open(json_file, 'r', encoding='utf-8') as f:
                articles = json.load(f)
                if isinstance(articles, list):
                    return len(articles)
    except Exception as e:
        print(f"Warning: Could not count articles - {e}")
    return 0


def print_summary(result, articles_count):
    """Print execution summary"""
    print(f"\n{BLUE}{'='*60}")
    print("EXECUTION SUMMARY")
    print(f"{'='*60}{RESET}\n")
    
    # Status
    if result['status'] == 'SUCCESS':
        status_color = GREEN
        status_icon = '✅'
    elif result['status'] == 'TIMEOUT':
        status_color = YELLOW
        status_icon = '⏱️'
    else:
        status_color = RED
        status_icon = '❌'
    
    print(f"Status:    {status_color}{status_icon} {result['status']}{RESET}")
    print(f"Started:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Finished:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if articles_count > 0:
        print(f"Articles:  {GREEN}{articles_count} articles extracted{RESET}")
    
    print(f"Logs:      logs/scraper.log")
    print(f"Data:      data/\n")
    
    return result['status'] == 'SUCCESS'


def print_usage():
    """Print usage information"""
    print(f"\n{BLUE}USAGE EXAMPLES:{RESET}")
    print("  python run_all_spiders.py              # Standard extraction")
    print("  python run_all_spiders.py --debug      # With debug logging")
    print("  python run_all_spiders.py --no-cache   # Without caching\n")


def main():
    """Main execution function"""
    # Parse command line arguments
    debug = '--debug' in sys.argv
    no_cache = '--no-cache' in sys.argv
    
    if '--help' in sys.argv or '-h' in sys.argv:
        print_usage()
        sys.exit(0)
    
    try:
        # Setup
        print_header()
        setup_directories()
        
        # Run spider
        result = run_news_spider(debug=debug, no_cache=no_cache)
        
        # Check outputs
        has_output = check_output_files()
        
        # Count articles
        articles_count = count_articles() if has_output else 0
        
        # Summary
        success = print_summary(result, articles_count)
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}⚠️  Extraction interrupted by user{RESET}")
        sys.exit(130)  # 128 + SIGINT
        
    except Exception as e:
        print(f"\n\n{RED}❌ Unexpected error: {str(e)}{RESET}")
        sys.exit(1)


if __name__ == '__main__':
    main()
