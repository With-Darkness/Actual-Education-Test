# Data Processing Module

This module contains scripts to extract SAT curriculum content from various sources and merge them into a unified knowledge base.

## Overview

The data processing module includes:

1. **Khan Academy Scraper** (`khan_academy_scraper.py`)
   - Extracts SAT content from Khan Academy's free SAT prep resources
   - Khan Academy has partnered with College Board for official SAT practice

2. **College Board Scraper** (`college_board_scraper.py`)
   - Extracts SAT content structure from College Board's official specifications
   - Based on official SAT test structure and content domains

3. **Educational Content Scraper** (`educational_content_scraper.py`)
   - Extracts SAT-related content from various educational resources
   - Includes topics from open educational resources and study guides

4. **Data Merger** (`merge_sources.py`)
   - Merges knowledge points from all sources
   - Handles duplicate detection and ID conflict resolution
   - Creates a unified knowledge base

5. **Run All Scrapers** (`run_all_scrapers.py`)
   - Main script to run all scrapers sequentially
   - Automatically merges results into expanded knowledge base

## Usage

### Quick Start: Run All Scrapers

The easiest way to expand the knowledge base is to run all scrapers:

```bash
python src/data_processing/run_all_scrapers.py
```

This will:
1. Extract content from Khan Academy
2. Extract content from College Board
3. Extract content from Educational Resources
4. Merge all sources into `data/sat_knowledge_base_expanded.json`

### Run Individual Scrapers

You can also run each scraper individually:

```bash
# Khan Academy
python src/data_processing/khan_academy_scraper.py

# College Board
python src/data_processing/college_board_scraper.py

# Educational Resources
python src/data_processing/educational_content_scraper.py
```

### Merge Existing Data

If you already have extracted data files, you can merge them:

```bash
python src/data_processing/merge_sources.py
```

## Output Structure

### Individual Source Files

Each scraper saves its output to `data/processed/`:

- `khan_academy_sat_content.json`
- `college_board_sat_content.json`
- `educational_sat_content.json`

Each file contains:
```json
{
  "source": "Source Name",
  "extraction_date": "YYYY-MM-DD",
  "knowledge_points": [...]
}
```

### Merged Knowledge Base

The merger creates:
- `data/sat_knowledge_base_expanded.json`

This file follows the same structure as the original `sat_knowledge_base.json` but contains merged content from all sources.

## Knowledge Point Format

Each knowledge point follows this structure:

```json
{
  "id": "MATH_001",
  "category": "Math",
  "subcategory": "Algebra",
  "topic": "Linear Equations",
  "description": "...",
  "key_concepts": [...],
  "common_applications": [...],
  "example_problem": "...",
  "example_solution": "...",
  "difficulty": "Easy|Medium|Hard",
  "related_topics": [...],
  "source": "Source Name"
}
```

## Duplicate Handling

The merger automatically:
- Detects duplicate topics (same topic name)
- Merges duplicate entries, combining:
  - Key concepts from all sources
  - Common applications from all sources
  - Related topics from all sources
  - Uses the longest description available
  - Combines source names
- Resolves ID conflicts by generating unique IDs

## Customization

### Adding New Sources

To add a new source:

1. Create a new scraper file (e.g., `new_source_scraper.py`)
2. Implement a scraper class with:
   - `scrape_all_topics()` method
   - `save_to_json()` method
3. Add the source file to `merge_sources.py` source list
4. Update `run_all_scrapers.py` to include the new scraper

### Modifying Extraction Logic

Each scraper can be customized to:
- Extract more detailed information
- Use actual web scraping (with proper rate limiting)
- Integrate with APIs if available
- Add more sophisticated content extraction

## Notes

### Rate Limiting

All scrapers include rate limiting to be respectful to source websites:
- Default: 0.2-0.5 seconds between requests
- Adjust as needed based on source policies

### Terms of Service

Always respect:
- Website terms of service
- robots.txt files
- Rate limiting policies
- Copyright and usage rights

### Data Quality

The current implementation uses structured topic lists. For production use:
- Implement actual web scraping/API integration
- Add content validation
- Include error handling and retry logic
- Add logging for debugging

## Dependencies

Required packages (already in main requirements.txt):
- `requests` - HTTP requests
- `beautifulsoup4` - HTML parsing
- `lxml` - XML/HTML parser backend

Install with:
```bash
pip install -r requirements.txt
```

## Troubleshooting

### Import Errors

If you get import errors, ensure you're running from the project root:
```bash
cd /path/to/project
python src/data_processing/run_all_scrapers.py
```

### No Data Extracted

- Check internet connection
- Verify source websites are accessible
- Review scraper logs for errors
- Ensure output directory exists (`data/processed/`)

### Merge Errors

- Ensure at least one source file exists in `data/processed/`
- Check JSON file format is valid
- Review merge logs for duplicate detection issues

## Future Enhancements

Potential improvements:
- [ ] Actual web scraping implementation
- [ ] API integration for Khan Academy/College Board
- [ ] Content validation and quality checks
- [ ] Automatic content updates
- [ ] More sophisticated duplicate detection
- [ ] Content enrichment with examples
- [ ] Export to different formats (CSV, SQLite, etc.)
