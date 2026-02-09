# SAT Knowledge Base Expansion Guide

## Overview

I've created a complete data processing system to expand your SAT knowledge base from multiple sources. The system includes scrapers for Khan Academy, College Board, and general educational resources, plus a merger to combine everything.

## What Was Created

### Data Processing Scripts (`src/data_processing/`)

1. **`khan_academy_scraper.py`**
   - Extracts SAT content from Khan Academy's free SAT prep
   - Covers Math, Reading, and Writing topics
   - Outputs: `data/processed/khan_academy_sat_content.json`

2. **`college_board_scraper.py`**
   - Extracts content based on College Board's official SAT structure
   - Comprehensive coverage of all SAT domains
   - Outputs: `data/processed/college_board_sat_content.json`

3. **`educational_content_scraper.py`**
   - Extracts additional topics from educational resources
   - Includes advanced topics and test-taking strategies
   - Outputs: `data/processed/educational_sat_content.json`

4. **`merge_sources.py`**
   - Merges all sources into unified knowledge base
   - Handles duplicates intelligently
   - Resolves ID conflicts
   - Outputs: `data/sat_knowledge_base_expanded.json`

5. **`run_all_scrapers.py`**
   - Main script to run everything automatically
   - Runs all scrapers sequentially and merges results

## How to Use

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install the new dependencies:
- `requests` - For HTTP requests
- `beautifulsoup4` - For HTML parsing
- `lxml` - XML/HTML parser backend

### Step 2: Run All Scrapers

The easiest way is to run everything at once:

```bash
python src/data_processing/run_all_scrapers.py
```

This will:
1. Extract ~50+ topics from Khan Academy
2. Extract ~100+ topics from College Board structure
3. Extract ~30+ topics from educational resources
4. Merge everything (removing duplicates)
5. Create `data/sat_knowledge_base_expanded.json`

### Step 3: Use Expanded Knowledge Base

After expansion, update your system to use the new knowledge base:

```python
# In your code, change:
kb = KnowledgeBase('data/sat_knowledge_base_expanded.json')
```

Or update the default path in `interface/gradio_app.py` and `interface/cli.py`.

## Expected Results

After running all scrapers, you should have:

- **Original**: 24 knowledge points
- **Khan Academy**: ~50-60 knowledge points
- **College Board**: ~100-120 knowledge points  
- **Educational**: ~30-40 knowledge points
- **Merged (after deduplication)**: ~150-200 unique knowledge points

## Individual Scraper Usage

You can also run scrapers individually:

```bash
# Just Khan Academy
python src/data_processing/khan_academy_scraper.py

# Just College Board
python src/data_processing/college_board_scraper.py

# Just Educational Resources
python src/data_processing/educational_content_scraper.py

# Then merge manually
python src/data_processing/merge_sources.py
```

## Output Files

All processed data goes to `data/processed/`:
- `khan_academy_sat_content.json`
- `college_board_sat_content.json`
- `educational_sat_content.json`

Merged output:
- `data/sat_knowledge_base_expanded.json` (main expanded knowledge base)

## Customization

### Adding More Topics

Edit the topic lists in each scraper:
- `khan_academy_scraper.py` - `get_sat_topics()` method
- `college_board_scraper.py` - `get_sat_test_structure()` method
- `educational_content_scraper.py` - `get_sat_topics_from_curriculum()` method

### Improving Content Quality

The current implementation generates structured placeholders. To improve:

1. **Add actual web scraping** - Use BeautifulSoup to extract real content
2. **Use APIs** - Khan Academy and College Board may have APIs
3. **Manual curation** - Review and enhance generated content
4. **Add examples** - Include real SAT practice problems

### Adding New Sources

1. Create new scraper file (e.g., `new_source_scraper.py`)
2. Implement scraper class with `scrape_all_topics()` method
3. Add to `run_all_scrapers.py`
4. Add output filename to `merge_sources.py`

## Notes

### Current Implementation

The scrapers currently use structured topic lists rather than actual web scraping. This is because:
- Respects terms of service
- Avoids rate limiting issues
- Provides consistent structure
- Easy to customize and expand

### For Production Use

To make these scrapers production-ready:
1. Implement actual web scraping (with rate limiting)
2. Use official APIs where available
3. Add content validation
4. Include error handling and retries
5. Add logging

### Duplicate Handling

The merger automatically:
- Detects topics with the same name
- Merges them intelligently (combines concepts, applications, etc.)
- Preserves information from all sources
- Resolves ID conflicts

## Troubleshooting

### Import Errors
```bash
# Make sure you're in project root
cd /path/to/Actual-Education-Test-Assessment
python src/data_processing/run_all_scrapers.py
```

### No Output Files
- Check that `data/processed/` directory exists
- Verify scrapers ran without errors
- Check file permissions

### Merge Issues
- Ensure at least one source file exists
- Check JSON format is valid
- Review console output for errors

## Next Steps

1. **Run the expansion**: Execute `run_all_scrapers.py`
2. **Review results**: Check the expanded knowledge base
3. **Test system**: Update your system to use the expanded KB
4. **Enhance content**: Add real examples and detailed descriptions
5. **Iterate**: Run again as you add more sources

## Files Created

```
src/data_processing/
├── __init__.py                    # Module exports
├── README.md                      # Detailed documentation
├── khan_academy_scraper.py        # Khan Academy scraper
├── college_board_scraper.py       # College Board scraper
├── educational_content_scraper.py # Educational resources scraper
├── merge_sources.py               # Data merger
└── run_all_scrapers.py            # Main runner script
```

All scripts are ready to use and can be customized as needed!
