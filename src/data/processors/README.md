# WikiText-103 Normalization Tool

Comprehensive text normalization for WikiText-103 dataset, fixing common formatting issues while preserving document structure.

## Usage

### Basic Usage
```bash
python normalize_wikitext.py input.txt output.txt
```

### In-place Modification
```bash
python normalize_wikitext.py --in-place data.txt
```

### With Statistics
```bash
python normalize_wikitext.py input.txt output.txt --verbose
```

## Fixes Applied

### Character-Level
- ✅ Remove @ artifacts (`@-@`, `@.@`, standalone `@`)
- ✅ Remove leading/trailing spaces from lines
- ✅ Collapse multiple spaces/tabs to single space

### Punctuation & Spacing
- ✅ Spaces before closing punctuation: `, . ! ? ; : ) ] }`
- ✅ Spaces after opening punctuation: `( [ {`
- ✅ Multiple spaces between words

### Quotes & Apostrophes
- ✅ Quote spacing: `"text"` → `"text"` (no spaces inside quotes)
- ✅ Apostrophes in words: `Victoria 's` → `Victoria's`
- ✅ Decade formatting: `' 90s` → `'90s`

### Currency & Numbers
- ✅ Currency spacing: `$ 22` → `$22`
- ✅ Supported currencies: `$ £ € ¥`

### Punctuation
- ✅ En-dash/em-dash spacing: `30 – 40` → `30–40`
- ✅ Removes extra spaces around dashes

### Structure
- ✅ Collapse extra newlines: `3+ newlines` → `2 newlines`
- ✅ Preserves paragraph breaks (double newlines)
- ✅ Header formatting: `= = Career = =` → `== Career ==`

## Examples

### Before
```
 = Robert Boulter =


 Robert Boulter is an English film , television and theatre actor .
He had a guest @-@ starring role on the television series .
Released in 1998 for $ 22 , the " Dream " album was ...
The 1980s and ' 90s saw growth of 30 – 40 % .
```

### After
```
= Robert Boulter =

Robert Boulter is an English film, television and theatre actor. He had a guest-starring role on the television series.
Released in 1998 for $22, the "Dream" album was ...
The 1980s and '90s saw growth of 30–40%.
```

## Statistics

Typical reduction on WikiText-103:
- **File size**: ~4% reduction
- **Newlines**: 1,466 extra newlines removed (3+ → 2)
- **Formatting issues**: 20,000+ fixed

## Options

- `-v, --verbose`: Show sample output and detailed statistics
- `--in-place`: Modify file in place instead of creating new file

## Output Naming

If no output file is specified:
- Input: `data.txt` → Output: `data.normalized.txt`
- With `--in-place`: modifies original file
