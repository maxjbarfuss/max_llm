# WikiText-103 Normalization

Fixes WikiText-103 formatting artifacts while preserving document structure.

## Usage

```bash
python src/data/datasets/wikitext/normalize.py input.txt output.txt
python src/data/datasets/wikitext/normalize.py input.txt output.txt --verbose
python src/data/datasets/wikitext/normalize.py --in-place data.txt
```

If no output path is given, writes to `<input>.normalized.txt`.

## Fixes Applied

- `@-@`, `@.@`, standalone `@` artifacts removed
- Spaces before `. , ! ? ; : ) ] }` and after `( [ {`
- Spaces inside quotes and around apostrophes: `Victoria 's` → `Victoria's`, `' 90s` → `'90s`
- Currency spacing: `$ 22` → `$22` (supports `$ £ € ¥`)
- En-dash/em-dash spacing: `30 – 40` → `30–40`
- Excess newlines collapsed (3+ → 2)
- Section header formatting: `= = Career = =` → `== Career ==`

## Before / After

```
 = Robert Boulter =


 Robert Boulter is an English film , television and theatre actor .
He had a guest @-@ starring role on the television series .
Released in 1998 for $ 22 , the " Dream " album was ...
The 1980s and ' 90s saw growth of 30 – 40 % .
```
```
= Robert Boulter =

Robert Boulter is an English film, television and theatre actor.
He had a guest-starring role on the television series.
Released in 1998 for $22, the "Dream" album was ...
The 1980s and '90s saw growth of 30–40%.
```

Typical result on the full WikiText-103 train split: ~4% size reduction, 20K+ formatting issues fixed.

## Options

| Flag | Description |
|---|---|
| `-v, --verbose` | Show sample output and fix statistics |
| `--in-place` | Modify file in place |
