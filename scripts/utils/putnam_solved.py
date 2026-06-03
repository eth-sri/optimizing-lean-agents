#!/usr/bin/env python3
import json
from pathlib import Path

# ----- 1) Build the target names from your string -----
s = "1962 A5, 1962 A6, 1962 B2, 1963 B1, 1964 A4, 1964 B2, 1965 A6, 1965 B2, 1965 B3, 1966 A1, 1966 A5, 1966 B6, 1967 B2, 1968 A1, 1968 A2, 1968 B2, 1969 A5, 1970 B5, 1971 A2, 1971 B1, 1971 B2, 1972 A5, 1973 B2, 1975 A1, 1975 B1, 1977 A3, 1977 A5, 1979 A3, 1979 B6, 1980 B3, 1981 B2, 1982 A5, 1984 B2, 1985 A3, 1985 A4, 1986 A1, 1986 A2, 1986 B1, 1986 B2, 1987 A1, 1987 B3, 1988 A2, 1988 B1, 1988 B2, 1990 A1, 1990 A5, 1990 A6, 1991 A2, 1991 B1, 1991 B2, 1992 A1, 1993 A2, 1993 B1, 1994 B5, 1995 A1, 1995 A3, 1995 B4, 1997 B2, 1998 B1, 1999 A1, 2000 A2, 2000 B2, 2001 A1, 2001 B1, 2001 B2, 2003 B4, 2004 A3, 2004 B2, 2004 B4, 2005 A4, 2005 B1, 2006 B5, 2007 A1, 2007 B1, 2008 A1, 2009 A1, 2010 A3, 2010 B1, 2011 B1, 2012 A2, 2012 B1, 2013 B1, 2015 A2, 2018 A1, 2019 A1"
parts = s.split(", ")
target_names = [f"putnam_{year}_{code.lower()}" for year, code in (p.split() for p in parts)]

# For quick membership tests:
target_set = set(target_names)

# ----- 2) Read input JSONL and collect matches -----
input_jsonl = Path("dataset/Putnam/putnam_rewrite.jsonl")   # <-- change if needed
output_jsonl = Path("dataset/Putnam/putnam_rewrite_solved.jsonl")

# Map by name so we can output in the same order as target_names later.
found_by_name = {}

with input_jsonl.open("r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[WARN] Skipping line {line_no}: invalid JSON ({e})")
            continue

        name = obj.get("name")
        if isinstance(name, str) and name in target_set:
            # If duplicates exist, keep the first occurrence.
            found_by_name.setdefault(name, obj)

# ----- 3) Write matches in your list’s order to a new JSONL -----
count_written = 0
with output_jsonl.open("w", encoding="utf-8") as out:
    for name in target_names:
        if name in found_by_name:
            out.write(json.dumps(found_by_name[name], ensure_ascii=False) + "\n")
            count_written += 1

print(f"Wrote {count_written} records to {output_jsonl}")

# ----- 4) Optional diagnostics: which requested names were missing? -----
missing = [n for n in target_names if n not in found_by_name]
if missing:
    print("Missing (not found in input):")
    for n in missing:
        print("  -", n)
