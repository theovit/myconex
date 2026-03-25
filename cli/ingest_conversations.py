#!/usr/bin/env python3
"""
ingest_conversations.py - Ingest downloaded conversation HTML files into MYCONEX

Usage:
    python cli/ingest_conversations.py --source grok
    python cli/ingest_conversations.py --source gemini
    python cli/ingest_conversations.py --source claude
    python cli/ingest_conversations.py --all
"""
import argparse, json, os, sys, hashlib
from datetime import datetime
from html.parser import HTMLParser

class TitleExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.title = None
        self._in_title = False
    def handle_starttag(self, tag, attrs):
        if tag == 'title':
            self._in_title = True
    def handle_data(self, data):
        if self._in_title:
            self.title = data.strip()
    def handle_endtag(self, tag):
        if tag == 'title':
            self._in_title = False

def extract_title(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(4096)
        p = TitleExtractor()
        p.feed(content)
        return p.title or os.path.basename(filepath)
    except:
        return os.path.basename(filepath)

def ingest_source(source, base_dir):
    src_dir = os.path.join(base_dir, source)
    if not os.path.isdir(src_dir):
        print(f"ERROR: Directory not found: {src_dir}")
        return None
    files = sorted([f for f in os.listdir(src_dir) if f.endswith('.html')])
    results = []
    for f in files:
        fp = os.path.join(src_dir, f)
        size = os.path.getsize(fp)
        title = extract_title(fp)
        fhash = hashlib.md5(open(fp,'rb').read()).hexdigest()
        results.append({
            "id": os.path.splitext(f)[0],
            "source": source,
            "filename": f,
            "title": title,
            "size": size,
            "md5": fhash,
            "path": fp,
            "ingested_at": datetime.now().isoformat()
        })
    return results

def main():
    parser = argparse.ArgumentParser(description="Ingest conversations into MYCONEX")
    parser.add_argument("--source", choices=["grok","gemini","claude"])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--base-dir", default=os.path.expanduser("~/myconex/conversation_histories"))
    args = parser.parse_args()
    
    sources = ["grok","gemini","claude"] if args.all else [args.source] if args.source else []
    if not sources:
        parser.print_help()
        return
    
    all_results = {}
    total = 0
    for src in sources:
        print(f"Ingesting {src} conversations...")
        results = ingest_source(src, args.base_dir)
        if results:
            all_results[src] = results
            total += len(results)
            print(f"  {src}: {len(results)} conversations ingested")
    
    registry_path = os.path.join(args.base_dir, "conversation_registry.json")
    registry = {"updated": datetime.now().isoformat(), "total": total, "sources": all_results}
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"\nRegistry saved: {registry_path}")
    print(f"Total conversations ingested: {total}")

if __name__ == "__main__":
    main()
