#!/usr/bin/env python3
"""Compare our Rust tokenizer's punctuation output against spaCy's tokenization.

Usage:
    python3 .context/spacy_compare.py
    python3 .context/spacy_compare.py --verbose
    python3 .context/spacy_compare.py --only-failures

Requires:
    pip install spacy
    python -m spacy download en_core_web_sm
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass

try:
    import spacy
except ImportError:
    print("spacy is required: pip install spacy", file=sys.stderr)
    print("Then: python -m spacy download en_core_web_sm", file=sys.stderr)
    sys.exit(1)


PUNCT_CHARS = set('.,!?;:—…()\u201c\u201d"')

# For comparison purposes, treat ASCII " and curly quotes as equivalent.
# Our pipeline may normalize " → " or " depending on position.
QUOTE_EQUIVALENTS = {'"', "\u201c", "\u201d"}

TEST_CASES = [
    # Basic sentence-ending punctuation
    "Hello.",
    "Hello!",
    "Hello?",
    # Comma
    "Hello, world.",
    # Multiple sentence-ending
    "Hello. World.",
    "Wait! What?",
    "Wait! What? Really.",
    # Multiple trailing punct
    "Really?!",
    "No way!?",
    # Ellipsis
    "Hmm...",
    "Wait... what?",
    # Brackets
    "(Hello)",
    "(yes) and (no)",
    # Quotes
    '"Hello"',
    'She said, "hello."',
    '"Wait," he said.',
    # Currency
    "$100",
    "$3.50",
    # Contractions — should NOT split the apostrophe
    "don't",
    "I'm fine.",
    "it's great.",
    "can't stop, won't stop.",
    # Numbers — should NOT split internal punct
    "3.14",
    "1,000",
    "3.14 is pi.",
    # Abbreviations
    "Mr. Smith",
    "U.S.A.",
    "Dr. Jones said hello.",
    # Semicolons and colons
    "Hello; world.",
    "Note: this works.",
    # Em-dash
    "Hello — world.",
    # Complex
    "Hello, world. How are you? I'm fine!",
    "First sentence. Second sentence. Third sentence.",
    "It's amazing how fast voice is. We should build more.",
    # Edge cases
    "",
    "hello",
    "...",
    "()",
    "!?!?",
    "42",
]


@dataclass
class TokenInfo:
    text: str
    tag: str


@dataclass
class ComparisonResult:
    input_text: str
    spacy_tokens: list[TokenInfo]
    our_phonemes: str
    spacy_punct_chars: set[str]
    our_punct_chars: set[str]
    missing: set[str]
    extra: set[str]

    @property
    def ok(self) -> bool:
        return len(self.missing) == 0


def get_spacy_tokens(nlp, text: str) -> list[TokenInfo]:
    doc = nlp(text)
    return [TokenInfo(text=tk.text, tag=tk.tag_) for tk in doc]


def normalize_punct_set(chars: set[str]) -> set[str]:
    """Normalize a set of punct characters so that all quote variants collapse to a single canonical form."""
    result = set()
    for c in chars:
        if c in QUOTE_EQUIVALENTS:
            result.add('"')  # canonical form
        else:
            result.add(c)
    return result


def extract_punct_from_spacy(tokens: list[TokenInfo]) -> set[str]:
    """Extract the set of punctuation characters that spaCy produced as separate tokens."""
    punct_tags = {".", ",", ":", "-LRB-", "-RRB-", "``", '""', "''", "#", "NFP"}
    chars = set()
    for tk in tokens:
        if tk.tag in punct_tags:
            for c in tk.text:
                if c in PUNCT_CHARS:
                    chars.add(c)
    return normalize_punct_set(chars)


def extract_punct_from_phonemes(phonemes: str) -> set[str]:
    """Extract the set of punctuation characters present in our phoneme output."""
    return normalize_punct_set({c for c in phonemes if c in PUNCT_CHARS})


def run_comparison(
    voice_binary: str, nlp, verbose: bool = False
) -> list[ComparisonResult]:
    p = subprocess.Popen(
        [voice_binary, "--jsonrpc", "-q"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )

    req_id = 0

    def rpc(method, params=None):
        nonlocal req_id
        req_id += 1
        msg = {"jsonrpc": "2.0", "method": method, "id": req_id}
        if params:
            msg["params"] = params
        p.stdin.write(json.dumps(msg) + "\n")
        p.stdin.flush()
        while True:
            line = p.stdout.readline()
            if not line:
                return None
            obj = json.loads(line)
            if "id" in obj and obj["id"] is not None:
                return obj

    rpc("ping")

    results = []

    for text in TEST_CASES:
        if not text:
            continue

        spacy_tokens = get_spacy_tokens(nlp, text)
        spacy_puncts = extract_punct_from_spacy(spacy_tokens)

        resp = rpc("speak", {"text": text, "detail": "full"})
        if resp and "result" in resp:
            phonemes = " | ".join(resp["result"]["phonemes"])
        elif resp and "error" in resp:
            phonemes = f"ERROR: {resp['error']['message']}"
        else:
            phonemes = "ERROR: no response"

        our_puncts = extract_punct_from_phonemes(phonemes)
        missing = spacy_puncts - our_puncts
        extra = our_puncts - spacy_puncts

        results.append(
            ComparisonResult(
                input_text=text,
                spacy_tokens=spacy_tokens,
                our_phonemes=phonemes,
                spacy_punct_chars=spacy_puncts,
                our_punct_chars=our_puncts,
                missing=missing,
                extra=extra,
            )
        )

    p.stdin.close()
    p.wait()
    return results


def print_results(results: list[ComparisonResult], verbose: bool, only_failures: bool):
    passed = sum(1 for r in results if r.ok)
    failed = sum(1 for r in results if not r.ok)
    total = len(results)

    header = f"{'Input':<40} {'spaCy punct':<15} {'Our punct':<15} {'Status'}"
    print(header)
    print("-" * len(header))

    for r in results:
        if only_failures and r.ok:
            continue

        spacy_str = "".join(sorted(r.spacy_punct_chars)) or "(none)"
        our_str = "".join(sorted(r.our_punct_chars)) or "(none)"
        status = (
            "\033[32m✓\033[0m"
            if r.ok
            else f"\033[31m✗ missing: {''.join(sorted(r.missing))}\033[0m"
        )

        print(f"{r.input_text:<40} {spacy_str:<15} {our_str:<15} {status}")

        if verbose and not r.ok:
            spacy_tok_str = " ".join(f"{t.text}|{t.tag}" for t in r.spacy_tokens)
            print(f"  spaCy: {spacy_tok_str}")
            print(f"  phon:  {r.our_phonemes}")
            if r.extra:
                print(f"  extra: {''.join(sorted(r.extra))}")
            print()

    print()
    print(f"Results: {passed}/{total} passed, {failed} failed")

    if failed == 0:
        print(
            "\033[32m✅ All punctuation from spaCy appears in our phoneme output.\033[0m"
        )
    else:
        print(
            f"\033[33m⚠️  {failed} input(s) have punctuation that spaCy produces but we don't.\033[0m"
        )

    return failed


def main():
    parser = argparse.ArgumentParser(
        description="Compare voice G2P punctuation against spaCy"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show details for failures"
    )
    parser.add_argument(
        "--only-failures", "-f", action="store_true", help="Only show failures"
    )
    parser.add_argument(
        "--binary",
        default="voice",
        help="Path to voice binary (default: voice from PATH)",
    )
    args = parser.parse_args()

    print("Loading spaCy model...", file=sys.stderr)
    nlp = spacy.load("en_core_web_sm")

    print("Running comparison...\n", file=sys.stderr)
    results = run_comparison(args.binary, nlp, verbose=args.verbose)
    failed = print_results(
        results, verbose=args.verbose, only_failures=args.only_failures
    )

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
