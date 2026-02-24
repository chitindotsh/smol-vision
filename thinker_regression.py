#!/usr/bin/env python3
"""
Thinker mode regression harness for qwen_asr (Qwen2.5-Omni-7B).

Validates generative text output using keyword presence checks rather than
exact text matching, since thinker mode uses temperature sampling.

Usage examples:
  # Run all thinker tests
  ./thinker_regression.py

  # Run with custom model dir
  ./thinker_regression.py --model-dir qwen2.5-omni-7b

  # Run only audio tests
  ./thinker_regression.py --audio-only

  # Run only text tests
  ./thinker_regression.py --text-only

  # Run determinism check (greedy temp=0, verifies identical output across runs)
  ./thinker_regression.py --determinism-only

  # Show full model output
  ./thinker_regression.py --verbose
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# ---- ANSI colors ----

_USE_COLOR = hasattr(sys.stdout, "isatty") and sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

def _sgr(code: str) -> str:
    return f"\033[{code}m" if _USE_COLOR else ""

C_RESET   = _sgr("0")
C_BOLD    = _sgr("1")
C_DIM     = _sgr("2")
C_RED     = _sgr("31")
C_GREEN   = _sgr("32")
C_YELLOW  = _sgr("33")
C_CYAN    = _sgr("36")
C_BRED    = _sgr("1;31")
C_BGREEN  = _sgr("1;32")
C_BYELLOW = _sgr("1;33")
C_BCYAN   = _sgr("1;36")
C_BWHITE  = _sgr("1;37")


def fmt_time(secs: float) -> str:
    if secs < 60:
        return f"{secs:.1f}s"
    m, s = divmod(int(secs), 60)
    return f"{m}m{s:02d}s"


# ---- Test case definitions ----

@dataclass
class ThinkerTest:
    """A single thinker mode test case."""
    name: str
    prompt: str
    # At least one of audio_file or user_text must be set
    audio_file: Optional[str] = None         # relative to samples_root
    user_text: Optional[str] = None          # --text argument
    # Keywords: output (lowercased) must contain ALL of these
    required_keywords: List[str] = field(default_factory=list)
    # Forbidden keywords: output must NOT contain any of these
    forbidden_keywords: List[str] = field(default_factory=list)
    # Minimum output length (characters)
    min_length: int = 5
    # Maximum output length (characters, 0 = no limit)
    max_length: int = 0
    # Sampling parameters
    temperature: float = 0.3
    repeat_penalty: float = 1.2
    top_k: int = 40
    max_tokens: int = 256
    # Extra CLI flags
    extra_args: List[str] = field(default_factory=list)
    # Category for filtering
    category: str = "general"


# -- Audio Q&A tests --

AUDIO_TESTS = [
    ThinkerTest(
        name="jfk_speaker_identity",
        prompt="Who is the speaker, what was his job?",
        audio_file="jfk.wav",
        required_keywords=["kennedy", "president"],
        temperature=0.3,
        repeat_penalty=1.2,
        category="audio",
    ),
    ThinkerTest(
        name="jfk_topic",
        prompt="What is the main topic of this speech? Answer in one sentence.",
        audio_file="jfk.wav",
        # Model may say "inauguration", "patriotism", "civic duty", etc. — all valid
        # Just verify it produces a substantive sentence-length answer
        min_length=20,
        temperature=0.3,
        repeat_penalty=1.2,
        max_tokens=128,
        category="audio",
    ),
    ThinkerTest(
        name="jfk_language",
        prompt="What language is being spoken? Answer with just the language name.",
        audio_file="jfk.wav",
        required_keywords=["english"],
        temperature=0.1,
        max_tokens=32,
        category="audio",
    ),
    ThinkerTest(
        name="jfk_sentiment",
        prompt="Describe the tone and sentiment of this speech in 2-3 words.",
        audio_file="jfk.wav",
        min_length=3,
        max_tokens=64,
        category="audio",
    ),
    ThinkerTest(
        name="movie_scene_description",
        prompt="Describe what is happening in this audio clip. Who is talking and what are they saying?",
        audio_file="night_of_the_living_dead_1968/45s_dont_be_afraid_of_me.wav",
        min_length=20,
        max_tokens=256,
        category="audio",
    ),
]

# -- Text-only tests --

TEXT_TESTS = [
    ThinkerTest(
        name="math_simple",
        prompt="You are a helpful assistant. Be concise.",
        user_text="What is 2+2?",
        required_keywords=["4"],
        min_length=1,
        temperature=0.3,
        max_tokens=64,
        category="text",
    ),
    ThinkerTest(
        name="capital_france",
        prompt="You are a helpful assistant. Answer in one word.",
        user_text="What is the capital of France?",
        required_keywords=["paris"],
        temperature=0.1,
        max_tokens=32,
        category="text",
    ),
    ThinkerTest(
        name="translate_french",
        prompt="You are a translator. Translate to French.",
        user_text="Good morning",
        required_keywords=["bonjour"],
        temperature=0.3,
        max_tokens=32,
        category="text",
    ),
    ThinkerTest(
        name="list_primes",
        prompt="You are a math tutor. Be concise.",
        user_text="List the first 4 prime numbers, separated by commas.",
        required_keywords=["2", "3", "5", "7"],
        min_length=1,
        temperature=0.3,
        repeat_penalty=1.2,
        max_tokens=64,
        category="text",
    ),
    ThinkerTest(
        name="creative_poem",
        prompt="You are a creative writer.",
        user_text="Write a short 4-line poem about the ocean.",
        min_length=40,
        temperature=0.8,
        top_k=50,
        max_tokens=256,
        category="text",
    ),
]

# -- Audio + Text combined tests --

COMBINED_TESTS = [
    ThinkerTest(
        name="jfk_summarize_audio",
        prompt="You are a helpful assistant.",
        audio_file="jfk.wav",
        user_text="Summarize this audio in one sentence.",
        min_length=20,
        temperature=0.3,
        max_tokens=128,
        category="combined",
    ),
    ThinkerTest(
        name="jfk_question_with_context",
        prompt="Answer the user's question based on the audio.",
        audio_file="jfk.wav",
        user_text="Is the speaker optimistic or pessimistic about the future?",
        min_length=10,
        max_tokens=128,
        category="combined",
    ),
]

ALL_TESTS = AUDIO_TESTS + TEXT_TESTS + COMBINED_TESTS


# ---- Runner ----

def run_thinker(
    binary: Path,
    model_dir: Path,
    test: ThinkerTest,
    samples_root: Path,
    timeout_s: int,
) -> Tuple[int, str, str, float]:
    """Run a single thinker test case, return (rc, stdout, stderr, elapsed_s)."""
    cmd = [
        str(binary),
        "-d", str(model_dir),
        "--thinker",
        "--silent",
        "--prompt", test.prompt,
        "--temp", str(test.temperature),
        "--repeat-penalty", str(test.repeat_penalty),
        "--top-k", str(test.top_k),
        "--max-tokens", str(test.max_tokens),
    ]

    if test.audio_file:
        wav_path = samples_root / test.audio_file
        cmd += ["-i", str(wav_path)]

    if test.user_text:
        cmd += ["--text", test.user_text]

    cmd += test.extra_args

    t0 = time.monotonic()
    cp = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    elapsed = time.monotonic() - t0
    return cp.returncode, cp.stdout.strip(), cp.stderr.strip(), elapsed


def check_keywords(output: str, test: ThinkerTest) -> Tuple[bool, List[str]]:
    """Check keyword requirements. Returns (all_pass, list_of_failure_reasons)."""
    lower = output.lower()
    reasons = []

    for kw in test.required_keywords:
        if kw.lower() not in lower:
            reasons.append(f"missing required keyword: \"{kw}\"")

    for kw in test.forbidden_keywords:
        if kw.lower() in lower:
            reasons.append(f"found forbidden keyword: \"{kw}\"")

    if len(output) < test.min_length:
        reasons.append(f"output too short: {len(output)} < {test.min_length} chars")

    if test.max_length > 0 and len(output) > test.max_length:
        reasons.append(f"output too long: {len(output)} > {test.max_length} chars")

    return len(reasons) == 0, reasons


def run_all_tests(
    tests: List[ThinkerTest],
    binary: Path,
    model_dir: Path,
    samples_root: Path,
    timeout_s: int,
    verbose: bool,
) -> int:
    """Run a list of thinker tests, return number of failures."""
    total = len(tests)
    failures = 0
    t_start = time.monotonic()

    print(f"\n{C_BOLD}Running {total} thinker test(s){C_RESET}")
    print()

    for idx, test in enumerate(tests, 1):
        # Check if audio file exists
        if test.audio_file:
            wav_path = samples_root / test.audio_file
            if not wav_path.exists():
                print(
                    f"{C_BYELLOW}[SKIP {idx}/{total}]{C_RESET} "
                    f"{C_BWHITE}{test.name}{C_RESET} | missing: {wav_path}"
                )
                continue

        print(
            f"{C_BCYAN}[START {idx}/{total}]{C_RESET} {C_BWHITE}{test.name}{C_RESET} ...",
            flush=True,
        )

        try:
            rc, stdout, stderr, elapsed = run_thinker(
                binary=binary,
                model_dir=model_dir,
                test=test,
                samples_root=samples_root,
                timeout_s=timeout_s,
            )
        except subprocess.TimeoutExpired:
            failures += 1
            print(
                f"[DONE: {C_RED}FAIL{C_RESET} {idx}/{total}] "
                f"{C_BWHITE}{test.name}{C_RESET} | {C_RED}TIMEOUT{C_RESET}"
            )
            continue

        if rc != 0:
            failures += 1
            errmsg = stderr[:200] if stderr else f"exit code {rc}"
            print(
                f"[DONE: {C_RED}FAIL{C_RESET} {idx}/{total}] "
                f"{C_BWHITE}{test.name}{C_RESET} | {C_RED}crashed: {errmsg}{C_RESET}"
            )
            continue

        ok, reasons = check_keywords(stdout, test)

        if ok:
            preview = stdout[:80] + ("..." if len(stdout) > 80 else "")
            print(
                f"[DONE: {C_GREEN}OK{C_RESET}   {idx}/{total}] "
                f"{C_BWHITE}{test.name}{C_RESET} | "
                f"{C_DIM}{fmt_time(elapsed)}{C_RESET} | "
                f"{C_DIM}\"{preview}\"{C_RESET}"
            )
        else:
            failures += 1
            print(
                f"[DONE: {C_RED}FAIL{C_RESET} {idx}/{total}] "
                f"{C_BWHITE}{test.name}{C_RESET} | "
                f"{C_DIM}{fmt_time(elapsed)}{C_RESET}"
            )
            for reason in reasons:
                print(f"       {C_RED}{reason}{C_RESET}")
            preview = stdout[:200] + ("..." if len(stdout) > 200 else "")
            print(f"       {C_DIM}output: \"{preview}\"{C_RESET}")

        if verbose and stdout:
            print(f"       {C_CYAN}full output:{C_RESET}")
            for line in stdout.split("\n"):
                print(f"       {C_DIM}| {line}{C_RESET}")

    total_time = time.monotonic() - t_start
    print()
    if failures:
        print(
            f"{C_BRED}Thinker regression FAILED: "
            f"{failures}/{total} test(s) failed ({fmt_time(total_time)} total){C_RESET}"
        )
        return failures
    print(
        f"{C_BGREEN}Thinker regression PASSED: "
        f"{total}/{total} test(s) passed ({fmt_time(total_time)} total){C_RESET}"
    )
    return 0


def run_determinism_check(
    binary: Path,
    model_dir: Path,
    samples_root: Path,
    timeout_s: int,
    verbose: bool,
) -> int:
    """Run the same greedy prompt twice and verify identical output."""
    print(f"\n{C_BOLD}Running determinism check (greedy, temp=0){C_RESET}")
    print()

    cases = [
        ThinkerTest(
            name="determinism_text",
            prompt="You are a helpful assistant.",
            user_text="What is the capital of France?",
            temperature=0.0,
            max_tokens=64,
            category="determinism",
        ),
        ThinkerTest(
            name="determinism_audio",
            prompt="Who is speaking in this audio?",
            audio_file="jfk.wav",
            temperature=0.0,
            max_tokens=128,
            category="determinism",
        ),
    ]

    failures = 0
    total = len(cases)

    for idx, test in enumerate(cases, 1):
        if test.audio_file:
            wav_path = samples_root / test.audio_file
            if not wav_path.exists():
                print(
                    f"{C_BYELLOW}[SKIP {idx}/{total}]{C_RESET} "
                    f"{C_BWHITE}{test.name}{C_RESET} | missing: {wav_path}"
                )
                continue

        print(
            f"{C_BCYAN}[START {idx}/{total}]{C_RESET} "
            f"{C_BWHITE}{test.name}{C_RESET} (2 runs) ...",
            flush=True,
        )

        outputs = []
        times = []
        for run_num in range(2):
            try:
                rc, stdout, stderr, elapsed = run_thinker(
                    binary=binary,
                    model_dir=model_dir,
                    test=test,
                    samples_root=samples_root,
                    timeout_s=timeout_s,
                )
            except subprocess.TimeoutExpired:
                failures += 1
                print(
                    f"[DONE: {C_RED}FAIL{C_RESET} {idx}/{total}] "
                    f"{C_BWHITE}{test.name}{C_RESET} | run {run_num+1} TIMEOUT"
                )
                break
            if rc != 0:
                failures += 1
                print(
                    f"[DONE: {C_RED}FAIL{C_RESET} {idx}/{total}] "
                    f"{C_BWHITE}{test.name}{C_RESET} | run {run_num+1} crashed (rc={rc})"
                )
                break
            outputs.append(stdout)
            times.append(elapsed)
        else:
            # Both runs completed
            if outputs[0] == outputs[1]:
                preview = outputs[0][:80] + ("..." if len(outputs[0]) > 80 else "")
                print(
                    f"[DONE: {C_GREEN}OK{C_RESET}   {idx}/{total}] "
                    f"{C_BWHITE}{test.name}{C_RESET} | "
                    f"identical across 2 runs | "
                    f"{C_DIM}{fmt_time(times[0])}/{fmt_time(times[1])}{C_RESET} | "
                    f"{C_DIM}\"{preview}\"{C_RESET}"
                )
            else:
                failures += 1
                print(
                    f"[DONE: {C_RED}FAIL{C_RESET} {idx}/{total}] "
                    f"{C_BWHITE}{test.name}{C_RESET} | "
                    f"{C_RED}outputs differ across greedy runs{C_RESET}"
                )
                # Show diff
                r1_preview = outputs[0][:150] + ("..." if len(outputs[0]) > 150 else "")
                r2_preview = outputs[1][:150] + ("..." if len(outputs[1]) > 150 else "")
                print(f"       {C_GREEN}run 1: \"{r1_preview}\"{C_RESET}")
                print(f"       {C_RED}run 2: \"{r2_preview}\"{C_RESET}")

            if verbose:
                for i, out in enumerate(outputs):
                    print(f"       {C_CYAN}run {i+1} output:{C_RESET}")
                    for line in out.split("\n"):
                        print(f"       {C_DIM}| {line}{C_RESET}")

    print()
    if failures:
        print(f"{C_BRED}Determinism check FAILED: {failures}/{total}{C_RESET}")
    else:
        print(f"{C_BGREEN}Determinism check PASSED: {total}/{total}{C_RESET}")
    return failures


# ---- CLI ----

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Thinker mode regression suite for qwen_asr (Qwen2.5-Omni-7B)"
    )
    ap.add_argument(
        "--samples-root",
        default="samples",
        help="Root folder for audio samples (default: samples)",
    )
    ap.add_argument(
        "--binary",
        default="./qwen_asr",
        help="Path to qwen_asr binary (default: ./qwen_asr)",
    )
    ap.add_argument(
        "--model-dir",
        default="qwen2.5-omni-7b",
        help="Model directory (default: qwen2.5-omni-7b)",
    )
    ap.add_argument(
        "--timeout-s",
        type=int,
        default=300,
        help="Per-test timeout in seconds (default: 300)",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Show full model output for each test",
    )
    # Filter flags
    ap.add_argument(
        "--audio-only",
        action="store_true",
        help="Run only audio Q&A tests",
    )
    ap.add_argument(
        "--text-only",
        action="store_true",
        help="Run only text-only tests",
    )
    ap.add_argument(
        "--combined-only",
        action="store_true",
        help="Run only audio+text combined tests",
    )
    ap.add_argument(
        "--determinism-only",
        action="store_true",
        help="Run only the greedy determinism check",
    )
    ap.add_argument(
        "--skip-determinism",
        action="store_true",
        help="Skip the greedy determinism check",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    binary = Path(args.binary).resolve()
    model_dir = Path(args.model_dir).resolve()
    samples_root = Path(args.samples_root).resolve()

    if not binary.exists():
        print(f"missing binary: {binary}", file=sys.stderr)
        return 2
    if not model_dir.exists():
        print(f"missing model dir: {model_dir}", file=sys.stderr)
        return 2
    if not samples_root.exists():
        print(f"missing samples root: {samples_root}", file=sys.stderr)
        return 2

    focused_count = sum(
        1 for f in (args.audio_only, args.text_only, args.combined_only, args.determinism_only) if f
    )
    if focused_count > 1:
        print(
            "--audio-only, --text-only, --combined-only, and --determinism-only "
            "are mutually exclusive",
            file=sys.stderr,
        )
        return 2

    print(f"{C_BOLD}Thinker regression suite{C_RESET}")
    print(f"  binary:    {binary}")
    print(f"  model:     {model_dir}")
    print(f"  samples:   {samples_root}")

    failures = 0

    if args.determinism_only:
        failures += run_determinism_check(
            binary=binary,
            model_dir=model_dir,
            samples_root=samples_root,
            timeout_s=args.timeout_s,
            verbose=args.verbose,
        )
        return 1 if failures else 0

    # Select test set
    if args.audio_only:
        tests = AUDIO_TESTS
    elif args.text_only:
        tests = TEXT_TESTS
    elif args.combined_only:
        tests = COMBINED_TESTS
    else:
        tests = ALL_TESTS

    failures += run_all_tests(
        tests=tests,
        binary=binary,
        model_dir=model_dir,
        samples_root=samples_root,
        timeout_s=args.timeout_s,
        verbose=args.verbose,
    )

    # Determinism check (unless skipped or in focused mode)
    if not args.skip_determinism and not (args.audio_only or args.text_only or args.combined_only):
        failures += run_determinism_check(
            binary=binary,
            model_dir=model_dir,
            samples_root=samples_root,
            timeout_s=args.timeout_s,
            verbose=args.verbose,
        )

    print()
    if failures:
        print(f"{C_BRED}Overall: FAILED ({failures} failure(s)){C_RESET}")
        return 1
    print(f"{C_BGREEN}Overall: PASSED{C_RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
