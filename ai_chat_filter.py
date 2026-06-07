"""
╔══════════════════════════════════════════════════════╗
║         AI-POWERED CHAT FILTER SYSTEM v3.0           ║
║   Multi-Algorithm Toxicity & Offensive Content Guard ║
╚══════════════════════════════════════════════════════╝
"""

import re
import math
import time
import unicodedata
from datetime import datetime
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum


# ─────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────

class ToxicityLevel(Enum):
    CLEAN    = "✅ CLEAN"
    MILD     = "⚠️  MILD"
    MODERATE = "🟠 MODERATE"
    SEVERE   = "🔴 SEVERE"
    BLOCKED  = "🚫 BLOCKED"


@dataclass
class FilterResult:
    original_message:  str
    cleaned_message:   str
    toxicity_level:    ToxicityLevel
    toxicity_score:    float
    flags:             list[str]
    algorithm_scores:  dict[str, float]
    is_allowed:        bool
    processing_time_ms: float
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


# ─────────────────────────────────────────────
# ALGORITHM 1 — KEYWORD & PATTERN DETECTOR
# ─────────────────────────────────────────────

class KeywordPatternDetector:
    """
    Weighted keyword list with leet-speak normaliser, homoglyph
    collapse, and spaced-character evasion detection.
    """

    OFFENSIVE_PATTERNS: list = []

    RAW_KEYWORDS: list[tuple[str, int, str]] = [
        # Severe
        ("kill yourself", 3, "self-harm encouragement"),
        ("kys",           3, "self-harm abbreviation"),
        ("go die",        3, "death threat"),
        ("i will hurt",   3, "threat"),
        ("i will kill",   3, "death threat"),
        ("bomb threat",   3, "terrorism"),
        ("rape",          3, "sexual violence"),
        ("molest",        3, "sexual violence"),
        # Moderate
        ("stupid idiot",  2, "personal insult"),
        ("shut the f",    2, "profanity"),
        ("piece of sh",   2, "profanity"),
        ("go to hell",    2, "hostility"),
        ("worthless",     2, "demeaning"),
        ("loser",         2, "personal insult"),
        ("moron",         2, "personal insult"),
        ("retard",        2, "ableist slur"),
        ("retarded",      2, "ableist slur"),
        # Mild
        ("dumb",          1, "mild insult"),
        ("idiot",         1, "mild insult"),
        ("jerk",          1, "mild insult"),
        ("hate you",      1, "hostility"),
        ("terrible",      1, "negative sentiment"),
    ]

    LEET_MAP = str.maketrans({
        '0':'o','1':'i','3':'e','4':'a','5':'s',
        '6':'g','7':'t','8':'b','@':'a','$':'s','!':'i','+':'t',
    })
    HOMOGLYPH_MAP = str.maketrans({
        'а':'a','е':'e','о':'o','р':'p','с':'c','х':'x','і':'i','ї':'i',
    })

    def __init__(self):
        if not self.OFFENSIVE_PATTERNS:
            self._compile()

    def _compile(self):
        for kw, sev, tag in self.RAW_KEYWORDS:
            spaced = r'[\s\-_\.]*'.join(re.escape(c) for c in kw)
            pat = re.compile(spaced, re.IGNORECASE)
            KeywordPatternDetector.OFFENSIVE_PATTERNS.append((pat, sev, tag))

    def _normalize(self, text: str) -> str:
        text = unicodedata.normalize('NFKD', text)
        text = text.translate(self.HOMOGLYPH_MAP)
        text = text.translate(self.LEET_MAP)
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        return text.lower()

    def analyze(self, text: str) -> tuple[float, list[str]]:
        norm = self._normalize(text)
        hits = [(sev, tag) for pat, sev, tag in self.OFFENSIVE_PATTERNS if pat.search(norm)]
        if not hits:
            return 0.0, []
        max_sev = max(h[0] for h in hits)
        tags    = list({h[1] for h in hits})
        base    = max_sev / 3.0
        boost   = min(0.2, len(hits) * 0.05)
        return min(1.0, base + boost), tags

    def censor(self, text: str) -> str:
        norm   = self._normalize(text)
        result = text
        for pat, _, _ in self.OFFENSIVE_PATTERNS:
            if pat.search(norm):
                result = pat.sub(
                    lambda m: (m.group()[0] + '*' * (len(m.group()) - 2) + m.group()[-1])
                              if len(m.group()) > 2 else '**',
                    result
                )
        return result


# ─────────────────────────────────────────────
# ALGORITHM 2 — SENTIMENT & TOXICITY SCORER
# ─────────────────────────────────────────────

class SentimentToxicityScorer:
    """Lexicon-based scorer with intensifier boosting and negation skipping."""

    TOXIC_LEXICON: dict[str, float] = {
        "hate":0.80,"despise":0.80,"loathe":0.75,"disgusting":0.70,
        "horrible":0.65,"awful":0.60,"terrible":0.55,"pathetic":0.70,
        "worthless":0.75,"garbage":0.65,"trash":0.60,"scum":0.85,
        "filth":0.80,"disgusted":0.65,"ugly":0.50,"kill":0.90,
        "murder":0.95,"destroy":0.70,"attack":0.60,"hurt":0.60,
        "harm":0.55,"beat":0.55,"punish":0.50,"stab":0.90,
        "dumb":0.35,"stupid":0.40,"idiot":0.45,"jerk":0.35,
        "fool":0.35,"moron":0.45,"annoying":0.30,"useless":0.50,"lame":0.30,
    }
    INTENSIFIERS = {"very","really","so","extremely","absolutely","totally"}
    NEGATORS     = {"not","never","no","don't","won't","isn't","aren't","can't"}

    def analyze(self, text: str) -> float:
        tokens = re.findall(r"\b\w+\b", text.lower())
        scores: list[float] = []
        skip = False
        for i, tok in enumerate(tokens):
            if skip:
                skip = False
                continue
            if tok in self.NEGATORS:
                skip = True
                continue
            if tok in self.TOXIC_LEXICON:
                s = self.TOXIC_LEXICON[tok]
                if i > 0 and tokens[i-1] in self.INTENSIFIERS:
                    s = min(1.0, s * 1.3)
                scores.append(s)
        if not scores:
            return 0.0
        return round(sum(scores)/len(scores)*0.4 + max(scores)*0.6, 3)


# ─────────────────────────────────────────────
# ALGORITHM 3 — CONTEXTUAL THREAT ANALYZER
# ─────────────────────────────────────────────

class ContextualThreatAnalyzer:
    """Regex-based detector for implicit threats, harassment, and spam."""

    THREAT_PATTERNS: list[tuple] = [
        (re.compile(r"\b(i('ll| will|'m going to))\b.{0,30}\b(kill|hurt|destroy|attack|find)\b", re.I), 0.95, "direct threat"),
        (re.compile(r"\b(you('re| are))\b.{0,30}\b(dead|finished|done|over)\b", re.I),                  0.85, "implicit threat"),
        (re.compile(r"\b(watch\s+your\s+(back|step))\b", re.I),                                          0.70, "veiled threat"),
        (re.compile(r"\b(better\s+(run|hide|leave))\b", re.I),                                           0.65, "intimidation"),
        (re.compile(r"\b(no\s+one\s+(will|would)\s+(miss|care))\b", re.I),                               0.80, "harassment"),
        (re.compile(r"\b(you\s+should\s+(die|disappear|leave))\b", re.I),                                0.90, "self-harm encouragement"),
        (re.compile(r"\b(everyone\s+hates?\s+you)\b", re.I),                                             0.75, "psychological harassment"),
        (re.compile(r"\b(go\s+(back\s+to|crawl\s+under))\b", re.I),                                     0.65, "exclusion / hostility"),
    ]
    SPAM_PATTERNS: list[tuple] = [
        (re.compile(r"(.)\1{4,}"),              0.30, "character spam"),
        (re.compile(r"[A-Z]{5,}"),              0.25, "aggressive caps"),
        (re.compile(r"(!{3,}|\?{3,})"),         0.20, "punctuation spam"),
        (re.compile(r"(https?://\S+){3,}"),     0.35, "link spam"),
        (re.compile(r"\b(\w+)\b(?:\s+\1){2,}"),0.25, "word repetition"),
    ]

    def analyze(self, text: str) -> tuple[float, list[str]]:
        score = 0.0
        flags: list[str] = []
        for pat, w, label in self.THREAT_PATTERNS + self.SPAM_PATTERNS:
            if pat.search(text):
                score = max(score, w)
                flags.append(label)
        return round(score, 3), flags


# ─────────────────────────────────────────────
# ALGORITHM 4 — STATISTICAL ANOMALY DETECTOR
# ─────────────────────────────────────────────

class StatisticalAnomalyDetector:
    """Rolling Z-score + flood detection per user."""

    def __init__(self, window: int = 20):
        self._scores: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self._times:  dict[str, deque] = defaultdict(lambda: deque(maxlen=10))

    def update(self, user_id: str, base_score: float) -> float:
        now = time.time()
        hist  = self._scores[user_id]
        times = self._times[user_id]
        times.append(now)
        hist.append(base_score)

        if len(hist) < 3:
            return 0.0

        mean = sum(hist) / len(hist)
        var  = sum((x - mean)**2 for x in hist) / len(hist)
        std  = math.sqrt(var) if var > 0 else 0.0
        z    = (base_score - mean) / std if std > 0 else 0.0
        spike = min(1.0, max(0.0, z / 3.0))

        recent = [t for t in times if now - t < 10]
        flood  = min(1.0, len(recent) / 5.0) * 0.4

        return round(max(spike, flood), 3)


# ─────────────────────────────────────────────
# ALGORITHM 5 — ENTROPY / OBFUSCATION DETECTOR
# ─────────────────────────────────────────────

class EntropyObfuscationDetector:
    """Shannon entropy flags obfuscation attempts and character-spam."""

    def analyze(self, text: str) -> float:
        if len(text) < 6:
            return 0.0
        freq  = defaultdict(int)
        for ch in text:
            freq[ch] += 1
        entropy = -sum((c/len(text))*math.log2(c/len(text)) for c in freq.values())
        if entropy > 5.0:
            return min(1.0, (entropy - 5.0) / 2.0)
        if entropy < 1.0 and len(text) > 10:
            return 0.35
        return 0.0


# ─────────────────────────────────────────────
# MAIN FILTER ENGINE
# ─────────────────────────────────────────────

class AIChatFilter:
    THRESHOLDS = {
        ToxicityLevel.BLOCKED:  0.90,
        ToxicityLevel.SEVERE:   0.75,
        ToxicityLevel.MODERATE: 0.50,
        ToxicityLevel.MILD:     0.25,
    }
    ALGO_WEIGHTS = {
        "keyword":   0.30,
        "sentiment": 0.20,
        "threat":    0.30,
        "anomaly":   0.10,
        "entropy":   0.10,
    }

    def __init__(self, block_threshold: float = 0.90):
        self.block_threshold = block_threshold
        self._kw   = KeywordPatternDetector()
        self._sent = SentimentToxicityScorer()
        self._thr  = ContextualThreatAnalyzer()
        self._anom = StatisticalAnomalyDetector()
        self._ent  = EntropyObfuscationDetector()
        self.stats = {"total": 0, "blocked": 0, "flagged": 0, "clean": 0}

    def filter(self, message: str, user_id: str = "user") -> FilterResult:
        t0 = time.perf_counter()

        kw_score,  kw_flags  = self._kw.analyze(message)
        sent_score            = self._sent.analyze(message)
        thr_score, thr_flags  = self._thr.analyze(message)
        base = max(kw_score, sent_score, thr_score)
        anm_score             = self._anom.update(user_id, base)
        ent_score             = self._ent.analyze(message)

        algo_scores = {
            "keyword":   round(kw_score, 3),
            "sentiment": round(sent_score, 3),
            "threat":    round(thr_score, 3),
            "anomaly":   round(anm_score, 3),
            "entropy":   round(ent_score, 3),
        }
        composite = round(min(1.0, sum(
            s * self.ALGO_WEIGHTS[a] for a, s in algo_scores.items()
        )), 4)

        level = ToxicityLevel.CLEAN
        for lvl, thresh in self.THRESHOLDS.items():
            if composite >= thresh:
                level = lvl
                break

        all_flags  = list(set(kw_flags + thr_flags))
        is_allowed = composite < self.block_threshold

        if not is_allowed:
            cleaned = "[Message blocked by AI filter]"
        elif level != ToxicityLevel.CLEAN:
            cleaned = self._kw.censor(message)
        else:
            cleaned = message

        ms = round((time.perf_counter() - t0) * 1000, 2)

        self.stats["total"] += 1
        if not is_allowed:
            self.stats["blocked"] += 1
        elif level != ToxicityLevel.CLEAN:
            self.stats["flagged"] += 1
        else:
            self.stats["clean"] += 1

        return FilterResult(
            original_message=message,
            cleaned_message=cleaned,
            toxicity_level=level,
            toxicity_score=composite,
            flags=all_flags,
            algorithm_scores=algo_scores,
            is_allowed=is_allowed,
            processing_time_ms=ms,
        )

    def session_stats(self) -> dict:
        t = self.stats["total"] or 1
        return {
            **self.stats,
            "block_rate":  f"{self.stats['blocked']/t*100:.1f}%",
            "flagged_rate":f"{self.stats['flagged']/t*100:.1f}%",
            "clean_rate":  f"{self.stats['clean']/t*100:.1f}%",
        }


# ─────────────────────────────────────────────
# RICH TERMINAL UI
# ─────────────────────────────────────────────

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.rule import Rule
    from rich.align import Align
    from rich import box
    RICH = True
except ImportError:
    RICH = False

console = Console() if RICH else None

LEVEL_COLOR = {
    ToxicityLevel.CLEAN:    "green",
    ToxicityLevel.MILD:     "yellow",
    ToxicityLevel.MODERATE: "dark_orange",
    ToxicityLevel.SEVERE:   "red",
    ToxicityLevel.BLOCKED:  "bold red",
}


def bar(score: float, width: int = 18) -> str:
    filled = int(score * width)
    return "█" * filled + "░" * (width - filled)


def print_banner():
    if not RICH:
        print("\n" + "═"*52)
        print("   🛡️  AI CHAT FILTER SYSTEM  v3.0")
        print("   Multi-Algorithm Toxicity & Safety Guard")
        print("═"*52)
        print("  Commands:  !stats · !clear · !help · !quit\n")
        return

    lines = [
        "",
        "  [bold cyan]╔══════════════════════════════════════════════════╗[/bold cyan]",
        "  [bold white]║      🛡️  AI CHAT FILTER SYSTEM  v3.0  🛡️         ║[/bold white]",
        "  [bold cyan]║    Multi-Algorithm Toxicity & Safety Guard      ║[/bold cyan]",
        "  [bold cyan]╚══════════════════════════════════════════════════╝[/bold cyan]",
        "",
        "  [dim]Commands:[/dim]  [bold]!stats[/bold] · [bold]!clear[/bold] · [bold]!help[/bold] · [bold]!quit[/bold]",
        "",
    ]
    for l in lines:
        console.print(l)


def print_help():
    if not RICH:
        print("\n  !stats  — session statistics")
        print("  !clear  — reset statistics")
        print("  !help   — this message")
        print("  !quit   — exit\n")
        return
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t.add_column(style="bold cyan")
    t.add_column(style="dim")
    t.add_row("!stats",  "Show session statistics")
    t.add_row("!clear",  "Reset session statistics")
    t.add_row("!help",   "Show this help message")
    t.add_row("!quit",   "Exit the filter")
    console.print(Panel(t, title="[bold]Commands[/bold]", border_style="cyan", padding=(0,1)))


def print_result(r: FilterResult):
    color = LEVEL_COLOR[r.toxicity_level]

    if not RICH:
        print(f"\n  {'─'*46}")
        print(f"  Verdict  : {r.toxicity_level.value}")
        print(f"  Score    : {r.toxicity_score:.4f}  [{bar(r.toxicity_score, 16)}] {r.toxicity_score*100:.1f}%")
        print(f"  Allowed  : {'YES ✓' if r.is_allowed else 'NO ✗'}")
        print(f"  Original : {r.original_message[:65]}")
        print(f"  Output   : {r.cleaned_message[:65]}")
        if r.flags:
            print(f"  Flags    : {', '.join(r.flags)}")
        print(f"  Time     : {r.processing_time_ms} ms")
        print(f"\n  {'Algorithm':<12} {'Score':>6}  {'Weight':>6}  {'Contrib':>7}  Visual")
        print(f"  {'─'*56}")
        for algo, score in r.algorithm_scores.items():
            w = AIChatFilter.ALGO_WEIGHTS[algo]
            print(f"  {algo.capitalize():<12} {score:>6.3f}  {w:>6.2f}  {score*w:>7.4f}  [{bar(score,14)}]")
        print()
        return

    console.print(Rule(style="dim"))

    # ── main result grid ──
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold dim", min_width=12)
    grid.add_column()

    allowed_str = "[green]YES ✓[/green]" if r.is_allowed else "[red]NO ✗[/red]"
    score_str   = (f"[{color}]{r.toxicity_score:.4f}[/{color}]  "
                   f"[{color}]{bar(r.toxicity_score)}[/{color}]  "
                   f"[{color}]{r.toxicity_score*100:.1f}%[/{color}]")

    grid.add_row("VERDICT",  Text(r.toxicity_level.value, style=f"bold {color}"))
    grid.add_row("SCORE",    score_str)
    grid.add_row("ALLOWED",  allowed_str)
    grid.add_row("ORIGINAL", f"[dim italic]{r.original_message[:75]}[/dim italic]")
    grid.add_row("OUTPUT",   f"[bold]{r.cleaned_message[:75]}[/bold]")
    if r.flags:
        grid.add_row("FLAGS",    f"[yellow]{', '.join(r.flags)}[/yellow]")
    grid.add_row("TIME",     f"[dim]{r.processing_time_ms} ms  ·  {r.timestamp}[/dim]")

    console.print(Panel(grid, border_style=color, padding=(0, 2)))

    # ── algorithm breakdown ──
    at = Table(box=box.SIMPLE_HEAD, show_header=True,
               header_style="bold cyan", style="dim", padding=(0, 1))
    at.add_column("Algorithm",    style="cyan",  min_width=12)
    at.add_column("Score",        justify="right")
    at.add_column("Weight",       justify="right")
    at.add_column("Contribution", justify="right")
    at.add_column("Visual",       min_width=22)

    for algo, score in r.algorithm_scores.items():
        w   = AIChatFilter.ALGO_WEIGHTS[algo]
        con = round(score * w, 4)
        bc  = "green" if score < 0.30 else ("yellow" if score < 0.60 else "red")
        at.add_row(
            algo.capitalize(),
            f"{score:.3f}",
            f"{w:.2f}",
            f"{con:.4f}",
            f"[{bc}][{bar(score, 16)}] {score*100:.0f}%[/{bc}]",
        )
    console.print(at)


def print_stats(stats: dict):
    if not RICH:
        print("\n── Session Statistics ──")
        for k, v in stats.items():
            print(f"  {k.replace('_',' ').title():<16}: {v}")
        print()
        return

    t = Table(title="📊 Session Statistics", box=box.ROUNDED,
              border_style="cyan", show_header=True, header_style="bold")
    t.add_column("Metric", style="bold")
    t.add_column("Value",  justify="right", style="cyan")

    icons = {"total":"📨","blocked":"🚫","flagged":"⚠️ ","clean":"✅",
             "block_rate":"📊","flagged_rate":"📊","clean_rate":"📊"}
    for k, v in stats.items():
        icon = icons.get(k, "•")
        t.add_row(f"{icon}  {k.replace('_',' ').title()}", str(v))
    console.print(t)


# ─────────────────────────────────────────────
# CHAT HISTORY DISPLAY
# ─────────────────────────────────────────────

_history: list[FilterResult] = []

def print_history_line(r: FilterResult, idx: int):
    """Print a compact one-line entry for the scrolling chat view."""
    color = LEVEL_COLOR[r.toxicity_level]
    icon  = r.toxicity_level.value.split()[0]          # just the emoji
    score = f"{r.toxicity_score*100:.0f}%"

    if RICH:
        label = f"[{color}]{icon} {score:>4}[/{color}]"
        msg   = (r.cleaned_message if not r.is_allowed else r.original_message)[:60]
        console.print(f"  {label}  [dim]{r.timestamp}[/dim]  {msg}")
    else:
        msg = (r.cleaned_message if not r.is_allowed else r.original_message)[:55]
        print(f"  {icon} {score:>4}  {r.timestamp}  {msg}")


# ─────────────────────────────────────────────
# MAIN LOOP — USER INPUT ONLY
# ─────────────────────────────────────────────

def main():
    print_banner()

    filt = AIChatFilter(block_threshold=0.90)

    if RICH:
        console.print("  [dim]Type any message and press Enter to analyse it.[/dim]\n")
    else:
        print("  Type any message and press Enter to analyse it.\n")

    while True:
        try:
            if RICH:
                raw = console.input("[bold green]  You ›[/bold green] ").strip()
            else:
                raw = input("  You › ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not raw:
            continue

        cmd = raw.lower()
        if cmd in ("!quit", "!exit", "!q"):
            break
        if cmd in ("!stats", "!s"):
            print_stats(filt.session_stats())
            continue
        if cmd in ("!clear", "!c"):
            filt.stats = {"total": 0, "blocked": 0, "flagged": 0, "clean": 0}
            _history.clear()
            if RICH:
                console.print("  [dim]Statistics reset.[/dim]\n")
            else:
                print("  Statistics reset.\n")
            continue
        if cmd in ("!help", "!h"):
            print_help()
            continue

        result = filt.filter(raw, user_id="user")
        _history.append(result)
        print_result(result)

    # ── goodbye ──
    if filt.stats["total"] > 0:
        print_stats(filt.session_stats())
    if RICH:
        console.print("\n  [bold cyan]Goodbye — stay safe online! 🛡️[/bold cyan]\n")
    else:
        print("\n  Goodbye — stay safe online!\n")


if __name__ == "__main__":
    main()