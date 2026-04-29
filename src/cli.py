"""Subcommand CLI for the applied AI music recommender.

Three subcommands:

    python -m src.cli build   ...    # build a UserProfile from NL inputs
    python -m src.cli recommend ...  # rank the catalog against a profile
    python -m src.cli profiles ...   # list / show / edit / delete saved profiles

The `build` subcommand takes one optional flag per BuildInputs field
(--activity / --instruments / --genres /
--description). At least one must be non-blank or argparse exits with a
clear error. `--save NAME` persists the candidate profile under NAME.
`--from-profile NAME` and `--from-preset NAME` are mutually exclusive
seed sources for the re-describe-to-update flow (D38).

The `recommend` subcommand requires either --profile NAME (a saved
user profile) or --preset NAME (one of the five Module 3 presets).
Output mirrors the legacy single-shot CLI: a top-k list with grounded
explanations and mechanical reasons.

`profiles edit NAME` interactively prompts for each of the seven
UserProfile fields with the current value pre-filled; blank input keeps
the current value. Editing a preset name refuses with a directive per D33.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from typing import Sequence

from dotenv import load_dotenv

from src.agents.profile_extractor import ProfileExtractionError
from src.llm.client import CachedLLMClient, GeminiClient, LLMClient, StubLLMClient
from src.pipeline import (
    BuildInputs,
    EmptyBuildInputsError,
    ProfileBuildResult,
    RecommendationResult,
    build_profile,
    recommend,
)
from src.profiles import (
    PRESET_DISPLAY_NAMES,
    PRESET_PROFILES,
    ProfileExistsError,
    ProfileNotFoundError,
    delete_profile,
    edit_profile_fields,
    list_profiles,
    load_preset,
    load_profile,
    save_profile,
)
from src.recommender import UserProfile


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------


def _build_client(*, no_cache: bool = False) -> LLMClient:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        print(
            "[stub mode - set GEMINI_API_KEY in .env for real LLM calls]",
            file=sys.stderr,
        )
        return StubLLMClient([])
    inner = GeminiClient(api_key=api_key)
    return inner if no_cache else CachedLLMClient(inner)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


_PROFILE_FIELDS_ORDERED: tuple[str, ...] = tuple(
    f.name for f in dataclasses.fields(UserProfile)
)


def _format_profile_block(profile: UserProfile, header: str) -> list[str]:
    avoid_display = ", ".join(profile.avoid_genres) if profile.avoid_genres else "(none)"
    return [
        header,
        f"  favorite_genre:       {profile.favorite_genre}",
        f"  favorite_mood:        {profile.favorite_mood}",
        f"  target_energy:        {profile.target_energy}",
        f"  target_tempo_bpm:     {profile.target_tempo_bpm}",
        f"  target_valence:       {profile.target_valence}",
        f"  target_danceability:  {profile.target_danceability}",
        f"  target_acousticness:  {profile.target_acousticness}",
        f"  avoid_genres:         {avoid_display}",
    ]


def _format_refinement_summary(build: ProfileBuildResult) -> str:
    parts = [f"iter {step.iter_index}: {step.verdict}" for step in build.refinement_history]
    summary = "Refinement summary: " + " -> ".join(parts) if parts else "Refinement summary: (none)"
    if build.ambiguous_match:
        summary += "  [!] ambiguous match"
    return summary


def _render_build(build: ProfileBuildResult) -> str:
    out: list[str] = []
    out.extend(_format_profile_block(build.extracted_profile, "Extracted profile:"))
    if build.candidate_profile != build.extracted_profile:
        out.append("")
        out.extend(
            _format_profile_block(
                build.candidate_profile, "Candidate profile (after critic refinement):"
            )
        )
    out.append("")
    if build.extractor_warnings:
        out.append("Extractor warnings:")
        for warning in build.extractor_warnings:
            out.append(f"  - {warning}")
        out.append("")
    out.append(_format_refinement_summary(build))
    if build.cache_stats is not None:
        out.append(
            f"build cache: {build.cache_stats['hits']} hits, "
            f"{build.cache_stats['misses']} misses"
        )
    return "\n".join(out)


def _render_recommendations(rec: RecommendationResult) -> str:
    out: list[str] = []
    for index, (scored, expl) in enumerate(
        zip(rec.recommendations, rec.explanations), start=1
    ):
        out.append(
            f"{index}. {scored.song.title} - {scored.song.artist}  (score {scored.score:.2f})"
        )
        if expl.text is not None:
            out.append(f"   {expl.text}")
        else:
            out.append(f"   [mechanical reasons only - {expl.fallback_reason}]")
        for reason in scored.reasons:
            out.append(f"   - {reason}")
        out.append("")
    if rec.cache_stats is not None:
        out.append(
            f"recommend cache: {rec.cache_stats['hits']} hits, "
            f"{rec.cache_stats['misses']} misses"
        )
    return "\n".join(out)


# ---------------------------------------------------------------------------
# `build` subcommand
# ---------------------------------------------------------------------------


def _inputs_from_args(args: argparse.Namespace) -> BuildInputs:
    return BuildInputs(
        activity=args.activity,
        instruments=args.instruments,
        genres=args.genres,
        description=args.description,
    )


def _resolve_seed(args: argparse.Namespace) -> UserProfile | None:
    if args.from_profile and args.from_preset:
        raise SystemExit(
            "build: --from-profile and --from-preset are mutually exclusive"
        )
    if args.from_profile:
        try:
            return load_profile(args.from_profile)
        except ProfileNotFoundError as exc:
            raise SystemExit(f"build: {exc}")
    if args.from_preset:
        try:
            return load_preset(args.from_preset)
        except ProfileNotFoundError as exc:
            raise SystemExit(f"build: {exc}")
    return None


def _cmd_build(args: argparse.Namespace) -> int:
    load_dotenv()
    inputs = _inputs_from_args(args)
    if not inputs.has_minimum():
        print(
            "build: provide at least one of --activity / --instruments /\n"
            "       --genres / --description (or pass --help).",
            file=sys.stderr,
        )
        return 2

    seed = _resolve_seed(args)
    client = _build_client(no_cache=args.no_cache)

    try:
        build = build_profile(inputs, client, starting_from=seed)
    except EmptyBuildInputsError as exc:
        print(f"build: {exc}", file=sys.stderr)
        return 2
    except ProfileExtractionError as exc:
        print("build: could not parse a profile from the inputs - try rephrasing.", file=sys.stderr)
        print(f"  reason: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print("build: pipeline failed before producing a profile.", file=sys.stderr)
        print(f"  {exc.__class__.__name__}: {exc}", file=sys.stderr)
        return 1

    print(_render_build(build))

    if args.save:
        try:
            path = save_profile(args.save, build.candidate_profile, overwrite=not args.no_overwrite)
        except ProfileExistsError as exc:
            print(f"\nbuild: {exc}", file=sys.stderr)
            print("  re-run without --no-overwrite to clobber.", file=sys.stderr)
            return 1
        print(f"\nSaved profile {args.save!r} -> {path}")

    return 0


# ---------------------------------------------------------------------------
# `recommend` subcommand
# ---------------------------------------------------------------------------


def _resolve_target_profile(args: argparse.Namespace) -> UserProfile:
    if args.profile:
        try:
            return load_profile(args.profile)
        except ProfileNotFoundError as exc:
            raise SystemExit(f"recommend: {exc}")
    if args.preset:
        try:
            return load_preset(args.preset)
        except ProfileNotFoundError as exc:
            raise SystemExit(f"recommend: {exc}")
    raise SystemExit(
        "recommend: must pass either --profile NAME or --preset NAME"
    )


def _cmd_recommend(args: argparse.Namespace) -> int:
    load_dotenv()
    profile = _resolve_target_profile(args)
    client = _build_client(no_cache=args.no_cache)
    try:
        rec = recommend(profile, client, k=args.k)
    except Exception as exc:
        print("recommend: pipeline failed before producing recommendations.", file=sys.stderr)
        print(f"  {exc.__class__.__name__}: {exc}", file=sys.stderr)
        return 1

    label = args.profile or f"preset:{args.preset}"
    print(f"Recommendations for {label}:")
    print()
    print(_render_recommendations(rec))
    return 0


# ---------------------------------------------------------------------------
# `profiles` subcommand
# ---------------------------------------------------------------------------


def _cmd_profiles_list(args: argparse.Namespace) -> int:
    rows = list_profiles()
    if not rows:
        print("(no saved profiles)")
    else:
        print(f"{'name':<28}  created_at")
        print("-" * 60)
        for name, created in rows:
            print(f"{name:<28}  {created.isoformat()}")

    if PRESET_DISPLAY_NAMES:
        print()
        print("Presets (immutable, use `recommend --preset NAME`):")
        for slug, display in PRESET_DISPLAY_NAMES.items():
            print(f"  {slug:<24}  ({display})")
    return 0


def _cmd_profiles_show(args: argparse.Namespace) -> int:
    name = args.name
    try:
        if name in PRESET_PROFILES:
            profile = load_preset(name)
            header = f"Preset: {name} ({PRESET_DISPLAY_NAMES.get(name, name)})"
        else:
            profile = load_profile(name)
            header = f"Profile: {name}"
    except ProfileNotFoundError as exc:
        print(f"profiles show: {exc}", file=sys.stderr)
        return 1

    print("\n".join(_format_profile_block(profile, header)))
    return 0


def _prompt_for_field(field: str, current: object) -> object:
    if isinstance(current, list):
        display = ", ".join(str(x) for x in current) if current else "(none)"
        raw = input(f"  {field} [{display}]: ").strip()
        if not raw:
            return current
        # Comma-separated parse; "(none)" or "-" clears the list explicitly.
        if raw in {"(none)", "-"}:
            return []
        return [g.strip().lower() for g in raw.split(",") if g.strip()]
    raw = input(f"  {field} [{current}]: ").strip()
    if not raw:
        return current
    if isinstance(current, float):
        try:
            return float(raw)
        except ValueError:
            print(f"  (could not parse {raw!r} as float; keeping {current})", file=sys.stderr)
            return current
    return raw


def _cmd_profiles_edit(args: argparse.Namespace) -> int:
    name = args.name
    if name in PRESET_PROFILES:
        print(
            f"profiles edit: {name!r} is a preset and cannot be edited in place.\n"
            f"  Use `build --from-preset {name} --save NEW_NAME` to derive a new profile.",
            file=sys.stderr,
        )
        return 2

    try:
        existing = load_profile(name)
    except ProfileNotFoundError as exc:
        print(f"profiles edit: {exc}", file=sys.stderr)
        return 1

    print(f"Editing profile {name!r} (blank input keeps the current value):")
    updates: dict[str, object] = {}
    for field in _PROFILE_FIELDS_ORDERED:
        new_value = _prompt_for_field(field, getattr(existing, field))
        if new_value != getattr(existing, field):
            updates[field] = new_value

    if not updates:
        print("(no changes)")
        return 0

    try:
        updated = edit_profile_fields(name, **updates)
    except (ValueError, ProfileNotFoundError) as exc:
        print(f"profiles edit: {exc}", file=sys.stderr)
        return 1

    print()
    print("\n".join(_format_profile_block(updated, "Updated profile:")))
    return 0


def _cmd_profiles_delete(args: argparse.Namespace) -> int:
    name = args.name
    if name in PRESET_PROFILES:
        print(
            f"profiles delete: {name!r} is a preset and cannot be deleted.",
            file=sys.stderr,
        )
        return 2

    if not args.force:
        confirm = input(f"Delete profile {name!r}? [y/N]: ").strip().lower()
        if confirm not in {"y", "yes"}:
            print("(aborted)")
            return 0

    try:
        delete_profile(name)
    except ProfileNotFoundError as exc:
        print(f"profiles delete: {exc}", file=sys.stderr)
        return 1
    print(f"deleted: {name}")
    return 0


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="Build profiles, recommend songs, manage saved profiles.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="{build,recommend,profiles}")

    # build -----------------------------------------------------------------
    p_build = sub.add_parser(
        "build",
        help="Build a UserProfile from labeled NL inputs.",
        description=(
            "Build a UserProfile from any combination of the three guided "
            "questions plus an optional free-form description. At least one "
            "field must be non-blank."
        ),
    )
    p_build.add_argument("--activity",    help="What you're doing or want this music for")
    p_build.add_argument("--instruments", help="Acoustic, electronic, or a mix")
    p_build.add_argument("--genres",      help="Genres you want or want to avoid")
    p_build.add_argument("--description", help="Free-form mood description")
    p_build.add_argument("--save", metavar="NAME", help="Save the resulting profile under NAME")
    p_build.add_argument(
        "--no-overwrite", action="store_true",
        help="With --save, refuse to clobber an existing profile of the same NAME.",
    )
    seed = p_build.add_mutually_exclusive_group()
    seed.add_argument(
        "--from-profile", metavar="NAME",
        help="Use saved profile NAME as a seed (re-describe-to-update flow).",
    )
    seed.add_argument(
        "--from-preset", metavar="NAME",
        help="Use preset NAME as a seed (the resulting profile is a NEW user profile).",
    )
    p_build.add_argument("--no-cache", action="store_true", help="Bypass the LLM disk cache.")
    p_build.set_defaults(func=_cmd_build)

    # recommend -------------------------------------------------------------
    p_rec = sub.add_parser(
        "recommend",
        help="Rank the catalog against a profile.",
        description="Recommend top-k songs for a saved profile or a preset.",
    )
    target = p_rec.add_mutually_exclusive_group(required=True)
    target.add_argument("--profile", metavar="NAME", help="Saved user profile NAME")
    target.add_argument("--preset",  metavar="NAME", help="Preset NAME (e.g. chill_lofi)")
    p_rec.add_argument("-k", type=int, default=5, help="How many recommendations (default 5)")
    p_rec.add_argument("--no-cache", action="store_true", help="Bypass the LLM disk cache.")
    p_rec.set_defaults(func=_cmd_recommend)

    # profiles --------------------------------------------------------------
    p_profiles = sub.add_parser(
        "profiles",
        help="Manage saved profiles.",
        description="List, show, edit, or delete saved user profiles.",
    )
    p_profiles_sub = p_profiles.add_subparsers(
        dest="profiles_cmd", required=True, metavar="{list,show,edit,delete}"
    )

    p_list = p_profiles_sub.add_parser("list", help="List saved profiles + presets.")
    p_list.set_defaults(func=_cmd_profiles_list)

    p_show = p_profiles_sub.add_parser("show", help="Print the seven fields of a profile.")
    p_show.add_argument("name", help="Saved profile name OR preset slug.")
    p_show.set_defaults(func=_cmd_profiles_show)

    p_edit = p_profiles_sub.add_parser("edit", help="Interactively edit a saved profile's fields.")
    p_edit.add_argument("name", help="Saved profile name (presets are rejected).")
    p_edit.set_defaults(func=_cmd_profiles_edit)

    p_del = p_profiles_sub.add_parser("delete", help="Delete a saved profile.")
    p_del.add_argument("name", help="Saved profile name (presets are rejected).")
    p_del.add_argument("--force", action="store_true", help="Skip the confirmation prompt.")
    p_del.set_defaults(func=_cmd_profiles_delete)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
