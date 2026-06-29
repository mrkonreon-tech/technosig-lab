from __future__ import annotations


def validate_positive_control(
    top_clusters: list[dict],
    top_n: int = 5,
) -> dict:
    checked = top_clusters[:top_n]
    matches = [c for c in checked if c.get("morphology") == "narrowband_drifting"]
    return {
        "passed": bool(matches),
        "reason": (
            "narrowband_drifting cluster found in top candidates"
            if matches
            else "no narrowband_drifting cluster found in top candidates"
        ),
        "top_n": top_n,
        "matching_count": len(matches),
        "best_match": matches[0] if matches else None,
    }

