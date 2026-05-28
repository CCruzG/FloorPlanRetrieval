from .fit_rooms import fit_plan


class FittingError(Exception):
    pass


def fit_rooms_in_boundary(fp, boundary):
    """Fit plan rooms to query boundary and return (fitted_fp, stats)."""
    try:
        fitted_fp = fit_plan(fp, boundary)
    except Exception as exc:
        raise FittingError(f"fit error: {exc}") from exc

    stats = fitted_fp.pop("_fit_stats", {})
    return fitted_fp, stats
