"""
fit_svg.py — PNG rendering of a floor plan within a unit boundary.

Rooms are colour-coded by source_type.
Edge indices are shown in red at each edge midpoint.
Room type abbreviations and IDs are shown at room centroids.
The unit boundary is drawn as a thick dark outline.

Works in any coordinate space (normalized or mm); coordinates are auto-scaled
to fill the image canvas.
"""

import base64
import io
import math

from PIL import Image, ImageDraw, ImageFont


# ── colour palette ─────────────────────────────────────────────────────────

ROOM_COLOURS = {
    0:  (245, 235, 210),   # LivingRoom  — warm beige
    1:  (180, 210, 240),   # MasterRoom  — soft blue
    2:  (250, 240, 160),   # Kitchen     — light yellow
    3:  (180, 240, 245),   # Bathroom    — light cyan
    4:  (255, 215, 170),   # DiningRoom  — peach
    5:  (240, 200, 220),   # ChildRoom   — pink
    6:  (200, 235, 200),   # StudyRoom   — light green
    7:  (220, 205, 240),   # SecondRoom  — lilac
    8:  (195, 240, 220),   # GuestRoom   — mint
    9:  (205, 225, 200),   # Balcony     — sage
    10: (225, 210, 185),   # Entrance    — tan
    11: (210, 200, 190),   # Storage     — taupe
    12: (225, 215, 235),   # WalkIn      — pale lavender
}
DEFAULT_COLOUR = (220, 220, 220)

ROOM_ABBR = {
    0: "LR", 1: "MR", 2: "K",  3: "Ba",
    4: "DR", 5: "CR", 6: "SR", 7: "S2",
    8: "GR", 9: "Bl", 10: "En", 11: "St", 12: "Wi",
}

IMAGE_SIZE = 800
PAD = 48   # pixel padding around the drawing area


# ── coordinate transform ───────────────────────────────────────────────────

def _compute_transform(all_points: list, image_size: int, pad: int):
    """Compute (scale, offset_x, offset_y) to fit all_points into the canvas."""
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    w = max_x - min_x or 1.0
    h = max_y - min_y or 1.0
    drawable = image_size - 2 * pad
    scale = drawable / max(w, h)
    offset_x = pad + (drawable - w * scale) / 2 - min_x * scale
    offset_y = pad + (drawable - h * scale) / 2 - min_y * scale
    return scale, offset_x, offset_y


def _to_px(pt, scale, offset_x, offset_y, img_h):
    """Convert a 2-D coordinate to pixel, flipping y-axis."""
    return (
        int(pt[0] * scale + offset_x),
        int(img_h - (pt[1] * scale + offset_y)),
    )


# ── font helpers ───────────────────────────────────────────────────────────

def _load_fonts():
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for path in candidates:
        try:
            return (
                ImageFont.truetype(path, 13),
                ImageFont.truetype(path, 10),
            )
        except (IOError, OSError):
            continue
    default = ImageFont.load_default()
    return default, default


_FONT_LABEL, _FONT_EDGE = _load_fonts()


# ── rendering ──────────────────────────────────────────────────────────────

def render_plan_png(plan: dict, unit_polygon: list,
                    image_size: int = IMAGE_SIZE) -> bytes:
    """
    Render plan rooms + unit boundary to PNG.

    unit_polygon : [[x, y], ...] in the same coordinate space as plan polygons.
    Returns raw PNG bytes.
    """
    # Collect all points to compute a bounding transform
    all_pts = list(unit_polygon)
    for room in plan.get("rooms", []):
        all_pts.extend(room.get("polygon") or [])

    scale, ox, oy = _compute_transform(all_pts, image_size, PAD)

    def px(pt):
        return _to_px(pt, scale, ox, oy, image_size)

    img = Image.new("RGB", (image_size, image_size), (248, 248, 248))
    draw = ImageDraw.Draw(img)

    # ── unit boundary fill (light grey background) ─────────────────────────
    unit_px = [px(p) for p in unit_polygon]
    draw.polygon(unit_px, fill=(235, 235, 235))

    # ── rooms ──────────────────────────────────────────────────────────────
    for room in plan.get("rooms", []):
        poly = room.get("polygon") or []
        if len(poly) < 3:
            continue
        st = room.get("source_type", -1)
        colour = ROOM_COLOURS.get(st, DEFAULT_COLOUR)
        px_poly = [px(p) for p in poly]

        draw.polygon(px_poly, fill=colour, outline=(80, 80, 80))

        # Room centroid label
        cx = sum(p[0] for p in px_poly) / len(px_poly)
        cy = sum(p[1] for p in px_poly) / len(px_poly)
        abbr = ROOM_ABBR.get(st, "?")
        label = f"{abbr} {room['id']}"
        draw.text((cx - 14, cy - 8), label, font=_FONT_LABEL, fill=(30, 30, 30))

        # Edge index labels
        n = len(poly)
        for ei in range(n):
            a_px = px(poly[ei])
            b_px = px(poly[(ei + 1) % n])
            mx = int((a_px[0] + b_px[0]) / 2)
            my = int((a_px[1] + b_px[1]) / 2)
            draw.text((mx + 2, my + 2), str(ei), font=_FONT_EDGE, fill=(180, 40, 40))

    # ── unit boundary outline (drawn on top) ───────────────────────────────
    n_u = len(unit_px)
    for i in range(n_u):
        a = unit_px[i]
        b = unit_px[(i + 1) % n_u]
        draw.line([a, b], fill=(20, 20, 20), width=3)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render_plan_data_url(plan: dict, unit_polygon: list,
                         image_size: int = IMAGE_SIZE) -> str:
    """Return base64 PNG data URL (convenience wrapper)."""
    png_bytes = render_plan_png(plan, unit_polygon, image_size)
    b64 = base64.b64encode(png_bytes).decode()
    return f"data:image/png;base64,{b64}"
