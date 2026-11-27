"""Findora colour palettes. theme_css(name) returns a <style> block that overrides the CSS variables."""

THEMES = {
    "Cream Violet": dict(bg="#FBF6EC", bg2="#F1E6CF", text="#2A2140", muted="#6E6486",
        card="rgba(255,255,255,.72)", stroke="rgba(124,58,237,.18)", accent="#7C3AED",
        accent2="#0EA5E9", accent3="#F59E0B", star="#7C3AED", side="#F6EEDD", glow="rgba(124,58,237,.35)"),
    "Rose Sunset": dict(bg="#FFF4F1", bg2="#FFDDD2", text="#3B1D2A", muted="#86606D",
        card="rgba(255,255,255,.72)", stroke="rgba(225,29,116,.18)", accent="#E11D74",
        accent2="#F97316", accent3="#8B5CF6", star="#E11D74", side="#FFEAE4", glow="rgba(225,29,116,.32)"),
    "Mint Ocean": dict(bg="#EFFAF6", bg2="#CFEFE4", text="#123A36", muted="#4F7A74",
        card="rgba(255,255,255,.72)", stroke="rgba(13,148,136,.2)", accent="#0D9488",
        accent2="#2563EB", accent3="#F59E0B", star="#0D9488", side="#E3F6EF", glow="rgba(13,148,136,.32)"),
    "Obsidian": dict(bg="#090A0F", bg2="#11131C", text="#F4F2FF", muted="#9A96B5",
        card="rgba(255,255,255,.05)", stroke="rgba(255,255,255,.12)", accent="#8B5CF6",
        accent2="#22D3EE", accent3="#F472B6", star="#FFFFFF", side="#0D0E16", glow="rgba(139,92,246,.45)"),
}
DEFAULT_THEME = "Cream Violet"


def _stars(color: str) -> str:
    c = "%23" + color.lstrip("#")
    dots = [(30, 40, 2), (120, 90, 1.4), (210, 30, 2.2), (300, 110, 1.5), (380, 60, 1.8),
            (60, 190, 1.6), (170, 230, 2.4), (260, 180, 1.4), (350, 250, 2), (90, 330, 1.8),
            (200, 360, 1.5), (310, 340, 2.2), (400, 390, 1.4), (20, 270, 1.3)]
    lines = [(30, 40, 120, 90), (120, 90, 210, 30), (210, 30, 300, 110), (300, 110, 380, 60),
             (60, 190, 170, 230), (170, 230, 260, 180), (260, 180, 350, 250),
             (90, 330, 200, 360), (200, 360, 310, 340), (310, 340, 400, 390)]
    circles = "".join(f"<circle cx='{x}' cy='{y}' r='{r}'/>" for x, y, r in dots)
    segs = "".join(f"<line x1='{a}' y1='{b}' x2='{c2}' y2='{d}'/>" for a, b, c2, d in lines)
    return (f"url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='420' height='420'>"
            f"<g fill='{c}' fill-opacity='.5'>{circles}</g>"
            f"<g stroke='{c}' stroke-opacity='.2' stroke-width='1'>{segs}</g></svg>\")")


def theme_css(name: str) -> str:
    t = THEMES.get(name, THEMES[DEFAULT_THEME])
    v = ";".join(f"--{k}:{val}" for k, val in t.items() if k != "star")
    return f"<style>:root{{{v};--stars:{_stars(t['star'])}}}</style>"
