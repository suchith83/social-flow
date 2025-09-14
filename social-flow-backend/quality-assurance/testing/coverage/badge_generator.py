"""
Badge generator - create coverage badges in SVG format.
"""

import os
from .config import CONFIG


class BadgeGenerator:
    """Generates an SVG badge for coverage."""

    def __init__(self, config=CONFIG):
        self.config = config

    def generate_svg(self, percent: float) -> str:
        """Generate SVG badge content."""
        color = "red" if percent < 50 else "orange" if percent < 80 else "green"
        return f"""
<svg xmlns="http://www.w3.org/2000/svg" width="120" height="20">
  <rect width="70" height="20" fill="#555"/>
  <rect x="70" width="50" height="20" fill="{color}"/>
  <text x="35" y="14" fill="#fff" font-family="Verdana" font-size="11" text-anchor="middle">coverage</text>
  <text x="95" y="14" fill="#fff" font-family="Verdana" font-size="11" text-anchor="middle">{percent}%</text>
</svg>
"""

    def save(self, percent: float):
        """Save badge as SVG file."""
        svg = self.generate_svg(percent)
        path = os.path.join(self.config.report_dir, "badge.svg")
        with open(path, "w") as f:
            f.write(svg)
        return path
