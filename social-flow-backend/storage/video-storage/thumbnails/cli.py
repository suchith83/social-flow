"""
Small CLI to run generation locally for testing and debugging
"""

import argparse
import sys
from .generator import ThumbnailGenerator
from .models import ThumbnailSpec

def main(argv=None):
    parser = argparse.ArgumentParser(description="Thumbnail generator CLI")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--video-id", required=True, help="Video ID to use for outputs")
    parser.add_argument("--count", type=int, default=5, help="Number of thumbnails to generate")
    parser.add_argument("--sizes", default="320x180,640x360", help="Comma-separated sizes")
    args = parser.parse_args(argv)
    sizes = [s.strip() for s in args.sizes.split(",")]
    specs = []
    for s in sizes:
        w,h = s.split("x")
        specs.append(ThumbnailSpec(width=int(w), height=int(h)))
    gen = ThumbnailGenerator()
    results = gen.extract_evenly_spaced(args.video, args.video_id, count=args.count, specs=specs)
    for r in results:
        print(r.dict())

if __name__ == "__main__":
    main()
