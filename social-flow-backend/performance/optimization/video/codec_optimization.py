# Optimizes video codecs for performance
"""
codec_optimization.py

Heuristics for codec and parameter selection based on content type, device, and network.

- CodecAdvisor: suggests codecs and encoding parameters based on constraints.
- Extensible rule engine for adding new heuristics (e.g., low-latency vs VOD).
"""

from typing import Dict, Optional


class CodecAdvisor:
    """
    Suggest codecs and encoding parameters.

    Example usage:
        advisor = CodecAdvisor()
        params = advisor.suggest(content_type='motion', device_capabilities={'hw_accel': ['nvenc']}, network_bandwidth_kbps=1500)
    """

    DEFAULTS = {
        "video_codec": "libx264",
        "audio_codec": "aac",
        "container": "mp4",
        "profiles": {
            "low_latency": {"g": 30, "preset": "fast", "tune": "zerolatency"},
            "high_quality": {"crf": 18, "preset": "medium"},
        }
    }

    def __init__(self, profile_overrides: Optional[Dict] = None):
        self.profile_overrides = profile_overrides or {}

    def suggest(self, *, content_type: str = "generic", device_capabilities: Dict = None, network_bandwidth_kbps: int = 2000, target_latency_ms: int = 100) -> Dict:
        device_capabilities = device_capabilities or {}
        suggestion = dict(self.DEFAULTS)

        # Choose HW acceleration if available and bandwidth is high enough
        if "nvenc" in device_capabilities.get("hw_accel", []):
            suggestion["video_codec"] = "h264_nvenc"
            suggestion["container"] = "mp4"
        elif "vaapi" in device_capabilities.get("hw_accel", []):
            suggestion["video_codec"] = "h264_vaapi"

        # Content heuristics
        if content_type == "motion":
            # Motion-heavy content benefits from higher bitrate
            suggestion["target_bitrate_kbps"] = max(1500, network_bandwidth_kbps)
            suggestion["profile"] = "high_motion"
        elif content_type == "screen":
            suggestion["target_bitrate_kbps"] = min(1200, network_bandwidth_kbps)
            suggestion["profile"] = "screen_content"
            suggestion["video_codec"] = "libx264"  # sometimes x264 with tune=zerolatency is better
        else:
            suggestion["target_bitrate_kbps"] = min(2000, network_bandwidth_kbps)

        # Latency tuning
        if target_latency_ms < 200:
            suggestion["tuning"] = "low_latency"
            suggestion["preset_params"] = self.DEFAULTS["profiles"].get("low_latency", {})
        else:
            suggestion["preset_params"] = self.DEFAULTS["profiles"].get("high_quality", {})

        # Apply any profile overrides
        suggestion.update(self.profile_overrides.get(suggestion.get("profile", ""), {}))

        return suggestion
