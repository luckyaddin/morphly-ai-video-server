import unittest
from pathlib import Path

from generation_timing import (
    build_generation_timing,
    calculate_ltx_frame_count,
    duration_differs_from_request,
)


class GenerationTimingTests(unittest.TestCase):
    def test_four_seconds_at_eight_fps_uses_33_frames(
        self,
    ) -> None:
        self.assertEqual(
            calculate_ltx_frame_count(4, 8),
            33,
        )

    def test_eight_seconds_at_eight_fps_uses_65_frames(
        self,
    ) -> None:
        self.assertEqual(
            calculate_ltx_frame_count(8, 8),
            65,
        )

    def test_ten_seconds_at_eight_fps_uses_81_frames(
        self,
    ) -> None:
        self.assertEqual(
            calculate_ltx_frame_count(10, 8),
            81,
        )

    def test_export_fps_matches_generation_fps(
        self,
    ) -> None:
        timing = build_generation_timing(
            requested_duration_seconds=8,
            frames=65,
            fps=8,
        )

        self.assertEqual(
            timing.pipeline_fps,
            timing.export_fps,
        )

        engine_source = Path(
            "ltx_engine.py"
        ).read_text(encoding="utf-8")

        self.assertEqual(
            engine_source.count(
                "frame_rate=timing.pipeline_fps"
            ),
            3,
        )
        self.assertEqual(
            engine_source.count(
                "fps=timing.export_fps"
            ),
            3,
        )

    def test_eight_second_request_cannot_silently_accept_four_seconds(
        self,
    ) -> None:
        self.assertTrue(
            duration_differs_from_request(
                requested_duration_seconds=8,
                actual_duration_seconds=4,
            )
        )

    def test_incorrect_frame_contract_is_rejected(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            build_generation_timing(
                requested_duration_seconds=8,
                frames=33,
                fps=8,
            )


if __name__ == "__main__":
    unittest.main()
