from dataclasses import dataclass
from pathlib import Path


DURATION_TOLERANCE_SECONDS = 0.25


def calculate_ltx_frame_count(
    requested_duration_seconds: int,
    fps: int,
) -> int:
    if requested_duration_seconds <= 0:
        raise ValueError(
            "requested_duration_seconds must be positive."
        )

    if fps <= 0:
        raise ValueError(
            "fps must be positive."
        )

    nominal_frames = (
        requested_duration_seconds * fps
    )

    return (
        ((nominal_frames - 1 + 7) // 8)
        * 8
        + 1
    )


@dataclass(frozen=True)
class GenerationTiming:
    requested_duration_seconds: int
    frames: int
    fps: int

    @property
    def pipeline_fps(self) -> int:
        return self.fps

    @property
    def export_fps(self) -> int:
        return self.fps

    def output_metadata(
        self,
        actual_duration_seconds: float,
    ) -> dict[str, float | int]:
        return {
            "requested_duration_seconds": (
                self.requested_duration_seconds
            ),
            "actual_duration_seconds": round(
                actual_duration_seconds,
                3,
            ),
            "frames": self.frames,
            "fps": self.fps,
        }


def build_generation_timing(
    requested_duration_seconds: int,
    frames: int,
    fps: int,
) -> GenerationTiming:
    expected_frames = calculate_ltx_frame_count(
        requested_duration_seconds,
        fps,
    )

    if frames != expected_frames:
        raise ValueError(
            "frames does not match the requested "
            f"duration and FPS: expected {expected_frames}, "
            f"received {frames}."
        )

    if (frames - 1) % 8 != 0:
        raise ValueError(
            "frames must follow the format 8n + 1."
        )

    return GenerationTiming(
        requested_duration_seconds=(
            requested_duration_seconds
        ),
        frames=frames,
        fps=fps,
    )


def duration_differs_from_request(
    requested_duration_seconds: int,
    actual_duration_seconds: float,
    tolerance_seconds: float = (
        DURATION_TOLERANCE_SECONDS
    ),
) -> bool:
    return abs(
        actual_duration_seconds
        - requested_duration_seconds
    ) > tolerance_seconds


def probe_video_duration(
    output_path: str,
) -> float:
    import av

    path = Path(output_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Generated video not found: {output_path}"
        )

    with av.open(str(path)) as container:
        if container.duration is not None:
            duration = float(
                container.duration / av.time_base
            )

            if duration > 0:
                return duration

        video_stream = next(
            (
                stream
                for stream in container.streams
                if stream.type == "video"
            ),
            None,
        )

        if video_stream is None:
            raise RuntimeError(
                "Generated MP4 has no video stream."
            )

        if (
            video_stream.duration is not None
            and video_stream.time_base is not None
        ):
            duration = float(
                video_stream.duration
                * video_stream.time_base
            )

            if duration > 0:
                return duration

        if (
            video_stream.frames
            and video_stream.average_rate
        ):
            duration = (
                float(video_stream.frames)
                / float(video_stream.average_rate)
            )

            if duration > 0:
                return duration

    raise RuntimeError(
        "Unable to measure the generated MP4 duration."
    )
