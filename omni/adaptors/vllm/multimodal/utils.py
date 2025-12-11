import numpy.typing as npt
import vllm.envs as envs
from vllm.multimodal.image import ImageMediaIO
from vllm.multimodal.video import VideoMediaIO


async def fetch_video_async(
    self,
    video_url: str,
    *,
    image_mode: str = "RGB",
    num_frames: int = 32,
    sample_fps: int = 1,
) -> npt.NDArray:
    """
    Asynchronously load video from a HTTP or base64 data URL.

    By default, the image is converted into RGB format.
    """
    image_io = ImageMediaIO(image_mode=image_mode)
    video_io = VideoMediaIO(
        image_io, num_frames=num_frames, sample_fps=sample_fps)

    return await self.load_from_url_async(
        video_url,
        video_io,
        fetch_timeout=envs.VLLM_VIDEO_FETCH_TIMEOUT,
    )
