#!/usr/bin/env python3

# -- IMPORTS

from __future__ import annotations;
from typing import Any;

def _patch_torchvision_functional_tensor() -> None:

    import sys;

    module_name = "torchvision.transforms.functional_tensor";

    if module_name in sys.modules:

        return;

    import torchvision.transforms._functional_tensor as functional_tensor;

    sys.modules[ module_name ] = functional_tensor;

_patch_torchvision_functional_tensor();

try:

    import argparse;
    from basicsr.archs.rrdbnet_arch import RRDBNet;
    import cv2;
    import ffmpeg;
    import math;
    import mimetypes;
    import numpy as np;
    from pathlib import Path;
    from realesrgan import RealESRGANer;
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact;
    import shutil;
    import sys;
    import torch;
    from tqdm import tqdm;

except ImportError as import_error:

    print( f"Missing dependency: {import_error}", file=sys.stderr );
    print( "Install with:", file=sys.stderr );
    print( "  run install_packages_cuda.bat, install_packages_rocm.bat, or install_packages_cpu.bat", file=sys.stderr );
    sys.exit( 1 );

# -- CONSTANTS

DEFAULT_MODEL_NAME = "realesrgan";
MODEL_SCALE = 4;
DEFAULT_MAX_RATIO = 4.0;
DEFAULT_COMPRESSION = 22;
DEFAULT_TILE_SIZE = 0;
AVAILABLE_MODEL_NAME_LIST = (
    "bsrgan",
    "bsrnet",
    "highfidelity",
    "realanime",
    "realdigital",
    "realesrgan",
    "realesrnet",
    "remacri",
    "ultramix",
    "ultrasharp",
    );
MODEL_BLOCK_COUNT = {
    "realdigital": 6,
    };
DEFAULT_MODEL_BLOCK_COUNT = 23;
SRVGG_MODEL_NAME_SET = {
    "realanime",
    };
DEFAULT_SRVGG_CONV_COUNT = 16;

APPLICATION_FOLDER_PATH = Path( __file__ ).resolve().parent;
MODEL_FOLDER_PATH = APPLICATION_FOLDER_PATH / "MODEL";

# -- TYPES

class CrispRealESRGANer( RealESRGANer ):

    def __init__(
        self,
        scale: int,
        model_path: str,
        model: torch.nn.Module,
        tile: int = 0,
        tile_pad: int = 10,
        pre_pad: int = 0,
        half: bool = False,
        device: torch.device | None = None,
        gpu_id: int | None = None
        ) -> None:

        self.scale = scale;
        self.tile_size = tile;
        self.tile_pad = tile_pad;
        self.pre_pad = pre_pad;
        self.mod_scale = None;
        self.half = half;

        if gpu_id is not None:

            self.device = (
                torch.device(
                    f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
                    )
                if device is None
                else device
                );

        else:

            self.device = (
                torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                    )
                if device is None
                else device
                );

        state_dict = get_state_dict_from_model_checkpoint( model_path );

        model.load_state_dict( state_dict, strict=True );
        model.eval();
        self.model = model.to( self.device );

        if self.half:

            self.model = self.model.half();

class VideoReader:

    # -- CONSTRUCTORS

    def __init__(
        self,
        video_reader_arguments: argparse.Namespace
        ) -> None:

        self.video_reader_arguments = video_reader_arguments;
        input_mime_type = (
            mimetypes.guess_type(
                video_reader_arguments.input_video_file_path
                )[ 0 ]
            );

        if input_mime_type is None or not input_mime_type.startswith( "video" ):

            raise ValueError(
                f"Input must be a video file: {video_reader_arguments.input_video_file_path}"
                );

        self.stream_reader = (
            ffmpeg.input( video_reader_arguments.input_video_file_path )
                .output(
                    "pipe:",
                    format="rawvideo",
                    pix_fmt="bgr24",
                    loglevel="error"
                    )
                .run_async(
                    pipe_stdin=True,
                    pipe_stdout=True,
                    cmd=video_reader_arguments.ffmpeg_binary
                    )
            );

        video_meta_info_dictionary = (
            get_video_meta_info(
                video_reader_arguments.input_video_file_path
                )
            );
        self.width = video_meta_info_dictionary[ "width" ];
        self.height = video_meta_info_dictionary[ "height" ];
        self.input_frames_per_second = video_meta_info_dictionary[ "frames_per_second" ];
        self.display_aspect_ratio = (
            video_meta_info_dictionary[ "display_aspect_ratio" ]
            );
        self.sample_aspect_ratio = (
            video_meta_info_dictionary[ "sample_aspect_ratio" ]
            );
        self.audio_stream = video_meta_info_dictionary[ "audio_stream" ];
        self.frame_count = video_meta_info_dictionary[ "frame_count" ];

    # -- INQUIRIES

    def get_resolution(
        self
        ) -> tuple[ int, int ]:

        return self.height, self.width;

    # ~~

    def get_frames_per_second(
        self
        ) -> float:

        if self.video_reader_arguments.frames_per_second is not None:

            return self.video_reader_arguments.frames_per_second;

        return self.input_frames_per_second;

    # ~~

    def get_display_aspect_ratio(
        self
        ) -> str | None:

        return self.display_aspect_ratio;

    # ~~

    def get_sample_aspect_ratio(
        self
        ) -> str | None:

        return self.sample_aspect_ratio;

    # ~~

    def get_audio_stream(
        self
        ) -> Any | None:

        return self.audio_stream;

    # ~~

    def __len__(
        self
        ) -> int:

        return self.frame_count;

    # -- OPERATIONS

    def get_frame(
        self
        ) -> np.ndarray | None:

        image_bytes = self.stream_reader.stdout.read(
            self.width * self.height * 3
            );

        if not image_bytes:

            return None;

        return (
            np.frombuffer( image_bytes, np.uint8 )
                .reshape(
                    [
                        self.height,
                        self.width,
                        3
                    ]
                )
            );

    # ~~

    def close(
        self
        ) -> None:

        self.stream_reader.stdin.close();
        self.stream_reader.wait();

# ~~

class VideoWriter:

    # -- CONSTRUCTORS

    def __init__(
        self,
        video_writer_arguments: argparse.Namespace,
        audio_stream: Any | None,
        output_height: int,
        output_width: int,
        output_video_file_path: str,
        frames_per_second: float,
        display_aspect_ratio: str | None = None,
        sample_aspect_ratio: str | None = None
        ) -> None:

        ffmpeg_output_argument_by_name_dictionary = {
            "pix_fmt": "yuv420p",
            "vcodec": "libx264",
            "crf": video_writer_arguments.compression,
            "loglevel": "error"
            };

        ffmpeg_output_argument_by_name_dictionary.update(
            get_video_aspect_ratio_output_options(
                display_aspect_ratio,
                sample_aspect_ratio
                )
            );

        raw_video_input = ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="bgr24",
            s=f"{output_width}x{output_height}",
            framerate=frames_per_second
            );

        if audio_stream is not None:

            self.stream_writer = (
                raw_video_input.output(
                    audio_stream,
                    output_video_file_path,
                    acodec="copy",
                    **ffmpeg_output_argument_by_name_dictionary
                    )
                .overwrite_output()
                .run_async(
                    pipe_stdin=True,
                    pipe_stdout=True,
                    cmd=video_writer_arguments.ffmpeg_binary
                    )
                );

        else:

            self.stream_writer = (
                raw_video_input.output(
                    output_video_file_path,
                    **ffmpeg_output_argument_by_name_dictionary
                    )
                .overwrite_output()
                .run_async(
                    pipe_stdin=True,
                    pipe_stdout=True,
                    cmd=video_writer_arguments.ffmpeg_binary
                    )
                );

    # -- OPERATIONS

    def write_frame(
        self,
        video_frame: np.ndarray
        ) -> None:

        self.stream_writer.stdin.write(
            video_frame.astype( np.uint8 ).tobytes()
            );

    # ~~

    def close(
        self
        ) -> None:

        self.stream_writer.stdin.close();
        self.stream_writer.wait();

# -- FUNCTIONS

def get_ffprobe_command(
    ffmpeg_binary: str
    ) -> str:

    if ffmpeg_binary == "ffmpeg":

        return "ffprobe";

    binary_name = Path( ffmpeg_binary ).name;

    if "ffmpeg" in binary_name:

        return str(
            Path( ffmpeg_binary ).with_name(
                binary_name.replace( "ffmpeg", "ffprobe" )
                )
            );

    return "ffprobe";

# ~~

def is_valid_aspect_ratio(
    aspect_ratio: str | None
    ) -> bool:

    return (
        aspect_ratio is not None
        and aspect_ratio not in ( "N/A", "0:1" )
        );

# ~~

def get_video_aspect_ratio_output_options(
    display_aspect_ratio: str | None,
    sample_aspect_ratio: str | None
    ) -> dict[str, str]:

    if is_valid_aspect_ratio( display_aspect_ratio ):

        return { "aspect": display_aspect_ratio };

    if is_valid_aspect_ratio( sample_aspect_ratio ):

        return { "sar": sample_aspect_ratio };

    return {};

# ~~

def get_output_aspect_ratios(
    output_width: int,
    output_height: int,
    input_width: int,
    input_height: int,
    display_aspect_ratio: str | None,
    sample_aspect_ratio: str | None
    ) -> tuple[ str | None, str | None ]:

    if output_width == input_width and output_height == input_height:

        return display_aspect_ratio, sample_aspect_ratio;

    # Uniform scale keeps the same picture aspect; crop does not.
    if output_width * input_height == input_width * output_height:

        return display_aspect_ratio, sample_aspect_ratio;

    # Crop/resize changed the frame geometry: do not reuse the source DAR.
    # Keep SAR when known so anamorphic pixels stay correct; otherwise set
    # DAR from the new pixel dimensions (square pixels).
    if is_valid_aspect_ratio( sample_aspect_ratio ):

        return None, sample_aspect_ratio;

    if is_valid_aspect_ratio( display_aspect_ratio ):

        return (
            get_resized_display_aspect_ratio(
                display_aspect_ratio,
                input_width,
                input_height,
                output_width,
                output_height
                ),
            None
            );

    return f"{output_width}:{output_height}", None;

# ~~

def get_aspect_ratio_parts(
    aspect_ratio: str
    ) -> tuple[ int, int ] | None:

    try:

        numerator_text, denominator_text = aspect_ratio.split( ":", 1 );
        numerator = int( numerator_text );
        denominator = int( denominator_text );

    except ValueError:

        return None;

    if numerator <= 0 or denominator <= 0:

        return None;

    return numerator, denominator;

# ~~

def get_simplified_aspect_ratio(
    numerator: int,
    denominator: int
    ) -> str:

    common_divisor = math.gcd( numerator, denominator );

    return f"{numerator // common_divisor}:{denominator // common_divisor}";

# ~~

def get_resized_display_aspect_ratio(
    display_aspect_ratio: str,
    input_width: int,
    input_height: int,
    output_width: int,
    output_height: int
    ) -> str:

    aspect_ratio_parts = get_aspect_ratio_parts( display_aspect_ratio );

    if aspect_ratio_parts is None:

        return f"{output_width}:{output_height}";

    display_width, display_height = aspect_ratio_parts;

    return get_simplified_aspect_ratio(
        display_width * output_width * input_height,
        display_height * output_height * input_width
        );

# ~~

def get_video_meta_info(
    input_video_file_path: str
    ) -> dict[str, Any]:

    video_probe_result_dictionary = ffmpeg.probe( input_video_file_path );

    video_stream_list = [
        stream
        for stream in video_probe_result_dictionary[ "streams" ]
        if stream[ "codec_type" ] == "video"
        ];

    has_audio_stream = (
        any(
            stream[ "codec_type" ] == "audio"
            for stream in video_probe_result_dictionary[ "streams" ]
            )
        );

    video_stream = video_stream_list[ 0 ];

    return (
        {
            "width": video_stream[ "width" ],
            "height": video_stream[ "height" ],
            "frames_per_second": eval( video_stream[ "avg_frame_rate" ] ),
            "display_aspect_ratio": video_stream.get( "display_aspect_ratio" ),
            "sample_aspect_ratio": video_stream.get( "sample_aspect_ratio" ),
            "audio_stream": (
                ffmpeg.input( input_video_file_path ).audio
                if has_audio_stream
                else None
                ),
            "frame_count": int( video_stream[ "nb_frames" ] )
        }
        );

# ~~

def parse_arguments(
    ) -> argparse.Namespace:

    argument_parser = (
        argparse.ArgumentParser(
            description="GPU-accelerated AI video upscaler",
            )
        );

    argument_parser.add_argument(
        "input_video_file_path",
        type=Path,
        help="Input .mp4 file"
        );

    argument_parser.add_argument(
        "output_video_file_path",
        type=Path,
        help="Output .mp4 file"
        );

    argument_parser.add_argument(
        "--compression",
        type=int,
        default=DEFAULT_COMPRESSION,
        help=f"Encoding compression (default: {DEFAULT_COMPRESSION}; higher = more compression)"
        );

    argument_parser.add_argument(
        "--crop",
        nargs=4,
        type=float,
        metavar=( "LEFT", "RIGHT", "TOP", "BOTTOM" ),
        default=None,
        help="Crop distances before upscaling (< 1 = ratio, >= 1 = pixels)"
        );

    argument_parser.add_argument(
        "--max-ratio",
        type=float,
        default=DEFAULT_MAX_RATIO,
        help=f"Maximum output scaling ratio (default: {DEFAULT_MAX_RATIO})"
        );

    argument_parser.add_argument(
        "--max-width",
        type=int,
        default=0,
        help="Maximum output width in pixels (0 = no limit)"
        );

    argument_parser.add_argument(
        "--max-height",
        type=int,
        default=0,
        help="Maximum output height in pixels (0 = no limit)"
        );

    argument_parser.add_argument(
        "--skip",
        action="store_true",
        help="Skip upscaling if output is newer than input"
        );

    argument_parser.add_argument(
        "--tile-size",
        type=int,
        default=DEFAULT_TILE_SIZE,
        help=f"Tile size if GPU runs out of memory (default: {DEFAULT_TILE_SIZE})"
        );

    argument_parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        choices=AVAILABLE_MODEL_NAME_LIST,
        help=f"Upscaling model (default: {DEFAULT_MODEL_NAME})"
        );

    compute_backend_group = argument_parser.add_mutually_exclusive_group();

    compute_backend_group.add_argument(
        "--cpu",
        action="store_const",
        const="cpu",
        dest="compute_backend",
        help="Force CPU computation"
        );

    compute_backend_group.add_argument(
        "--cuda",
        action="store_const",
        const="cuda",
        dest="compute_backend",
        help="Force NVIDIA CUDA computation"
        );

    compute_backend_group.add_argument(
        "--rocm",
        action="store_const",
        const="rocm",
        dest="compute_backend",
        help="Force AMD ROCm computation"
        );

    argument_parser.set_defaults( compute_backend=None );

    return argument_parser.parse_args();

# ~~

def validate_runtime(
    ) -> None:

    if shutil.which( "ffmpeg" ) is None:

        print(
            "ffmpeg not found. Install ffmpeg and add it to PATH.",
            file=sys.stderr
            );
        sys.exit( 1 );

    if shutil.which( get_ffprobe_command( "ffmpeg" ) ) is None:

        print(
            "ffprobe not found. Install ffmpeg with ffprobe.",
            file=sys.stderr
            );
        sys.exit( 1 );

# ~~

def validate_input_video_file_path(
    input_video_file_path: Path
    ) -> None:

    if not input_video_file_path.is_file():

        print(
            f"Input video not found: {input_video_file_path}",
            file=sys.stderr
            );
        sys.exit( 1 );

    if input_video_file_path.suffix.lower() != ".mp4":

        print(
            f"Input video must be an .mp4 file: {input_video_file_path}",
            file=sys.stderr
            );
        sys.exit( 1 );

# ~~

def validate_output_video_file_path(
    output_video_file_path: Path
    ) -> None:

    if output_video_file_path.suffix.lower() != ".mp4":

        print(
            f"Output video must be an .mp4 file: {output_video_file_path}",
            file=sys.stderr
            );
        sys.exit( 1 );

# ~~

def validate_maximum_ratio(
    maximum_ratio: float
    ) -> None:

    if maximum_ratio <= 0 or maximum_ratio > MODEL_SCALE:

        print(
            f"Maximum ratio must be greater than 0 and at most {MODEL_SCALE}: {maximum_ratio}",
            file=sys.stderr
            );
        sys.exit( 1 );

# ~~

def validate_crop_distances(
    input_width: int,
    input_height: int,
    left_distance: float,
    right_distance: float,
    top_distance: float,
    bottom_distance: float
    ) -> None:

    crop_distance_list = (
        left_distance,
        right_distance,
        top_distance,
        bottom_distance
        );

    if any( crop_distance < 0 for crop_distance in crop_distance_list ):

        print(
            "Crop distances must be greater than or equal to 0.",
            file=sys.stderr
            );
        sys.exit( 1 );

    cropped_width, cropped_height = (
        get_cropped_dimensions(
            input_width,
            input_height,
            left_distance,
            right_distance,
            top_distance,
            bottom_distance
            )
        );

    if cropped_width <= 0 or cropped_height <= 0:

        print(
            "Crop distances remove the entire frame.",
            file=sys.stderr
            );
        sys.exit( 1 );

# ~~

def should_skip_output(
    input_video_file_path: Path,
    output_video_file_path: Path
    ) -> bool:

    if not output_video_file_path.is_file():

        return False;

    try:

        input_modified_time = input_video_file_path.stat().st_mtime;
        output_modified_time = output_video_file_path.stat().st_mtime;

    except OSError:

        return False;

    return output_modified_time >= input_modified_time;

# ~~

def get_model_weights_file_path(
    model_name: str
    ) -> str:

    model_weights_file_path = MODEL_FOLDER_PATH / f"{model_name}.pth";

    if not model_weights_file_path.is_file():

        print(
            f"Model weights not found: {model_weights_file_path}",
            file=sys.stderr
            );
        sys.exit( 1 );

    return str( model_weights_file_path );

# ~~

def is_old_arch_state_dict(
    state_dict: dict[ str, Any ]
    ) -> bool:

    return any( key.startswith( "model." ) for key in state_dict );

# ~~

def is_esrgan_arch_state_dict(
    state_dict: dict[ str, Any ]
    ) -> bool:

    return any( key.startswith( "RRDB_trunk." ) for key in state_dict );

# ~~

def convert_esrgan_arch_state_dict_to_rrdbnet_state_dict(
    esrgan_arch_state_dict: dict[ str, Any ]
    ) -> dict[ str, Any ]:

    rrdbnet_state_dict: dict[ str, Any ] = {};

    for key, value in esrgan_arch_state_dict.items():

        if key.startswith( "module." ):

            key = key[ 7: ];

        rrdbnet_key = (
            key
            .replace( "RRDB_trunk.", "body." )
            .replace( ".RDB1.", ".rdb1." )
            .replace( ".RDB2.", ".rdb2." )
            .replace( ".RDB3.", ".rdb3." )
            .replace( "trunk_conv.", "conv_body." )
            .replace( "upconv1.", "conv_up1." )
            .replace( "upconv2.", "conv_up2." )
            .replace( "HRconv.", "conv_hr." )
            );

        rrdbnet_state_dict[ rrdbnet_key ] = value;

    return rrdbnet_state_dict;

# ~~

def convert_old_arch_state_dict_to_rrdbnet_state_dict(
    old_arch_state_dict: dict[ str, Any ]
    ) -> dict[ str, Any ]:

    rrdbnet = (
        RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=MODEL_SCALE
            )
        );
    rrdbnet_state_dict = rrdbnet.state_dict();
    pretrained_state_dict: dict[ str, Any ] = {};

    for key, value in old_arch_state_dict.items():

        if key.startswith( "module." ):

            pretrained_state_dict[ key[ 7: ] ] = value;

        else:

            pretrained_state_dict[ key ] = value;

    remaining_key_set = set( rrdbnet_state_dict.keys() );

    for key, value in rrdbnet_state_dict.items():

        if (
            key in pretrained_state_dict
            and pretrained_state_dict[ key ].size() == value.size()
            ):

            rrdbnet_state_dict[ key ] = pretrained_state_dict[ key ];
            remaining_key_set.discard( key );

    rrdbnet_state_dict[ "conv_first.weight" ] = (
        pretrained_state_dict[ "model.0.weight" ]
        );
    rrdbnet_state_dict[ "conv_first.bias" ] = (
        pretrained_state_dict[ "model.0.bias" ]
        );

    for key in list( remaining_key_set ):

        if "rdb" not in key:

            continue;

        old_arch_key = (
            key.replace( "body.", "model.1.sub." )
            .replace( "rdb", "RDB" )
            );

        if key.endswith( ".weight" ):

            old_arch_key = old_arch_key.replace( ".weight", ".0.weight" );

        elif key.endswith( ".bias" ):

            old_arch_key = old_arch_key.replace( ".bias", ".0.bias" );

        rrdbnet_state_dict[ key ] = pretrained_state_dict[ old_arch_key ];
        remaining_key_set.discard( key );

    rrdbnet_state_dict[ "conv_body.weight" ] = (
        pretrained_state_dict[ "model.1.sub.23.weight" ]
        );
    rrdbnet_state_dict[ "conv_body.bias" ] = (
        pretrained_state_dict[ "model.1.sub.23.bias" ]
        );
    rrdbnet_state_dict[ "conv_up1.weight" ] = (
        pretrained_state_dict[ "model.3.weight" ]
        );
    rrdbnet_state_dict[ "conv_up1.bias" ] = (
        pretrained_state_dict[ "model.3.bias" ]
        );
    rrdbnet_state_dict[ "conv_up2.weight" ] = (
        pretrained_state_dict[ "model.6.weight" ]
        );
    rrdbnet_state_dict[ "conv_up2.bias" ] = (
        pretrained_state_dict[ "model.6.bias" ]
        );
    rrdbnet_state_dict[ "conv_hr.weight" ] = (
        pretrained_state_dict[ "model.8.weight" ]
        );
    rrdbnet_state_dict[ "conv_hr.bias" ] = (
        pretrained_state_dict[ "model.8.bias" ]
        );
    rrdbnet_state_dict[ "conv_last.weight" ] = (
        pretrained_state_dict[ "model.10.weight" ]
        );
    rrdbnet_state_dict[ "conv_last.bias" ] = (
        pretrained_state_dict[ "model.10.bias" ]
        );

    return rrdbnet_state_dict;

# ~~

def get_state_dict_from_model_checkpoint(
    model_weights_file_path: str
    ) -> dict[ str, Any ]:

    checkpoint = (
        torch.load(
            model_weights_file_path,
            map_location=torch.device( "cpu" )
            )
        );

    if isinstance( checkpoint, dict ):

        if "params_ema" in checkpoint:

            state_dict = checkpoint[ "params_ema" ];

        elif "params" in checkpoint:

            state_dict = checkpoint[ "params" ];

        else:

            state_dict = checkpoint;

    else:

        state_dict = checkpoint;

    if is_old_arch_state_dict( state_dict ):

        state_dict = (
            convert_old_arch_state_dict_to_rrdbnet_state_dict(
                state_dict
                )
            );

    elif is_esrgan_arch_state_dict( state_dict ):

        state_dict = (
            convert_esrgan_arch_state_dict_to_rrdbnet_state_dict(
                state_dict
                )
            );

    return state_dict;

# ~~

def get_model_block_count(
    model_name: str
    ) -> int:

    return MODEL_BLOCK_COUNT.get( model_name, DEFAULT_MODEL_BLOCK_COUNT );

# ~~

def get_model(
    model_name: str
    ) -> torch.nn.Module:

    if model_name in SRVGG_MODEL_NAME_SET:

        return (
            SRVGGNetCompact(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_conv=DEFAULT_SRVGG_CONV_COUNT,
                upscale=MODEL_SCALE,
                act_type="prelu"
                )
            );

    return (
        RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=get_model_block_count( model_name ),
            num_grow_ch=32,
            scale=MODEL_SCALE
            )
        );

# ~~

def is_rocm_pytorch(
    ) -> bool:

    return getattr( torch.version, "hip", None ) is not None;

# ~~

def is_nvidia_cuda_available(
    ) -> bool:

    return torch.cuda.is_available() and not is_rocm_pytorch();

# ~~

def is_amd_rocm_available(
    ) -> bool:

    return torch.cuda.is_available() and is_rocm_pytorch();

# ~~

def resolve_compute_backend(
    requested_compute_backend: str | None
    ) -> str:

    nvidia_cuda_is_available = is_nvidia_cuda_available();
    amd_rocm_is_available = is_amd_rocm_available();

    if requested_compute_backend == "cpu":

        return "cpu";

    if requested_compute_backend == "cuda":

        if not nvidia_cuda_is_available:

            print( "CUDA was requested but is not available.", file=sys.stderr );
            sys.exit( 1 );

        return "cuda";

    if requested_compute_backend == "rocm":

        if not amd_rocm_is_available:

            print( "ROCm was requested but is not available.", file=sys.stderr );
            sys.exit( 1 );

        return "rocm";

    if nvidia_cuda_is_available:

        return "cuda";

    if amd_rocm_is_available:

        return "rocm";

    return "cpu";

# ~~

def get_upsampler(
    model_name: str,
    model_weights_file_path: str,
    tile_size: int,
    compute_backend: str
    ) -> CrispRealESRGANer:

    if compute_backend == "cpu":

        print( "Using CPU (slow).", file=sys.stderr );
        device = torch.device( "cpu" );
        half = False;

    elif compute_backend == "cuda":

        print( "Using CUDA.", file=sys.stderr );
        device = torch.device( "cuda" );
        half = True;

    else:

        print( "Using ROCm.", file=sys.stderr );
        device = torch.device( "cuda" );
        half = True;

    return (
        CrispRealESRGANer(
            scale=MODEL_SCALE,
            model_path=model_weights_file_path,
            model=get_model( model_name ),
            tile=tile_size,
            tile_pad=10,
            pre_pad=0,
            half=half,
            device=device
            )
        );

# ~~

def get_crop_distance_in_pixels(
    crop_distance: float,
    image_dimension: int
    ) -> int:

    if crop_distance < 1:

        return int( round( image_dimension * crop_distance ) );

    return int( round( crop_distance ) );

# ~~

def get_cropped_dimensions(
    input_width: int,
    input_height: int,
    left_distance: float,
    right_distance: float,
    top_distance: float,
    bottom_distance: float
    ) -> tuple[ int, int ]:

    left_pixels = get_crop_distance_in_pixels( left_distance, input_width );
    right_pixels = get_crop_distance_in_pixels( right_distance, input_width );
    top_pixels = get_crop_distance_in_pixels( top_distance, input_height );
    bottom_pixels = get_crop_distance_in_pixels( bottom_distance, input_height );

    cropped_width = input_width - left_pixels - right_pixels;
    cropped_height = input_height - top_pixels - bottom_pixels;

    return cropped_width, cropped_height;

# ~~

def get_cropped_frame(
    video_frame: np.ndarray,
    left_distance: float,
    right_distance: float,
    top_distance: float,
    bottom_distance: float
    ) -> np.ndarray:

    height, width = video_frame.shape[ :2 ];
    left_pixels = get_crop_distance_in_pixels( left_distance, width );
    right_pixels = get_crop_distance_in_pixels( right_distance, width );
    top_pixels = get_crop_distance_in_pixels( top_distance, height );
    bottom_pixels = get_crop_distance_in_pixels( bottom_distance, height );

    return (
        video_frame[
            top_pixels:height - bottom_pixels,
            left_pixels:width - right_pixels
            ]
        );

# ~~

def get_output_dimensions(
    input_width: int,
    input_height: int,
    maximum_ratio: float,
    maximum_width: int,
    maximum_height: int
    ) -> tuple[ int, int ]:

    output_width = int( round( input_width * MODEL_SCALE ) );
    output_height = int( round( input_height * MODEL_SCALE ) );

    if maximum_ratio < MODEL_SCALE:

        output_width = int( round( input_width * maximum_ratio ) );
        output_height = int( round( input_height * maximum_ratio ) );

    if maximum_width == 0 and maximum_height == 0:

        return output_width, output_height;

    aspect_ratio = output_width / output_height;

    if maximum_width > 0 and output_width > maximum_width:

        output_width = maximum_width;
        output_height = int( round( output_width / aspect_ratio ) );

    if maximum_height > 0 and output_height > maximum_height:

        output_height = maximum_height;
        output_width = int( round( output_height * aspect_ratio ) );

    return output_width, output_height;

# ~~

def needs_upscaling(
    input_width: int,
    input_height: int,
    output_width: int,
    output_height: int
    ) -> bool:

    return (
        output_width > input_width
        or output_height > input_height
        );

# ~~

def get_downsized_frame(
    input_video_frame: np.ndarray,
    output_width: int,
    output_height: int
    ) -> np.ndarray:

    return cv2.resize(
        input_video_frame,
        ( output_width, output_height ),
        interpolation=cv2.INTER_LANCZOS4
        );

# ~~

def upscale_mp4(
    input_video_file_path: Path,
    output_video_file_path: Path,
    upsampler: CrispRealESRGANer | None,
    compression: int,
    left_crop_distance: float,
    right_crop_distance: float,
    top_crop_distance: float,
    bottom_crop_distance: float,
    maximum_ratio: float,
    maximum_width: int,
    maximum_height: int
    ) -> None:

    output_video_file_path.parent.mkdir( parents=True, exist_ok=True );

    video_reader_arguments = (
        argparse.Namespace(
            input_video_file_path=str( input_video_file_path.resolve() ),
            ffmpeg_binary="ffmpeg",
            frames_per_second=None,
            )
        );

    video_writer_arguments = (
        argparse.Namespace(
            ffmpeg_binary="ffmpeg",
            compression=compression
            )
        );

    print( f"Reading {input_video_file_path}" );

    video_reader = VideoReader( video_reader_arguments );
    input_height, input_width = video_reader.get_resolution();

    validate_crop_distances(
        input_width,
        input_height,
        left_crop_distance,
        right_crop_distance,
        top_crop_distance,
        bottom_crop_distance
        );

    cropped_width, cropped_height = (
        get_cropped_dimensions(
            input_width,
            input_height,
            left_crop_distance,
            right_crop_distance,
            top_crop_distance,
            bottom_crop_distance
            )
        );

    output_width, output_height = (
        get_output_dimensions(
            cropped_width,
            cropped_height,
            maximum_ratio,
            maximum_width,
            maximum_height
            )
        );

    is_upscaling_needed = (
        needs_upscaling(
            cropped_width,
            cropped_height,
            output_width,
            output_height
            )
        );

    is_downsize_needed = (
        is_upscaling_needed
        and (
            output_width != cropped_width * MODEL_SCALE
            or output_height != cropped_height * MODEL_SCALE
            )
        );

    output_display_aspect_ratio, output_sample_aspect_ratio = (
        get_output_aspect_ratios(
            output_width,
            output_height,
            input_width,
            input_height,
            video_reader.get_display_aspect_ratio(),
            video_reader.get_sample_aspect_ratio()
            )
        );

    video_writer = None;
    progress_bar = None;

    print( f"Writing {output_video_file_path}" );

    try:

        video_writer = (
            VideoWriter(
                video_writer_arguments,
                video_reader.get_audio_stream(),
                output_height,
                output_width,
                str( output_video_file_path ),
                video_reader.get_frames_per_second(),
                output_display_aspect_ratio,
                output_sample_aspect_ratio
                )
            );

        compute_device = (
            upsampler.device
            if upsampler is not None
            else None
            );

        progress_bar = (
            tqdm(
                total=len( video_reader ) or None,
                unit="frame",
                desc="crisp"
                )
            );

        while True:

            current_video_frame = video_reader.get_frame();

            if current_video_frame is None:

                break;

            current_video_frame = (
                get_cropped_frame(
                    current_video_frame,
                    left_crop_distance,
                    right_crop_distance,
                    top_crop_distance,
                    bottom_crop_distance
                    )
                );

            if not is_upscaling_needed:

                enhanced_video_frame = (
                    get_downsized_frame(
                        current_video_frame,
                        output_width,
                        output_height
                        )
                    );

            else:

                try:

                    enhanced_video_frame, _unused_scale = upsampler.enhance(
                        current_video_frame,
                        outscale=MODEL_SCALE
                        );

                except RuntimeError as runtime_error:

                    print( f"Error: {runtime_error}", file=sys.stderr );
                    print(
                        "Try again with tile size 400 (or a smaller value).",
                        file=sys.stderr
                        );

                    if output_video_file_path.is_file():

                        output_video_file_path.unlink();

                    sys.exit( 1 );

                if is_downsize_needed:

                    enhanced_video_frame = (
                        get_downsized_frame(
                            enhanced_video_frame,
                            output_width,
                            output_height
                            )
                        );

            video_writer.write_frame( enhanced_video_frame );

            if compute_device is not None and compute_device.type == "cuda":

                torch.cuda.synchronize( compute_device );

            progress_bar.update( 1 );

    finally:

        if progress_bar is not None:

            progress_bar.close();

        if video_writer is not None:

            video_writer.close();

        video_reader.close();

# ~~

def main(
    ) -> None:

    command_line_arguments = parse_arguments();
    validate_runtime();
    validate_input_video_file_path( command_line_arguments.input_video_file_path );
    validate_output_video_file_path( command_line_arguments.output_video_file_path );
    validate_maximum_ratio( command_line_arguments.max_ratio );

    if (
        command_line_arguments.skip
        and should_skip_output(
            command_line_arguments.input_video_file_path,
            command_line_arguments.output_video_file_path
            )
        ):

        print( f"Skipping {command_line_arguments.output_video_file_path}" );
        return;

    left_crop_distance, right_crop_distance, top_crop_distance, bottom_crop_distance = (
        command_line_arguments.crop
        if command_line_arguments.crop is not None
        else ( 0, 0, 0, 0 )
        );

    upsampler = None;

    if command_line_arguments.max_ratio > 1:

        compute_backend = (
            resolve_compute_backend( command_line_arguments.compute_backend )
            );

        upsampler = (
            get_upsampler(
                command_line_arguments.model,
                get_model_weights_file_path( command_line_arguments.model ),
                tile_size=command_line_arguments.tile_size,
                compute_backend=compute_backend
                )
            );

    upscale_mp4(
        command_line_arguments.input_video_file_path,
        command_line_arguments.output_video_file_path,
        upsampler,
        compression=command_line_arguments.compression,
        left_crop_distance=left_crop_distance,
        right_crop_distance=right_crop_distance,
        top_crop_distance=top_crop_distance,
        bottom_crop_distance=bottom_crop_distance,
        maximum_ratio=command_line_arguments.max_ratio,
        maximum_width=command_line_arguments.max_width,
        maximum_height=command_line_arguments.max_height
        );

# -- STATEMENTS

if __name__ == "__main__":

    main();
