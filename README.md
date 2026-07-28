![](https://github.com/senselogic/CRISP/blob/master/LOGO/crisp.png)

# Crisp

GPU-accelerated AI video upscaler.

## Command line

```
crisp <input video file path> <output video file path> [<options>]
```

or

```
crisp_uv <input video file path> <output video file path> [<options>]
```

## Options

```
--model <model_name=realesrgan>
--crop <left_distance> <right_distance> <top_distance> <bottom_distance>
--max-ratio <maximum_ratio=4>
--max-width <maximum_width=0>
--max-height <maximum_height=0>
--tile-size <tile_size=0>
--compression <compression=22>
--skip
--cpu
--cuda
--rocm
```

If none of `--cpu`, `--cuda`, or `--rocm` is passed, Crisp uses CUDA when available, otherwise ROCm when available, otherwise CPU.

## Models

- `bsrgan` — real-world blind super-resolution
- `bsrnet` — smooth real-world blind super-resolution
- `highfidelity` — anime with high-frequency detail
- `realanime` — anime and manga
- `realdigital` — digital art
- `realesrgan` — sharp textures
- `realesrnet` — smooth output with minimal invented detail
- `remacri` — strong on skin, faces, and fine textures
- `ultramix` — balanced detail and smoothness
- `ultrasharp` — aggressive detail recovery

## Samples

```
crisp input_video.mp4 output_video.mp4
```

```
crisp input_video.mp4 output_video.mp4 --crop 0.2 0.2 0.1 0.1
```

```
crisp input_video.mp4 output_video.mp4 --crop 32 32 16 16 --max-ratio 2
```

```
crisp input_video.mp4 output_video.mp4 --model ultramix
```

```
crisp input_video.mp4 output_video.mp4 --compression 22
```

```
crisp input_video.mp4 output_video.mp4 --compression 22 --max-height 1080
```

```
crisp input_video.mp4 output_video.mp4 --compression 22 --tile-size 400
```

## Install

Run `install_ffmpeg.bat`, then one of:

- `install_packages_cuda.bat` / `install_uv_packages_cuda.bat` — NVIDIA CUDA
- `install_packages_rocm.bat` / `install_uv_packages_rocm.bat` — AMD ROCm
- `install_packages_cpu.bat` / `install_uv_packages_cpu.bat` — CPU only

## Dependencies

- Python 3.12.10
- CUDA 12.4 (NVIDIA) or ROCm 7.2.1 (AMD), optional
- ffmpeg (in the path)

## Limitations

- Only generates MP4 videos up to 4× the original resolution.
- Processes frames independently without enforcing temporal consistency.

## Version

0.1

## Author

Eric Pelzer (ecstatic.coder@gmail.com).

## License

This project is licensed under the GNU General Public License version 3.

See the [LICENSE.md](LICENSE.md) file for details.
