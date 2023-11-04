from pathlib import Path
from typing import Iterable

import demucs.api
import demucs.audio
import fire
import librosa
import numpy as np
import tqdm


class Processor:
    def _get_paths(self, data_dir: Path) -> Iterable[Path]:
        paths = []
        data_dir: Path = Path(data_dir)
        for ext in ("wav", "mp3", "m4a", "flac", "ogg"):
            paths += list(data_dir.rglob(f"*.{ext}"))
        return paths

    def run(
        self,
        data_dir: str,
        out_dir: str | None = None,
        sample_rate: float = 16000,
        split: bool = False,
        jobs: int = 0,
    ):
        """Convert audo files to .npy format

        Args:
            data_dir (str): Directory of audio files to convert
            out_dir (str): Directory to output .npy files. Defaults to data_dir/npy or data_dir/npy_split if split is true.
            sample_rate (float, optional): Sample rate of .npy files. Defaults to 16000.
            split (bool, optional): Split files with demucs
            jobs (int, optional): Number of jobs for demucs to use. This can increase memory usage but will be much faster \
                when multiple cores are available.
        """
        # create npy dir
        data_dir: Path = Path(data_dir)
        npy_dir = out_dir
        if npy_dir is None:
            npy_dir = data_dir / ("npy" if not split else "npy_split")
        npy_dir = Path(npy_dir)
        npy_dir.mkdir(parents=True, exist_ok=True)

        # convert files to npy
        if split:
            separator = demucs.api.Separator(jobs=jobs)

        for audio_file in tqdm.tqdm(self._get_paths(data_dir)):
            npy_file = npy_dir / (audio_file.relative_to(data_dir).with_suffix(".npy"))
            if not npy_file.exists():
                try:
                    if split:
                        _, separated = separator.separate_audio_file(audio_file)
                        stems = [
                            demucs.audio.convert_audio(
                                wav,
                                separator._samplerate,
                                sample_rate,
                                1,  # for now, only 1 channel
                            )[0]
                            for _, wav in sorted(
                                separated.items(), key=lambda t: t[0]
                            )  # sort by stem type
                        ]
                        data = np.stack(stems)
                    else:
                        data, _ = librosa.core.load(audio_file, sr=sample_rate)

                    npy_file.parent.mkdir(parents=True, exist_ok=True)
                    with npy_file.open("wb") as f:
                        np.save(f, data)
                except Exception:
                    # some audio files are broken
                    print(f"Could not convert '{audio_file}'")
                    continue


if __name__ == "__main__":
    p = Processor()
    fire.Fire({"run": p.run})
