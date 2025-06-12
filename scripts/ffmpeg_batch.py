import argparse
import csv
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path


class FFmpegBatchProcessor:
    """
    Batch processor for applying FFmpeg filters to .mp4 files in a directory structure.

    Supports filter specification via:
    - Global command-line mode string (e.g., scale,fps10)
    - Filename suffixes (e.g., __scale, __gray, __fps10)

    Features:
    - Folder structure preservation
    - Output file renaming with applied filter suffixes
    - Retry logic for FFmpeg failures
    - CSV logging with timestamped filenames
    - Dry-run mode for previewing changes
    """

    def __init__(
        self, input_dir: Path, output_dir: Path, retries: int = 2, dry_run: bool = False, mode: str | None = None
    ):
        """
        Initialize the batch processor.

        Args:
            input_dir (Path): Root input directory containing .mp4 files.
            output_dir (Path): Output directory where processed files will be written.
            retries (int): Number of retry attempts on FFmpeg failure.
            dry_run (bool): If True, only preview operations without running FFmpeg.
            mode (Optional[str]): Global filter mode (e.g., "scale,fps10,gray").
        """
        self.input_dir = input_dir.resolve()
        self.output_dir = output_dir.resolve()
        self.retries = retries
        self.dry_run = dry_run
        self.global_filters = self._parse_mode_string(mode) if mode else None
        self.log_fp = None
        self.logger = None

        if not self.dry_run:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.log_file = self.output_dir / f"ffmpeg_log_{date_str}.csv"
            self.log_fp = open(self.log_file, "w", newline="")
            self.logger = csv.writer(self.log_fp)
            self.logger.writerow(
                ["relative_path", "original_MB", "resized_MB", "status", "attempts", "applied_filters"]
            )

    def _parse_mode_string(self, mode_str: str) -> list[str]:
        """
        Parse a comma-separated global filter mode string into FFmpeg filter expressions.

        Args:
            mode_str (str): e.g., "scale,fps10,gray"

        Returns:
            list[str]: FFmpeg-compatible filter expressions.
        """
        filters = []
        for token in mode_str.split(","):
            token = token.strip().lower()
            if token == "scale":
                filters.append("scale=iw/2:ih/2")
            elif token == "gray":
                filters.append("format=gray")
            elif token.startswith("fps"):
                value = token[3:]
                if value.isdigit():
                    filters.append(f"fps={value}")
        return filters

    def parse_filters_from_filename(self, filename: str) -> list[str]:
        """
        Parse filters from a file name using suffix patterns like '__scale', '__fps10', '__gray'.

        Args:
            filename (str): Input filename.

        Returns:
            list[str]: FFmpeg-compatible filter expressions.
        """
        filters = []
        lowered = filename.lower()
        if "__scale" in lowered:
            filters.append("scale=iw/2:ih/2")
        if "__gray" in lowered:
            filters.append("format=gray")
        if "__fps" in lowered:
            match = re.search(r"__fps(\d+)", lowered)
            if match:
                filters.append(f"fps={match.group(1)}")
        return filters

    def filters_to_suffixes(self, filters: list[str]) -> str:
        """
        Convert FFmpeg filter expressions into filename suffixes.

        Args:
            filters (list[str]): List of applied filters.

        Returns:
            str: Suffix string to append to filename (e.g., '__scale__fps10').
        """
        suffixes = []
        for f in filters:
            if f.startswith("scale="):
                suffixes.append("scale")
            elif f.startswith("fps="):
                suffixes.append(f"fps{f.split('=')[1]}")
            elif f == "format=gray":
                suffixes.append("gray")
        return "__" + "__".join(suffixes) if suffixes else ""

    def run_ffmpeg(self, input_path: Path, output_path: Path, filters: list[str]) -> tuple[bool, int]:
        """
        Run FFmpeg to process a single file with the given filters.

        Args:
            input_path (Path): Input .mp4 file path.
            output_path (Path): Destination output path.
            filters (list[str]): List of FFmpeg filters.

        Returns:
            tuple[bool, int]: (success flag, number of attempts made)
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        attempt = 0
        success = False
        vf_chain = ",".join(filters)

        cmd = ["ffmpeg", "-i", str(input_path), "-vf", vf_chain, "-c:a", "copy", "-y", str(output_path)]

        while attempt <= self.retries:
            attempt += 1
            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                success = True
                break
            except subprocess.CalledProcessError:
                time.sleep(0.5 * attempt)
        return success, attempt

    def process_all(self):
        """
        Process all .mp4 files in the input path (recursively if directory) using assigned filters.

        - If dry_run is enabled, prints preview of each operation.
        - If not, performs FFmpeg processing and logs results.
        """
        if self.input_dir.is_dir():
            files_to_process = list(self.input_dir.rglob("*.mp4"))
            base_dir = self.input_dir
        elif self.input_dir.is_file() and self.input_dir.suffix.lower() == ".mp4":
            files_to_process = [self.input_dir]
            base_dir = self.input_dir.parent
        else:
            print(f"Error: Input path is not a valid directory or .mp4 file: {self.input_dir}")
            return

        for file in files_to_process:
            rel_path = file.relative_to(base_dir)

            # Determine filters early
            filters = self.global_filters if self.global_filters else self.parse_filters_from_filename(file.name)
            if not filters:
                if self.dry_run:
                    print(f"[SKIP] No filters matched: {rel_path}")
                continue

            # Generate output path with renamed file
            filter_suffix = self.filters_to_suffixes(filters)
            stem = file.stem
            suffix = file.suffix
            new_filename = stem + filter_suffix + suffix
            output_path = self.output_dir / rel_path.parent / new_filename

            if self.dry_run:
                print(f"[DRY RUN] Would process: {rel_path}")
                print(f"           Filters: {filters}")
                print(f"           Output : {output_path.relative_to(self.output_dir)}")
                continue

            original_size = file.stat().st_size / (1024 * 1024)
            success, attempts = self.run_ffmpeg(file, output_path, filters)
            resized_size = output_path.stat().st_size / (1024 * 1024) if success else 0

            if self.logger:
                self.logger.writerow(
                    [
                        str(rel_path),
                        f"{original_size:.2f}",
                        f"{resized_size:.2f}",
                        "Success" if success else "Failed",
                        attempts,
                        "; ".join(filters),
                    ]
                )

    def close(self):
        """
        Close the log file handle if open.
        """
        if self.log_fp:
            self.log_fp.close()


def main():
    """
    Entry point for command-line interface.
    Parses arguments and launches the batch processor.
    """
    parser = argparse.ArgumentParser(description="FFmpeg Batch Processor by Daisy")
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Input directory (recursive) or a single .mp4 file"
    )
    parser.add_argument("-o", "--output", type=str, required=True, help="Output directory (structure preserved)")
    parser.add_argument("--mode", type=str, help="Filters to apply globally (e.g., 'scale,fps10,gray')")
    parser.add_argument("--retries", type=int, default=2, help="Retries on failure (default: 2)")
    parser.add_argument("--dry-run", action="store_true", help="Preview actions without executing FFmpeg")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        return

    processor = FFmpegBatchProcessor(
        input_path, output_path, retries=args.retries, dry_run=args.dry_run, mode=args.mode
    )

    try:
        processor.process_all()
    finally:
        processor.close()


if __name__ == "__main__":
    main()
