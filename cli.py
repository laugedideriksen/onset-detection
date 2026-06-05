import argparse
import sys
from onset_detection import OnsetDetect

def run_detect(args):
    """Wrapper for detect_onsets method."""
    try:
        audio = OnsetDetect(
                sound_file=args.sound_file,
                title=args.title,
                start=args.start,
                end=args.end
                )

        do_plot = not args.no_plot

        audio.detect_onsets(
                envelope=args.envelope,
                hybrid_env_components=args.components,
                filtering=args.filtering,
                filter_kernel=args.filter_kernel,
                threshold_type=args.threshold_type,
                threshold_k=args.threshold_k,
                threshold_window=args.threshold_window,
                peak_picking=args.peak_picking,
                merge_onsets=args.merge_onsets,
                min_note_gap=args.min_note_gap,
                output=args.output,
                output_destination=args.destination,
                plot=do_plot
                )

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

def run_env_comp(args):
    """Wrapper for detect_onsets method."""
    try:
        audio = OnsetDetect(
                sound_file=args.sound_file,
                title=args.title,
                start=args.start,
                end=args.end
                )
        audio.compare(compare_parameter="envelopes", threshold_k=args.threshold_k, threshold_window=args.threshold_window)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

def run_filter_comp(args):
    """Wrapper for detect_onsets method."""
    try:
        audio = OnsetDetect(
                sound_file=args.sound_file,
                title=args.title,
                start=args.start,
                end=args.end
                )
        audio.compare(compare_parameter="filtering", filter_kernel=args.filter_kernel)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
            description="A moderately over-engineered onset-detection script.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
            Examples:
                # Basic usage with default settings
                python cli.py path-to-audio-file

                # Specify parameters and plot result
                python cli.py path-to-audio-file --envelope hybrid --threshold-type moving --threshold-k 1.0 --plot

                # Output result as csv
                python cli.py path-to-audio-file --output csv --destination results.csv
            """
            )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ONSET DETECTION
    od_parser = subparsers.add_parser("detect-onsets", help="Detect onsets in audio file, output results, and optionally plot spectrogram envelope, and onsets.")

    # Optional arguments for parameters.

    # Load audio
    od_parser.add_argument(
            "sound_file",
            type=str,
            help="Path to the input audio file. All codecs supported by 'soundfile' are also supported here."
            )
    od_parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds (default: 0.0).")
    od_parser.add_argument("--end", type=float, default=None, help="End time in seconds (default: full file).")

    # File Information
    od_parser.add_argument("--title", type=str, default="", help="Title of recording to be displayed in plot/output.")

    # Envelope
    od_parser.add_argument(
        "--envelope", 
        type=str, 
        default="spectral_flux",
        choices=["spectral_flux", "diff_spectral_flux", "diff_rms", "delta_rms", "chroma_cqt", "hybrid"],
        help="Envelope type (default: spectral_flux). The components of a 'hybrid' envelope can be specified using the --components flag."
    )
    od_parser.add_argument(
        "--components", 
        type=str, 
        nargs="+", 
        default=["spectral_flux", "delta_rms"],
        help="List of components for hybrid envelope (default: spectral_flux delta_rms)."
    )

    # Filtering
    od_parser.add_argument(
        "--filtering", 
        type=str, 
        default="median_filter",
        choices=["none", "median_filter"],
        help="Smoothing method (default: median_filter)."
    )
    od_parser.add_argument(
        "--filter-kernel", 
        type=int, 
        default=3,
        help="Kernel size for median filter (must be odd, default: 3)."
    )

    # Threshold
    od_parser.add_argument(
        "--threshold-type", 
        type=str, 
        default="global",
        choices=["global", "moving"],
        help="Threshold type (default: global)."
    )
    od_parser.add_argument(
        "--threshold-k", 
        type=float, 
        default=2.0,
        help="Sensitivity factor (k). Lower value = more sensitive onset detection (default: 0.9)."
    )
    od_parser.add_argument(
            "--threshold-window",
            type=float,
            default=2.0,
            help="Size of window for moving threshold in seconds (default: 1.0)"
            )

    # Peak picking
    od_parser.add_argument(
        "--peak-picking", 
        type=str, 
        default="backtrack",
        choices=["centroid", "backtrack", "librosa"],
        help="Peak picking algorithm (default: backtrack)."
    )

    # Merging
    od_parser.add_argument(
        "--merge-onsets", 
        action="store_true",
        help="Merge detected onsets that are very close. Set minimum note gap using the --min-note-gap flag. This option is usually best left disabled."
    )
    od_parser.add_argument(
        "--min-note-gap", 
        type=float, 
        default=0.08,
        help="Minimum gap in seconds(default: 0.08)."
    )

    # Output
    od_parser.add_argument(
        "--output", 
        type=str, 
        default="list",
        choices=["list", "rows", "csv"],
        help="Output format (default: list)."
    )
    od_parser.add_argument(
        "--destination", 
        type=str, 
        default=None,
        help="Destination filename (for CSV) or base name."
    )
    od_parser.add_argument(
        "--no-plot", 
        action="store_true",
        help="Disable visualisation."
    )

    od_parser.set_defaults(func=run_detect)


    # ENVELOPE COMPARISON
    ec_parser = subparsers.add_parser("compare-envelopes", help="Detect onsets in audio file, output results, and optionally plot spectrogram envelope, and onsets.")

    # Optional arguments for parameters.

    # Load audio
    ec_parser.add_argument(
            "sound_file",
            type=str,
            help="Path to the input audio file. All codecs supported by 'soundfile' are also supported here."
            )
    ec_parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds (default: 0.0).")
    ec_parser.add_argument("--end", type=float, default=None, help="End time in seconds (default: full file).")

    # File Information
    ec_parser.add_argument("--title", type=str, default="", help="Title of recording to be displayed in plot/output.")

    # Threshold
    ec_parser.add_argument(
         "--threshold-k", 
         type=float, 
         default=2.0,
         help="Sensitivity factor (k). Lower value = more sensitive onset detection (default: 0.9)."
         )
    ec_parser.add_argument(
            "--threshold-window",
            type=float,
            default=2.0,
            help="Size of window for moving threshold in seconds (default: 1.0)"
            )

    ec_parser.set_defaults(func=run_env_comp)

    
    # FILTER COMPARISON
    fc_parser = subparsers.add_parser("compare-filtering", help="Detect onsets in audio file, output results, and optionally plot spectrogram envelope, and onsets.")

    # Optional arguments for parameters.

    # Load audio
    fc_parser.add_argument(
            "sound_file",
            type=str,
            help="Path to the input audio file. All codecs supported by 'soundfile' are also supported here."
            )
    fc_parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds (default: 0.0).")
    fc_parser.add_argument("--end", type=float, default=None, help="End time in seconds (default: full file).")

    # File Information
    fc_parser.add_argument("--title", type=str, default="", help="Title of recording to be displayed in plot/output.")

    # Filtering
    fc_parser.add_argument(
            "--filter-kernel", 
            type=int, 
            default=3,
            help="Kernel size for median filter (must be odd, default: 3)."
            )
    
    fc_parser.set_defaults(func=run_filter_comp)

    # Parse arguments and execute command
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
