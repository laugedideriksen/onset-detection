import argparse
import sys
from onset_detection import OnsetDetect

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

    parser.add_argument(
            "sound_file",
            type=str,
            help="Path to the input audio file. All codecs supported by 'soundfile' are also supported here."
            )

    # Optional arguments for parameters.

    # Loading audio
    parser.add_argument("--title", type=str, default="", help="Title of recording to be displayed in plot/output.")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds (default: 0.0).")
    parser.add_argument("--end", type=float, default=None, help="End time in seconds (default: full file).")

    # Envelope
    parser.add_argument(
        "--envelope", 
        type=str, 
        default="spectral_flux",
        choices=["spectral_flux", "diff_spectral_flux", "diff_rms", "delta_rms", "chroma_cqt", "hybrid"],
        help="Envelope type (default: spectral_flux). The components of a 'hybrid' envelope can be specified using the --components flag."
    )
    parser.add_argument(
        "--components", 
        type=str, 
        nargs="+", 
        default=["spectral_flux", "delta_rms"],
        help="List of components for hybrid envelope (default: spectral_flux delta_rms)."
    )

    # Filtering
    # Filtering
    parser.add_argument(
        "--filtering", 
        type=str, 
        default="median_filter",
        choices=["none", "median_filter"],
        help="Smoothing method (default: median_filter)."
    )
    parser.add_argument(
        "--filter-kernel", 
        type=int, 
        default=3,
        help="Kernel size for median filter (must be odd, default: 3)."
    )

    # Threshold
    parser.add_argument(
        "--threshold-type", 
        type=str, 
        default="global",
        choices=["global", "moving"],
        help="Threshold type (default: global)."
    )
    parser.add_argument(
        "--threshold-k", 
        type=float, 
        default=2.0,
        help="Sensitivity factor (k). Lower value = more sensitive onset detection (default: 2.0)."
    )

    # Peak picking
    parser.add_argument(
        "--peak-picking", 
        type=str, 
        default="backtrack",
        choices=["centroid", "backtrack", "librosa"],
        help="Peak picking algorithm (default: backtrack)."
    )

    # Merging
    parser.add_argument(
        "--merge-onsets", 
        action="store_true",
        help="Merge detected onsets that are very close. Set minimum note gap using the --min-note-gap flag. This option is usually best left disabled."
    )
    parser.add_argument(
        "--min-note-gap", 
        type=float, 
        default=0.08,
        help="Minimum gap in seconds(default: 0.08)."
    )

    # Output
    parser.add_argument(
        "--output", 
        type=str, 
        default="list",
        choices=["list", "rows", "csv"],
        help="Output format (default: list)."
    )
    parser.add_argument(
        "--destination", 
        type=str, 
        default=None,
        help="Destination filename (for CSV) or base name."
    )
    parser.add_argument(
        "--no-plot", 
        action="store_true",
        help="Disable visualisation."
    )


    args = parser.parse_args()

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
            peak_picking=args.peak_picking,
            merge_onsets=args.merge_onsets,
            min_note_gap=args.min_note_gap,
            output=args.output,
            output_destination=args.dest,
            plot=do_plot
        )

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
