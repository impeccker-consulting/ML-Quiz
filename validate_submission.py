from pathlib import Path
import sys
import zipfile
import tempfile
import shutil


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

REQUIRED_FILES = {
    "results": "Results_Template.xlsx",
    "summary": "Executive_Summary.pdf",
}

ALLOWED_ANALYSIS_EXTENSIONS = {
    ".ipynb": "Python",
    ".rmd": "R",
    ".r": "R",
}


# ---------------------------------------------------------
# Validation function
# ---------------------------------------------------------

def validate_submission(zip_path):

    zip_path = Path(zip_path)

    print()
    print("=" * 60)
    print("ML ASSIGNMENT SUBMISSION VALIDATOR")
    print("=" * 60)

    if not zip_path.exists():
        print()
        print("ERROR: ZIP file not found.")
        print(f"Path: {zip_path}")
        return False

    if zip_path.suffix.lower() != ".zip":
        print()
        print("ERROR: Submission must be a ZIP file.")
        return False

    # -----------------------------------------------------
    # Temporary extraction directory
    # -----------------------------------------------------

    temp_dir = Path(
        tempfile.mkdtemp(prefix="ml_submission_")
    )

    try:

        # -------------------------------------------------
        # Extract ZIP
        # -------------------------------------------------

        with zipfile.ZipFile(zip_path, "r") as z:

            # Basic ZIP safety check
            for member in z.infolist():

                target = (temp_dir / member.filename).resolve()

                if not str(target).startswith(
                    str(temp_dir.resolve())
                ):
                    print()
                    print("ERROR: Unsafe ZIP structure detected.")
                    return False

            z.extractall(temp_dir)

        # -------------------------------------------------
        # Find all files
        # -------------------------------------------------

        files = [
            p
            for p in temp_dir.rglob("*")
            if p.is_file()
        ]

        print()
        print(f"Files found: {len(files)}")

        for file in files:
            print(f"  - {file.relative_to(temp_dir)}")

        # -------------------------------------------------
        # Check required files
        # -------------------------------------------------

        filenames = {
            p.name.lower(): p
            for p in files
        }

        results_file = filenames.get(
            REQUIRED_FILES["results"].lower()
        )

        summary_file = filenames.get(
            REQUIRED_FILES["summary"].lower()
        )

        # -------------------------------------------------
        # Find analysis file
        # -------------------------------------------------

        analysis_files = []

        for file in files:

            if file.suffix.lower() in ALLOWED_ANALYSIS_EXTENSIONS:
                analysis_files.append(file)

        print()
        print("-" * 60)
        print("VALIDATION RESULTS")
        print("-" * 60)

        valid = True

        # Results template
        if results_file:
            print("Results Template       ✓")
        else:
            print("Results Template       ✗  Missing")
            valid = False

        # Analysis
        if len(analysis_files) == 1:

            analysis = analysis_files[0]

            language = ALLOWED_ANALYSIS_EXTENSIONS[
                analysis.suffix.lower()
            ]

            print(
                f"Analysis File          ✓  ({language})"
            )

        elif len(analysis_files) == 0:

            print("Analysis File          ✗  Missing")
            valid = False

        else:

            print(
                "Analysis File          ✗  Multiple analysis files found"
            )

            for file in analysis_files:
                print(
                    f"                         {file.name}"
                )

            valid = False

        # Summary
        if summary_file:
            print("Executive Summary      ✓")
        else:
            print("Executive Summary      ✗  Missing")
            valid = False

        # -------------------------------------------------
        # Final result
        # -------------------------------------------------

        print()
        print("=" * 60)

        if valid:

            print("STATUS: READY FOR EVALUATION")

        else:

            print("STATUS: INCOMPLETE SUBMISSION")

        print("=" * 60)
        print()

        return valid

    finally:

        # -------------------------------------------------
        # Clean up temporary extraction
        # -------------------------------------------------

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )


# ---------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print()
        print(
            "Usage:"
        )
        print(
            "python validate_submission.py <submission.zip>"
        )
        print()

        sys.exit(1)

    submission_zip = sys.argv[1]

    success = validate_submission(
        submission_zip
    )

    sys.exit(
        0 if success else 1
    )