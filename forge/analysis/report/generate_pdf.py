#!/usr/bin/env python3
"""Generate PDF from analysis report markdown files."""

import markdown
from weasyprint import HTML, CSS
from pathlib import Path

REPORT_DIR = Path(__file__).parent
RESULTS_DIR = REPORT_DIR.parent / "results"

# Order of sections
SECTIONS = [
    "00_executive_summary.md",
    "01_baseline.md",
    "02_information.md",
    "03_counts.md",
    "04_symmetry.md",
    "05_topology.md",
    "06_scaling.md",
    "07_synthesis.md",
]

# CSS styling for the PDF
CSS_STYLE = """
@page {
    size: letter;
    margin: 1in;
    @bottom-center {
        content: counter(page);
    }
}

body {
    font-family: 'Helvetica', 'Arial', sans-serif;
    font-size: 11pt;
    line-height: 1.5;
    color: #333;
}

h1 {
    color: #1a1a2e;
    font-size: 24pt;
    border-bottom: 2px solid #4a90d9;
    padding-bottom: 8pt;
    margin-top: 0;
    page-break-after: avoid;
}

h2 {
    color: #2d3436;
    font-size: 16pt;
    margin-top: 24pt;
    border-bottom: 1px solid #ddd;
    padding-bottom: 4pt;
    page-break-after: avoid;
}

h3 {
    color: #4a5568;
    font-size: 13pt;
    margin-top: 18pt;
    page-break-after: avoid;
}

p {
    margin: 8pt 0;
    text-align: justify;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 12pt 0;
    font-size: 10pt;
}

th, td {
    border: 1px solid #ddd;
    padding: 8pt;
    text-align: left;
}

th {
    background-color: #f5f5f5;
    font-weight: bold;
}

tr:nth-child(even) {
    background-color: #fafafa;
}

img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 16pt auto;
    border: 1px solid #eee;
}

code {
    background-color: #f4f4f4;
    padding: 2pt 4pt;
    border-radius: 3pt;
    font-family: 'Courier New', monospace;
    font-size: 10pt;
}

pre {
    background-color: #f4f4f4;
    padding: 12pt;
    border-radius: 4pt;
    overflow-x: auto;
    font-size: 9pt;
}

blockquote {
    border-left: 4px solid #4a90d9;
    padding-left: 16pt;
    margin-left: 0;
    color: #666;
    font-style: italic;
}

strong {
    color: #1a1a2e;
}

.section-break {
    page-break-before: always;
}

hr {
    border: none;
    border-top: 1px solid #ddd;
    margin: 24pt 0;
}

ul, ol {
    margin: 8pt 0;
    padding-left: 24pt;
}

li {
    margin: 4pt 0;
}
"""


def fix_image_paths(html_content: str) -> str:
    """Convert relative image paths to absolute paths."""
    # Replace ../results/figures/ with absolute path
    figures_dir = RESULTS_DIR / "figures"
    html_content = html_content.replace(
        'src="../results/figures/',
        f'src="file://{figures_dir}/'
    )
    return html_content


def main():
    print("Generating PDF from analysis report...")

    # Combine all markdown files
    combined_md = []

    for i, filename in enumerate(SECTIONS):
        filepath = REPORT_DIR / filename
        if not filepath.exists():
            print(f"  Warning: {filename} not found, skipping")
            continue

        print(f"  Processing {filename}...")
        content = filepath.read_text()

        # Add section break (except for first section)
        if i > 0:
            combined_md.append('<div class="section-break"></div>\n\n')

        combined_md.append(content)
        combined_md.append("\n\n")

    full_markdown = "".join(combined_md)

    # Convert markdown to HTML
    md = markdown.Markdown(extensions=['tables', 'fenced_code', 'toc'])
    html_body = md.convert(full_markdown)

    # Fix image paths
    html_body = fix_image_paths(html_body)

    # Wrap in full HTML document
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Texas 42 Oracle State Space Analysis</title>
</head>
<body>
{html_body}
</body>
</html>
"""

    # Generate PDF
    output_path = REPORT_DIR / "analysis_report_detailed.pdf"
    print(f"  Writing PDF to {output_path}...")

    HTML(string=html_content, base_url=str(REPORT_DIR)).write_pdf(
        output_path,
        stylesheets=[CSS(string=CSS_STYLE)]
    )

    print(f"  Done! PDF saved to {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
