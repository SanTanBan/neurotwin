"""Render docs/REPORT.md to a print-ready PDF with headless Chrome (or Edge).

    python tools/report_pdf.py [--out docs/NeuroTwin_report.pdf] [--work DIR]

The Markdown is converted with python-markdown; each image followed by an italic caption becomes a figure that
is never split across pages. The browser runs headless with its own profile in --work, so it never touches a
running browser session.
"""
import argparse
import os
import re
import shutil
import subprocess
import tempfile

import markdown

BROWSERS = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            "google-chrome", "chromium", "chromium-browser"]

CSS = """
@page { size: A4; margin: 16mm 15mm 18mm 15mm;
        @bottom-center { content: counter(page) " / " counter(pages); font: 8.5pt 'Segoe UI', Arial, sans-serif;
                         color: #666; } }
body { font: 10pt/1.42 'Segoe UI', Calibri, Arial, sans-serif; color: #1a1a1a; }
h1 { font-size: 19pt; line-height: 1.2; margin: 0 0 6pt; }
h2 { font-size: 14pt; margin: 16pt 0 5pt; border-bottom: 1px solid #ccc; padding-bottom: 2pt;
     break-after: avoid; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt; break-after: avoid; }
p { margin: 4pt 0 6pt; text-align: justify; hyphens: auto; }
blockquote { margin: 6pt 0; padding: 4pt 10pt; border-left: 3px solid #bbb; color: #444; background: #f6f6f6; }
table { border-collapse: collapse; margin: 6pt auto 9pt; font-size: 8.4pt; break-inside: avoid; }
th, td { border: 1px solid #c8c8c8; padding: 2.5pt 5pt; vertical-align: top; }
th { background: #eef2f6; }
code { font: 8.6pt Consolas, 'Courier New', monospace; background: #f3f3f3; padding: 0 2pt; }
pre { background: #f3f3f3; padding: 6pt 8pt; font-size: 8.4pt; break-inside: avoid; white-space: pre-wrap; }
pre code { padding: 0; }
figure { margin: 8pt 0 10pt; break-inside: avoid; text-align: center; }
figure img { max-width: 100%; max-height: 105mm; }
figcaption { font-size: 8.6pt; color: #333; text-align: left; margin-top: 3pt; }
ul, ol { margin: 3pt 0 6pt 16pt; padding-left: 8pt; }
li { margin: 1.5pt 0; }
a { color: #1f5fa8; text-decoration: none; }
"""


def to_html(md_text, title):
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists", "attr_list"])
    # <p><img ...>\n<em>Figure N. ...</em></p>  ->  <figure><img ...><figcaption>...</figcaption></figure>
    body = re.sub(r"<p>(<img [^>]+>)\s*<em>(.*?)</em></p>", r"<figure>\1<figcaption>\2</figcaption></figure>",
                  body, flags=re.S)
    body = re.sub(r"<p>(<img [^>]+>)</p>", r"<figure>\1</figure>", body)
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{title}</title>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="docs/REPORT.md")
    ap.add_argument("--out", default="docs/NeuroTwin_report.pdf")
    ap.add_argument("--work", default=None, help="folder for the browser profile and temporary files")
    a = ap.parse_args()
    browser = next((b for b in BROWSERS if os.path.isfile(b) or shutil.which(b)), None)
    if browser is None:
        raise SystemExit("no Chrome/Edge/Chromium found")
    work = a.work or tempfile.mkdtemp(prefix="report_pdf_")
    os.makedirs(work, exist_ok=True)
    src_dir = os.path.dirname(os.path.abspath(a.src))
    html_path = os.path.join(src_dir, "_report_print.html")      # next to the .md so figure paths resolve
    md_text = open(a.src, encoding="utf-8").read()
    title = md_text.splitlines()[0].lstrip("# ").strip()
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(to_html(md_text, title))
    env = dict(os.environ, TEMP=work, TMP=work)
    out = os.path.abspath(a.out)
    try:
        subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
                        "--no-pdf-header-footer", f"--user-data-dir={os.path.join(work, 'profile')}",
                        f"--print-to-pdf={out}", "file:///" + html_path.replace(os.sep, "/")],
                       check=True, env=env, timeout=180, capture_output=True)
    finally:
        os.remove(html_path)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
