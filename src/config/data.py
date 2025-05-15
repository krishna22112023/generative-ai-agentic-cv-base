import pyprojroot
root = pyprojroot.find_root(pyprojroot.has_dir("src"))

PATHS = {
    "data": f"{root}/data/",
    "raw": f"{root}/data/raw",
    "external": f"{root}/data/external",
    "artefacts": f"{root}/data/artefacts",
    "processed": f"{root}/data/processed",
    "annotated": f"{root}/data/annotated",
    "processed_final": f"{root}/data/processed_final",
}