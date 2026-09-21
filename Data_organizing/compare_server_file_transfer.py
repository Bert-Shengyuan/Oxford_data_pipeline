"""
Compare the "proc" file tree staged on server 1 (Windows staging box, logged as a
flat "path|size" text file) against the same tree mirrored on server 2 (Linux
cephfs cluster, logged as a "size<TAB>path" manifest), and report which files
did not make it across intact.

A file only counts as successfully transferred if BOTH the relative path exists
on server 2 AND the size matches exactly - a same-named file with a different
size (e.g. a truncated/placeholder copy) is treated as missing, since only the
name (not the content) made it over.

Outputs four tab-separated tables (Excel-readable) into the same directory as
this script:
  1. all_missing_or_mismatched_files.tsv   - every proc/ file on server 1 that is
                                              absent or size-mismatched on server 2
  2. raw_mp4_behaviour_reference_sessions.tsv - subset of (1): *.mp4 files whose
                                              session is in the reference session list
  3. proc_mdl_mat_reference_sessions.tsv   - subset of (1): *.mdl.mat files whose
                                              session is in the reference session list
  4. cell_metrics_cellinfo_reference_sessions.tsv - subset of (1): *.cell_metrics.cellinfo.mat
                                              files whose session is in the reference session list
"""

import csv
import re
from pathlib import Path

SERVER1_LOG = Path("/Users/shengyuancai/Downloads/log_Oxford_proc_withsize.txt")
SERVER2_MANIFEST = Path("/Users/shengyuancai/Downloads/Temperal_stuff/yp_oxford_file_manifest.tsv")

SERVER1_PROC_ROOT = "\\Staging\\Yangfan\\proc\\"
SERVER2_PROC_ROOT = "/data/cephfs-2/unmirrored/groups/peng/YP_Oxford/proc/"

OUT_DIR = Path(__file__).resolve().parent

SESSION_LIST = {
    ("yp010", "220209"), ("yp010", "220210"), ("yp010", "220211"), ("yp010", "220212"),
    ("yp012", "220208"), ("yp012", "220209"), ("yp012", "220210"), ("yp012", "220211"), ("yp012", "220212"),
    ("yp013", "220209"), ("yp013", "220210"), ("yp013", "220211"), ("yp013", "220212"),
    ("yp014", "220208"), ("yp014", "220209"), ("yp014", "220210"), ("yp014", "220211"), ("yp014", "220212"),
    ("yp020", "220331"), ("yp020", "220401"), ("yp020", "220402"), ("yp020", "220403"),
    ("yp020", "220404"), ("yp020", "220405"), ("yp020", "220407"),
    ("yp021", "220331"), ("yp021", "220401"), ("yp021", "220402"), ("yp021", "220403"),
    ("yp021", "220404"), ("yp021", "220405"), ("yp021", "220407"),
    ("yp022", "220401"), ("yp022", "220402"), ("yp022", "220403"),
    ("yp022", "220404"), ("yp022", "220405"), ("yp022", "220407"),
}

SESSION_RE = re.compile(r"(yp\d{3})_(\d{6})")


def load_server1(path):
    files = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            win_path, _, size_str = line.rpartition("|")
            if not win_path:
                raise ValueError(f"{path}:{line_no}: could not split path|size in {line!r}")
            if not win_path.startswith(SERVER1_PROC_ROOT):
                raise ValueError(f"{path}:{line_no}: unexpected root in {win_path!r}")
            rel = win_path[len(SERVER1_PROC_ROOT):].replace("\\", "/")
            files[rel] = int(size_str)
    return files


def load_server2_proc(path):
    files = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            size_str, _, full_path = line.partition("\t")
            if not full_path.startswith(SERVER2_PROC_ROOT):
                continue
            rel = full_path[len(SERVER2_PROC_ROOT):]
            files[rel] = int(size_str)
    return files


def extract_session(rel_path):
    m = SESSION_RE.search(rel_path)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def write_tsv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


def main():
    server1 = load_server1(SERVER1_LOG)
    server2 = load_server2_proc(SERVER2_MANIFEST)
    print(f"server1 (proc/) files: {len(server1)}")
    print(f"server2 (proc/) files: {len(server2)}")

    header = [
        "relative_path", "animal", "session_date", "in_reference_session_list",
        "server1_size_bytes", "server2_size_bytes", "status",
    ]
    all_rows = []
    for rel, size1 in sorted(server1.items()):
        size2 = server2.get(rel)
        if size2 is None:
            status = "MISSING_ON_SERVER2"
        elif size2 != size1:
            status = "SIZE_MISMATCH"
        else:
            continue  # transferred correctly, not a diff

        animal, date = extract_session(rel)
        in_ref = (animal, date) in SESSION_LIST
        all_rows.append([
            rel, animal or "", date or "", in_ref,
            size1, size2 if size2 is not None else "",
            status,
        ])

    write_tsv(OUT_DIR / "all_missing_or_mismatched_files.tsv", header, all_rows)
    print(f"table 1 (all missing/mismatched): {len(all_rows)} rows")

    def reference_subset(suffix):
        return [r for r in all_rows if r[3] and r[0].lower().endswith(suffix)]

    mp4_rows = reference_subset(".mp4")
    write_tsv(OUT_DIR / "raw_mp4_behaviour_reference_sessions.tsv", header, mp4_rows)
    print(f"table 2 (mp4 behaviour, reference sessions): {len(mp4_rows)} rows")

    mdl_rows = reference_subset(".mdl.mat")
    write_tsv(OUT_DIR / "proc_mdl_mat_reference_sessions.tsv", header, mdl_rows)
    print(f"table 3 (mdl.mat, reference sessions): {len(mdl_rows)} rows")

    cell_metrics_rows = reference_subset(".cell_metrics.cellinfo.mat")
    write_tsv(OUT_DIR / "cell_metrics_cellinfo_reference_sessions.tsv", header, cell_metrics_rows)
    print(f"table 4 (cell_metrics.cellinfo.mat, reference sessions): {len(cell_metrics_rows)} rows")


if __name__ == "__main__":
    main()
