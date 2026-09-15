#!/usr/bin/env python3
"""
Entry point for rendering any step of the residualize()/pcca() walkthrough --
handles the file/scene-name pairing and always runs manim from this
directory, so `data_prep`/`theme`'s same-directory imports resolve
regardless of where you invoke this script from.

    python run_step.py 1              # preview, fast low-res (default -pql)
    python run_step.py 3 -qh          # step 3, rendered at 1080p60
    python run_step.py 7 -qh --format=mp4

Run with this project's own virtualenv (create once with the sibling
requirements.txt: `python3 -m venv .venv && .venv/bin/pip install -r
requirements.txt`), either activated or invoked directly:

    .venv/bin/python run_step.py 2
"""

import subprocess
import sys
from pathlib import Path

STEPS = {
    1: ("step1_single_trial_residual.py", "Step1SingleTrialResidual"),
    2: ("step2_ten_trial_scroll.py", "Step2TenTrialScroll"),
    3: ("step3_region_b_to_ten.py", "Step3RegionBToTen"),
    4: ("step4_region_b_to_full.py", "Step4RegionBToFull"),
    5: ("step5_region_a_to_three.py", "Step5RegionAToThree"),
    6: ("step6_region_a_to_full_matrix.py", "Step6RegionAToFullMatrix"),
    7: ("step7_cca_latent_space.py", "Step7CCALatentSpace"),
    8: ("step8_cca_process_video.py", "Step8CCAProcessVideo"),
}


def main() -> None:
    if len(sys.argv) < 2 or not sys.argv[1].isdigit() or int(sys.argv[1]) not in STEPS:
        names = "\n".join(f"  {n}: {f}" for n, (f, _) in STEPS.items())
        print(f"usage: python run_step.py <1-8> [extra manim flags, default -pql]\n\nsteps:\n{names}")
        raise SystemExit(1)

    file, scene = STEPS[int(sys.argv[1])]
    scenes = scene if isinstance(scene, list) else [scene]
    extra = sys.argv[2:] or ["-pql"]
    here = Path(__file__).resolve().parent
    cmd = [sys.executable, "-m", "manim", *extra, str(here / file), *scenes]
    print("+", " ".join(cmd))
    raise SystemExit(subprocess.call(cmd, cwd=here))


if __name__ == "__main__":
    main()
