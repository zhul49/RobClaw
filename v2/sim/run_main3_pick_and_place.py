import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                    # sibling imports inside sim/
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # 'v2.X.Y' resolution
sys.path.insert(0, os.path.expanduser("~/LIBERO"))

from v2.sim.main3 import Main3


DEFAULT_CONSTRAINT_DIR = os.path.join(
    HERE, "..", "..", "outputs", "v2_libero_object_0", "constraints",
    "PATCHED_kp0_pick_up_the_alphabet_soup_and_place_it_in_the_basket",
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="libero_object")
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--constraint-dir", default=DEFAULT_CONSTRAINT_DIR,
                   help="Directory holding stage*_subgoal/path_constraints.txt + metadata.json.")
    p.add_argument("--out", default=None,
                   help="Run output dir. Defaults to outputs/v2_<suite>_<task>/run_<ts>/.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_out = args.out or os.path.join(
        HERE, "..", "outputs", f"v2_{args.suite}_{args.task}", f"run_{ts}"
    )
    os.makedirs(run_out, exist_ok=True)
    video_target = os.path.join(run_out, "run.mp4")

    print(f"suite/task     : {args.suite}/{args.task}")
    print(f"constraint dir : {args.constraint_dir}")
    print(f"run output dir : {run_out}")

    t0 = time.perf_counter()
    m = Main3(suite_name=args.suite, task_idx=args.task, verbose=args.verbose)
    print(f"Main3 constructed in {time.perf_counter() - t0:.1f}s")

    # Redirect the env's video writer to our run dir.
    import cv2
    m.env._video_writer.release()
    m.env._video_path = video_target
    m.env._video_writer = cv2.VideoWriter(
        video_target, cv2.VideoWriter_fourcc(*"mp4v"), 10, (480, 480)
    )

    m.load_task(args.constraint_dir)

    t0 = time.perf_counter()
    diag = m.execute_task()
    print(f"\nexecute_task wall time: {time.perf_counter() - t0:.1f}s")
    print(f"total iters           : {diag['total_iters']}")
    for d in diag["per_stage"]:
        print(f"  stage {d['stage']}: iters={d['iters']} "
              f"wall={d['wall_time_s']:.2f}s "
              f"grasp={d['is_grasp_stage']} release={d['is_release_stage']}")
    if diag.get("backtracks"):
        print(f"  backtracks: {diag['backtracks']}")

    try:
        success = bool(m.env.env.check_success())
    except Exception as e:
        print(f"check_success raised: {e}")
        success = False

    video_path = m.env.save_video()
    print(f"\nLIBERO oracle  : {success}")
    print(f"video          : {video_path}")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
