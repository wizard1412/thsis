#!/usr/bin/env python3
"""
平行實驗執行器 (Zero-Shot + LLMLasso)

用法:
  python run_experiments.py --experiments qwen72b_no_amplitude qwen32b_amplitude  # 同時跑指定實驗
  python run_experiments.py --all                                                  # 同時跑所有實驗
  python run_experiments.py --list                                                 # 列出所有實驗

注意:
  每個實驗跑在獨立的 Thread 中，互不干擾，結果存到各自的 output_dir。
  Ollama 本地服務每次只能處理一個請求，多個實驗的 LLM 呼叫會在 Ollama 端自動排隊，
  整體吞吐量不會因平行而變快，但實驗管理（特徵提取、結果儲存）可同步進行。
  若你有多個 Ollama 實例或不同 model，平行才能真正提速。
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from batch_score import EXPERIMENTS, run_experiment


def main():
    parser = argparse.ArgumentParser(
        description="平行執行多個特徵組合實驗 (Zero-Shot + LLMLasso)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--experiments", "-e",
        nargs="+",
        metavar="NAME",
        help="要平行執行的實驗名稱\n例: --experiments qwen72b_no_amplitude qwen32b_amplitude",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="平行執行 EXPERIMENTS dict 中的所有實驗",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="列出所有可用的實驗名稱",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        metavar="N",
        help="最大同時執行數（預設 = 實驗數量）",
    )
    args = parser.parse_args()

    # --list
    if args.list:
        print("Available experiments:")
        for name, cfg in EXPERIMENTS.items():
            print(f"  {name}")
            print(f"    model   : {cfg.get('model')}")
            print(f"    output  : {cfg.get('output_dir')}")
            print(f"    features: {cfg.get('features')}")
        sys.exit(0)

    # 決定要跑哪些實驗
    if args.all:
        to_run = list(EXPERIMENTS.keys())
    elif args.experiments:
        to_run = args.experiments
    else:
        parser.print_help()
        sys.exit(1)

    # 驗證名稱
    for name in to_run:
        if name not in EXPERIMENTS:
            print(f"Error: Unknown experiment '{name}'. Use --list to see available experiments.")
            sys.exit(1)

    max_workers = args.workers or len(to_run)

    print("=" * 60)
    print("Parallel Experiment Runner (Zero-Shot + LLMLasso)")
    print("=" * 60)
    print(f"Experiments ({len(to_run)}): {', '.join(to_run)}")
    print(f"Max workers : {max_workers}")
    print("=" * 60 + "\n")

    results = {}
    failed = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_name = {
            executor.submit(run_experiment, name): name
            for name in to_run
        }

        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                future.result()
                results[name] = "OK"
                print(f"\n[{name}] ✓ Finished")
            except Exception as e:
                results[name] = f"FAILED: {e}"
                failed.append(name)
                print(f"\n[{name}] ✗ Failed: {e}")

    # 最終摘要
    print("\n" + "=" * 60)
    print("Experiment Summary")
    print("=" * 60)
    for name, status in results.items():
        mark = "✓" if status == "OK" else "✗"
        print(f"  {mark} {name}: {status}")

    if failed:
        print(f"\nFailed experiments: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("\nAll experiments completed successfully.")


if __name__ == "__main__":
    main()
