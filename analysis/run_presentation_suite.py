import os
import sys
import subprocess
import time

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)


ANALYSES = [
    ("YoY Reliability & Autocorrelation", "analysis/yoy_reliability.py"),
    ("Out-of-Sample Predictive Power", "analysis/predictive_validity.py"),
    ("Sample Size Stabilization Curve", "analysis/stabilization_analysis.py"),
    ("Macro vs Micro Talent Quadrants", "analysis/quadrant_analysis.py"),
    ("Historical Breakouts & Declines", "analysis/breakout_detection.py"),
    ("Pitcher Deception & Induced Errors", "analysis/pitcher_deception.py"),
]


def run_suite():
    print("=" * 80)
    print("  EXECUTING COMPREHENSIVE SABERMETRICS PRESENTATION SUITE (2020-2025)")
    print("=" * 80)
    
    t_start = time.time()
    results = []
    
    for idx, (title, script_path) in enumerate(ANALYSES, start=1):
        print(f"\n[{idx}/{len(ANALYSES)}] Running {title} ({script_path})...")
        t0 = time.time()
        try:
            res = subprocess.run([sys.executable, script_path], capture_output=True, text=True, check=True)
            elapsed = time.time() - t0
            print(f"  ✓ Finished in {elapsed:.1f}s")
            results.append((title, "SUCCESS", elapsed, res.stdout))
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - t0
            print(f"  ✗ Failed after {elapsed:.1f}s: {e.stderr}")
            results.append((title, "FAILED", elapsed, e.stderr))
            
    total_elapsed = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"  PRESENTATION SUITE COMPLETED in {total_elapsed:.1f}s")
    print("=" * 80)
    
    print("\nExecution Summary:")
    for title, status, elapsed, _ in results:
        status_sym = "✓" if status == "SUCCESS" else "✗"
        print(f"  {status_sym} {title:<40} {status:<10} ({elapsed:.1f}s)")
        
    print("\nGenerated Presentation Assets in assets/presentation/:")
    asset_dir = os.path.join(_PROJECT_ROOT, 'assets', 'presentation')
    if os.path.exists(asset_dir):
        for f in sorted(os.listdir(asset_dir)):
            if f.endswith(('.png', '.csv')):
                fsize = os.path.getsize(os.path.join(asset_dir, f)) / 1024
                print(f"  • {f:<38} ({fsize:.1f} KB)")


if __name__ == "__main__":
    run_suite()
