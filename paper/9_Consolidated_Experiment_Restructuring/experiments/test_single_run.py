import sys, json
sys.path.insert(0, r'd:\trunk\SIG\output\cognitive-outsourcing\paper\9_Consolidated_Experiment_Restructuring\experiments')
from common import collect_run, save_run, parse_kitchen_metrics, RESULTS_DIR

print("=== EXP-1 Single Run Test ===")
record = collect_run('EXP-1', 'SIG', '4B', 0, 'kitchen', 35)
print("ok:", record['ok'])
print("wall_clock_s:", round(record['wall_clock_s'], 2))

baselines = record.get('parsed_baselines', {})
print("parsed baselines:", list(baselines.keys()))

if 'SIG' in baselines and 'AppLoop' in baselines:
    sig = baselines['SIG']
    app = baselines['AppLoop']
    print("SIG: wc=%ss gen=%ss pf=%ss" % (sig['wall_clock_s'], sig['gen_s'], sig['prefill_s']))
    print("AppLoop: wc=%ss gen=%ss pf=%ss" % (app['wall_clock_s'], app['gen_s'], app['prefill_s']))
    if sig['wall_clock_s'] > 0:
        speedup = app['wall_clock_s'] / sig['wall_clock_s']
        print("Speedup: %.2fx" % speedup)
    print("PASS: Parser correctly extracted baseline data")
else:
    print("FAIL: Parser did not find expected baselines")
    print("Available keys:", list(baselines.keys()))

fname = save_run('EXP-1', 'test', 0, record)
print("Saved to:", fname)
print("File exists:", fname.exists())
