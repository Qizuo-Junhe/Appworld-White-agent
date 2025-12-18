import time
import os
import json
import datetime
from collections import defaultdict
from appworld import AppWorld, load_task_ids

try:
    from my_appworld_agent import AppWorldAgent, Config
except ImportError:
    print("❌ Error: can not find 'my_appworld_agent.py'")
    exit(1)

def evaluate_dataset(dataset_name: str, num_tasks: int = None):
    print(f"\n{'='*60}")
    print(f"🚀 STARTING EVALUATION FOR: {dataset_name.upper()}")
    print(f"{'='*60}")

    try:
        all_task_ids = load_task_ids(dataset_name)
    except Exception as e:
        print(f"❌ Failed to load dataset '{dataset_name}': {e}")
        return None

    task_list = all_task_ids[:num_tasks] if num_tasks else all_task_ids
    print(f"📂 Total Tasks Loaded: {len(task_list)}")

    scenario_map = defaultdict(list)
    results_log = []
    
    start_time = time.time()

    for i, tid in enumerate(task_list):
        scenario_id = tid.split('_')[0]
        
        print(f"[{i+1}/{len(task_list)}] Task: {tid:<12} | Scen: {scenario_id:<8} ... ", end="", flush=True)
        
        task_start_time = time.time()
        success = False
        used_steps = 0
        
        try:
            with AppWorld(task_id=tid) as world:
                agent = AppWorldAgent(world, tid)
                success = agent.run()
                used_steps = agent.step
                
        except Exception as e:
            print(f"⚠️ CRASH: {str(e)[:50]}...", end=" ")
            success = False
            if 'agent' in locals():
                used_steps = agent.step
        
        task_duration = time.time() - task_start_time
        
        scenario_map[scenario_id].append(success)
        
        results_log.append({
            "id": tid, 
            "success": success, 
            "steps": used_steps,
            "time_s": task_duration
        })
        
        status_icon = "✅ PASS" if success else "❌ FAIL"
        print(f"{status_icon} (Steps: {used_steps}, Time: {task_duration:.1f}s)")

    total_duration = time.time() - start_time
    
    total_tasks = len(results_log)
    passed_tasks = sum(1 for r in results_log if r['success'])
    
    tgc = (passed_tasks / total_tasks * 100) if total_tasks > 0 else 0.0

    total_scenarios = len(scenario_map)
    passed_scenarios = sum(1 for outcomes in scenario_map.values() if all(outcomes))
    sgc = (passed_scenarios / total_scenarios * 100) if total_scenarios > 0 else 0.0

    total_steps_sum = sum(r['steps'] for r in results_log)
    avg_steps = total_steps_sum / total_tasks if total_tasks > 0 else 0.0

    avg_time = total_duration / total_tasks if total_tasks > 0 else 0.0

    print(f"\n{'-'*60}")
    print(f"📊 REPORT: {dataset_name.upper()}")
    print(f"{'-'*60}")
    print(f"⏱️  Total Time:      {total_duration:.2f}s")
    print(f"⏱️  Avg Time/Task:   {avg_time:.2f}s")  
    print(f"👣 Avg Steps/Task:  {avg_steps:.2f}")
    print(f"🎯 TGC (Task %):    {tgc:.2f}%  ({passed_tasks}/{total_tasks})")
    print(f"🎯 SGC (Scenario %): {sgc:.2f}%  ({passed_scenarios}/{total_scenarios})")
    print(f"{'='*60}\n")

    return {
        "dataset": dataset_name,
        "tgc": tgc,
        "sgc": sgc,
        "avg_steps": avg_steps,
        "total_time": total_duration,
        "avg_time": avg_time,        
        "tasks": total_tasks,
        "scenarios": total_scenarios,
        "details": results_log
    }

def run_benchmark_suite(task_limit_per_set=None):
    Config.setup()
    
    # datasets = ["train", "dev"]
    datasets = ["test_normal", "test_challenge"]
    final_stats = []

    for ds in datasets:
        stats = evaluate_dataset(ds, num_tasks=task_limit_per_set)
        if stats:
            final_stats.append(stats)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"benchmark_report_{timestamp}.json"
    
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(final_stats, f, ensure_ascii=False, indent=2)
        print(f"💾 Results saved to local file: {output_filename}")
    except Exception as e:
        print(f"⚠️ Failed to save results: {e}")

    print("\n\n")
    print("="*100)
    print(f"{'🏆 FINAL BENCHMARK SUMMARY':^100}")
    print("="*100)
    header = f"{'Dataset':<16} | {'Tasks':<6} | {'Scen.':<6} | {'TGC (%)':<9} | {'SGC (%)':<9} | {'Avg Steps':<10} | {'Avg Time':<9} | {'Total Time':<10}"
    print(header)
    print("-" * 100)
    
    for stat in final_stats:
        row = f"{stat['dataset']:<16} | {stat['tasks']:<6} | {stat['scenarios']:<6} | {stat['tgc']:<9.2f} | {stat['sgc']:<9.2f} | {stat['avg_steps']:<10.2f} | {stat['avg_time']:<9.2f} | {stat['total_time']:<10.2f}"
        print(row)
    
    print("-" * 100)
    print("Note: SGC requires 100% success on all tasks within a scenario.")
    print("="*100)

if __name__ == "__main__":
    run_benchmark_suite(task_limit_per_set=None)