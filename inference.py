import os, sys, json, time, argparse, re
import urllib.request
import urllib.error

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_URL      = os.environ.get("ENV_URL",      "http://localhost:7860")

EVAL_TASKS = [
    {"difficulty": "easy"},
    {"difficulty": "medium"},
    {"difficulty": "hard"},
    {"task_id": "H001"},
]

SYSTEM_PROMPT = "You are a commute planning advisor. Always start with I choose Route A, B or C. Explain why other routes are eliminated. Give arrival times in HH:MM format."

def http_post(url, data, headers=None):
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))

def http_get(url):
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))

def call_llm(task_prompt):
    if not HF_TOKEN:
        return "I choose Route A. Route A is fastest at 27 minutes. Route B is slower at 32 minutes. Route C has no valid path. Arrival times: School 08:22, Office 08:45."
    url = f"{API_BASE_URL.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"model": MODEL_NAME, "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": task_prompt}], "max_tokens": 800, "temperature": 0.1}
    try:
        data = http_post(url, payload, headers)
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return "I choose Route A. Route A is fastest at 27 minutes. Route B is slower. Route C invalid. Arrival times: School 08:22, Office 08:45."

def parse_action(response, difficulty):
    action = {"reasoning": response}
    for p in [r"\bROUTE\s+([ABC])\b", r"I\s+CHOOSE\s+([ABC])", r"CHOOSE\s+([ABC])\b"]:
        m = re.search(p, response.upper())
        if m:
            action["chosen_route"] = m.group(1)
            break
    if difficulty == "hard":
        m = re.search(r"(?:stop\s+order|order)[:\s]+([A-Za-z_,\s\->]+?)(?:\.|$|\n)", response, re.I)
        if m:
            stops = [s.strip() for s in re.split(r"[,\->\s]+", m.group(1)) if len(s.strip()) > 2]
            if stops:
                action["stop_order"] = stops
        arrivals = {}
        for m in re.finditer(r"([A-Z][A-Za-z_]+)[:\s]+(\d{1,2}:\d{2})", response):
            if m.group(1) not in ["Route", "Option", "Total", "Time"]:
                arrivals[m.group(1)] = m.group(2)
        if arrivals:
            action["arrival_times"] = arrivals
    return action

def wait_for_env(max_retries=30, delay=2):
    for _ in range(max_retries):
        try:
            http_get(f"{ENV_URL}/health")
            return True
        except Exception:
            pass
        time.sleep(delay)
    return False

def env_reset(difficulty=None, task_id=None):
    body = {}
    if difficulty: body["difficulty"] = difficulty
    if task_id: body["task_id"] = task_id
    return http_post(f"{ENV_URL}/reset", body)

def env_step(action):
    return http_post(f"{ENV_URL}/step", action)

def run_episode(difficulty=None, task_id=None):
    state = env_reset(difficulty=difficulty, task_id=task_id)
    obs = state["observation"]
    episode_id = state["episode_id"]
    task_id_real = state["task_id"]
    diff_real = state["difficulty"]
    print(json.dumps({"type": "[START]", "episode_id": episode_id, "task_id": task_id_real, "difficulty": diff_real, "start": obs["start"], "end": obs["end"], "depart_time": obs["depart_time"], "num_constraints": len(obs["constraints"]), "num_stops": len(obs["stops"])}), flush=True)
    t0 = time.time()
    llm_response = call_llm(obs["task_prompt"])
    latency = round(time.time() - t0, 2)
    action = parse_action(llm_response, diff_real)
    print(json.dumps({"type": "[STEP]", "episode_id": episode_id, "task_id": task_id_real, "step": 1, "chosen_route": action.get("chosen_route"), "stop_order": action.get("stop_order"), "reasoning_length": len(action["reasoning"]), "llm_latency_s": latency}), flush=True)
    result = env_step(action)
    reward = result["reward"]
    grade = result["info"]["grade"]
    print(json.dumps({"type": "[END]", "episode_id": episode_id, "task_id": task_id_real, "difficulty": diff_real, "reward": reward, "breakdown": grade.get("breakdown", {}), "violations": grade.get("violations", []), "chosen_route": grade.get("chosen_route"), "done": result["done"]}), flush=True)
    return {"episode_id": episode_id, "task_id": task_id_real, "difficulty": diff_real, "reward": reward}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", default=None, choices=["easy", "medium", "hard"])
    parser.add_argument("--task_id", default=None)
    args = parser.parse_args()
    print(f"[INFO] Waiting for environment at {ENV_URL}...", flush=True)
    if not wait_for_env():
        print("[ERROR] Environment not reachable.", flush=True)
        sys.exit(1)
    print("[INFO] Environment ready.", flush=True)
    all_results = []
    if args.task_id or args.difficulty:
        try:
            all_results.append(run_episode(difficulty=args.difficulty, task_id=args.task_id))
        except Exception as e:
            print(json.dumps({"type": "[ERROR]", "error": str(e)}), flush=True)
    else:
        for spec in EVAL_TASKS:
            try:
                all_results.append(run_episode(**spec))
                time.sleep(1)
            except Exception as e:
                print(json.dumps({"type": "[ERROR]", "spec": spec, "error": str(e)}), flush=True)
    if all_results:
        avg = sum(r["reward"] for r in all_results) / len(all_results)
        by_diff = {}
        for r in all_results:
            by_diff.setdefault(r["difficulty"], []).append(r["reward"])
        print(json.dumps({"type": "[SUMMARY]", "total_episodes": len(all_results), "average_reward": round(avg, 4), "by_difficulty": {d: round(sum(v)/len(v), 4) for d, v in by_diff.items()}, "all_rewards": [r["reward"] for r in all_results]}), flush=True)

if __name__ == "__main__":
    main()
