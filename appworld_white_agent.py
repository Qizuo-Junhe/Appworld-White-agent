# my_agent_refactored.py
import os
import json
import re
import time
import logging
import traceback
from typing import List, Dict, Any, Optional, Set
from openai import OpenAI
from appworld import AppWorld, load_task_ids

class Config:
    APPWORLD_ROOT = r"D:\AI\appworld"
    OPENAI_MODEL = "gpt-4o-mini"
    MAX_STEPS = 50
    LOG_FILE = "agent_trace.jsonl"

    @staticmethod
    def setup():
        os.environ["APPWORLD_ROOT"] = Config.APPWORLD_ROOT
        if not os.environ.get("OPENAI_API_KEY"):
            print("WARNING: OPENAI_API_KEY not found in environment variables.")

class ConversationLogger:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.console_logger = logging.getLogger("Agent")
        self.console_logger.setLevel(logging.INFO)
        if not self.console_logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
            self.console_logger.addHandler(ch)

    def info(self, msg: str):
        self.console_logger.info(msg)

    def error(self, msg: str):
        self.console_logger.error(msg)

    def log_turn(self, step: int, memory_snapshot: List[Dict], action: Dict, observation: Dict):

        entry = {
            "timestamp": time.time(),
            "step": step,
            "memory_last_3": memory_snapshot[-3:] if len(memory_snapshot) > 3 else memory_snapshot,
            "action": action,
            "observation": observation
        }
        try:
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            self.error(f"Failed to write log: {e}")

class ToolExecutor:
    def __init__(self, world: AppWorld, logger: ConversationLogger):
        self.world = world
        self.logger = logger

    def execute(self, tool_name: str, args: Dict) -> Dict:
        if "." not in tool_name:
            return {"success": False, "error": "Invalid tool name format (expected app.api)"}
        
        app, api = tool_name.split(".", 1)
        
        safe_args = []
        for k, v in args.items():
            safe_val = json.dumps(v, ensure_ascii=False)
            safe_args.append(f"{k}={safe_val}")
        args_str = ", ".join(safe_args)

        code_block = [
            "import json",
            "output = {'success': False, 'result': None, 'error': None}",
            "try:",
            f"    resp = apis.{app}.{api}({args_str})",
            "    output['result'] = resp",
            "    output['success'] = True",
            "except Exception as e:",
            "    output['error'] = str(e)",
            "print(json.dumps(output))"
        ]
        
        try:
            raw_response = self.world.execute("\n".join(code_block))
            return self._safe_json_load(raw_response)
        except Exception as e:
            tb = traceback.format_exc()
            self.logger.error(f"Executor System Error: {e}\n{tb}")
            return {"success": False, "error": f"System execution failed: {str(e)}"}

    def _safe_json_load(self, text: str) -> Dict:
        try:
            return json.loads(text)
        except:
            m = re.search(r'(\{.*\})', text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1))
                except:
                    pass
            return {"success": False, "error": "Output is not valid JSON", "raw": text}
    
    def execute_raw_code(self, python_code: str) -> Dict:
        try:
            raw_response = self.world.execute(python_code)
            return self._safe_json_load(raw_response)
        except Exception as e:
            return {"success": False, "error": f"Raw execution failed: {e}"}
        
class TaskAnalyzer:
    ALL_SUPPORTED_APPS = [
        "amazon", "gmail", "spotify", "venmo", 
        "todoist", "simple_note", "calendar"
    ]

    def __init__(self, client: OpenAI, model: str, executor: ToolExecutor, logger: ConversationLogger):
        self.client = client
        self.model = model
        self.executor = executor
        self.logger = logger

    def identify_apps(self, task_instruction: str) -> List[str]:
        selected_apps = {"supervisor", "api_docs"}

        apps_list_str = ", ".join(self.ALL_SUPPORTED_APPS)
        
        system_prompt = f"""
        You are a dependency analyzer for an AI agent.
        Select the NECESSARY apps from the list below to complete the user's task.
        
        [AVAILABLE APPS CANDIDATES]
        {apps_list_str}
        
        [OUTPUT FORMAT]
        Return ONLY a JSON array of strings. Example: ["gmail", "calendar"]
        If no specific app is needed, return [].
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Task: {task_instruction}"}
                ],
                temperature=0.0,
                max_tokens=100
            )
            content = response.choices[0].message.content
            identified = self._parse_json_list(content)
        
            for app in identified:
                clean_app = app.lower().strip()
                if clean_app in self.ALL_SUPPORTED_APPS:
                    selected_apps.add(clean_app)
                    
        except Exception as e:
            self.logger.error(f"TaskAnalyzer LLM error: {e}")

        return list(selected_apps)

    def _parse_json_list(self, text: str) -> List[str]:
        text = text.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(text)
        except:
            match = re.search(r'\[.*\]', text, re.DOTALL)
            return json.loads(match.group(0)) if match else []

class ToolRegistry:
    def __init__(self, executor: ToolExecutor, logger: ConversationLogger):
        self.executor = executor
        self.logger = logger

    def fetch_tools_schema(self, apps: List[str]) -> Dict:
        schemas = {}
        for app in apps:
            resp = self.executor.execute("api_docs.show_api_descriptions", {"app_name": app})
            if resp.get("success"):
                schemas[app] = self._simplify_schema(resp.get("result"))
            else:
                self.logger.error(f"Failed to fetch docs for {app}: {resp.get('error')}")
        return schemas

    def fetch_supervisor_data(self, relevant_apps: List[str]) -> Dict:
        data = {"profile": {}, "credentials": []}
        
        p_resp = self.executor.execute("supervisor.show_profile", {})
        if p_resp.get("success"):
            data["profile"] = p_resp.get("result")

        c_resp = self.executor.execute("supervisor.show_account_passwords", {})
        if c_resp.get("success"):
            all_creds = c_resp.get("result", [])
            if isinstance(all_creds, list):
                for cred in all_creds:
                    acct_name = str(cred.get("account_name", "")).lower()
                    if any(app in acct_name for app in relevant_apps) or "supervisor" in acct_name:
                        data["credentials"].append(cred)
        
        return data

    def _simplify_schema(self, api_docs):
        simplified = []
        if isinstance(api_docs, list):
            for entry in api_docs:
                if isinstance(entry, dict):
                    name = entry.get("name")
                    desc = entry.get("description", "")
                    params = []
                    p = entry.get("parameters") or entry.get("params") or []
                    if isinstance(p, list):
                        for x in p:
                            if isinstance(x, dict):
                                params.append(x.get("name"))
                    elif isinstance(p, dict):
                        params = list(p.keys())
                    
                    simplified.append(f"{name}({', '.join(params if params else [])}): {desc[:100]}")
        return simplified

class PromptManager:
    def __init__(self, tools_schema: Dict, supervisor_data: Dict):
        self.tools_schema = tools_schema
        self.profile = supervisor_data.get("profile", {})
        self.credentials = supervisor_data.get("credentials", [])

    def get_system_prompt(self) -> str:
        tools_str = json.dumps(self.tools_schema, indent=2, ensure_ascii=False)
        profile_str = json.dumps(self.profile, ensure_ascii=False)
        creds_str = json.dumps(self.credentials, indent=2, ensure_ascii=False)

        prompt = f"""
        You are an AI agent in a simulated environment called AppWorld.
        You need to write Python code to operate the app.
        
        [USER PROFILE]
        User Profile: {profile_str}

        [AVAILABLE CREDENTIALS]
        (Use these strictly for login. Do not ask the user for passwords.Also, do not guess usernames, emails, passwords, etc.; obtain all necessary information from here.)
        {creds_str}

        [AVAILABLE TOOLS]
        (These are the APIs and their descriptions that you will use when accessing user data and operating the app. You must not guess the API names; all operations must use the APIs listed here.)
        {tools_str}

        [CRITICAL RULES - READ CAREFULLY]
        1. **LOGIN PARAMETERS MAPPING (CRITICAL)**:
           - When a login tool asks for a `username`, you MUST use the `email` address from the credentials.
           - **DO NOT** use the `account_name` or app name (e.g., "spotify") as the username.
           - Example: If creds are `{{'account_name': 'spotify', 'email': 'bob@mail.com'}}`, you MUST call `login(username='bob@mail.com')`.

        2. **STATELESS API & TOKEN PERSISTENCE (MOST IMPORTANT)**:
           - **DO NOT RE-LOGIN**: Once you successfully log in and get an `access_token`, **YOU MUST REUSE IT** for all future steps. Do not call `login` again unless you receive a specific "401 Unauthorized" error.
           - **CHECK PARAMETER NAME**: Look at the schema. Is the required parameter named `access_token` or `token`? Use the EXACT name.
           - **PUBLIC vs PRIVATE TOOLS**:
             * **Private Tools** (Require Token): `login`, `update_...`, `create_...`, `delete_...`, `..._privates`, `review_...`, `show_liked_...`.
             * **Public Tools** (NO Token): `search_...`, `show_song`, `show_album`, `show_artist`, `show_..._reviews`.
           - **WARNING**: Only pass the token IF the tool explicitly accepts it in the schema. Do not force it into public tools.
           - If an API call fails with "401 Unauthorized", it implies you forgot the token.

        3. **TERMINATION PROTOCOL (HOW TO SUBMIT)**:
           - **NEVER output `none` as a tool.** This causes an infinite loop.
           - You CANNOT stop just by finding the answer in your reasoning. You MUST submit it to the system.
           - **REQUIRED ACTION**: Use `supervisor.complete_task(answer='YOUR_ANSWER')` to finish the task.
           - If the task involves a change of state (e.g. sending an email), and no answer is required, use `complete_task(status='completed')` or check the schema for the correct argument.
           # [REMOVED DUPLICATE]: The text regarding token persistence and public/private tools was removed here as it was a duplicate of Rule #2.

        4. **CHAINING OUTPUTS**: 
           - Always use the output of the previous step. 
           - Example: `search_song` returns `song_id`. Next step MUST be `play_song(song_id=...)`. 
           - **NO ID GUESSING**: Do not fabricate IDs (e.g., `song_id="123"`). You must find them first.

        5. **DATA INTERPRETATION**:
           - **"Most Liked" != "Most Recent"**. Do not confuse timestamps (`liked_at`) with popularity (`popularity` or `like_count`).
           - If specific metrics aren't available, check for sorted lists or explicit "top" endpoints before guessing based on dates.
           - **DRILL DOWN**: If a summary list (e.g., `show_liked_songs`) does not show play counts, you MUST iterate through the items and call detail tools (e.g., `show_song`) to get the actual stats. Do not guess based on dates.

        6. **LOGIN FIRST**: 
           - If the first observation is a "401 Unauthorized" or "Login required", your IMMEDIATE next action must be to log in using the credentials provided above (remember Rule #1).

        7. **ARGUMENT ACCURACY**: 
           - **ANTI-HALLUCINATION**: Before generating args, compare them against the `[TOOL SCHEMAS]` above.
           - **ID Names**: Check if it requires `review_id` or `song_review_id`. Do not invent keys.
           - Ensure arguments match the tool schema exactly (e.g., integers for limits, correct string formats).
           - **COMMON TRAPS**: 
             * For reviews/ratings, the ID is usually `review_id` (NOT `song_review_id`).
             * The value to set is usually `rating` (NOT `new_rating`).
             * Status is usually `success` or `fail` (NOT `completed`).

        8. **STEP-BY-STEP**: 
           - Don't try to do everything in one step. Search -> Verify -> Act -> Submit.

        9. **CHECK BEFORE WRITING (AVOID 409/422 ERRORS)**:
           - Before rating or reviewing, ALWAYS call `show_..._reviews` first to check if you have already reviewed it.
           - **IF EXISTS**: Use `update_..._review` (using `review_id`).
           - **IF NOT EXISTS**: Use `review_...` or `create_...` (using `song_id`).
           - Do not blindly try to create; a 409 error means it already exists. A 422 error means you tried to update someone else's review.
        
        10. **EFFICIENT LIST PROCESSING**:
           - If you need to process a list of items (e.g., "rate all songs"), do it one by one efficiently.
           - **CRITICAL**: Do NOT log in between items. Use the SAME `access_token` for item 1, item 2, item 3...

        [RESPONSE FORMAT]
        You must strictly respond with a valid JSON object. Do not add any markdown or text outside the JSON.
        {{
            "reasoning": "1. Analyze previous observation. 2. State current goal. 3. Explain why this specific tool is chosen.",
            "action": {{
                "tool": "app_name.function_name",
                "args": {{ "arg_name": "value" }}
            }}
        }}

        """
        return prompt.strip()

    def get_step_prompt(self, step: int, instruction: str, last_observation: str) -> str:

        prompt = f"""
        [TASK INSTRUCTION]
        {instruction}

        [CURRENT STATUS]
        Step: {step}
        
        [LAST OBSERVATION]
        {last_observation}

        [INSTRUCTION FOR NEXT STEP]
        Based on the Last Observation:
        1. If it was an ERROR: Analyze why it failed. Do not repeat the exact same parameters. Change your approach or fix the arguments.
        2. If it was SUCCESS: Proceed to the next logical step.
        3. If you found IDs: Use them in the next action.
        
        Generate the next JSON response now.
        """
        return prompt.strip()

class AppWorldAgent:
    def __init__(self, world: AppWorld, task_id: str):
        self.world = world
        self.task_id = task_id
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.logger = ConversationLogger(Config.LOG_FILE)

        self.executor = ToolExecutor(world, self.logger)
        self.task_analyzer = TaskAnalyzer(
            self.client, 
            Config.OPENAI_MODEL, 
            self.executor,
            self.logger
        )

        self.registry = ToolRegistry(self.executor, self.logger)
        
        self.memory: List[Dict] = []
        self.step = 0
        self.prompt_manager: Optional[PromptManager] = None

    def initialize(self):
        self.logger.info(f"Initializing Agent for Task: {self.task_id}")
        task_instruction = getattr(self.world.task, "instruction", "")
        self.logger.info(f"Instruction: {task_instruction}")

        # 1. Identify Apps
        required_apps = self.task_analyzer.identify_apps(task_instruction)
        self.logger.info(f"Identified Apps: {required_apps}")

        # 2. Fetch Docs
        tools_schema = self.registry.fetch_tools_schema(required_apps)
        
        # 3. Fetch Supervisor Data
        supervisor_context = self.registry.fetch_supervisor_data(required_apps)
        self.logger.info(f"Context Loaded. Profile: {bool(supervisor_context['profile'])}, Creds: {len(supervisor_context['credentials'])}")

        # 4. Initialize Prompt Manager & Memory
        self.prompt_manager = PromptManager(tools_schema, supervisor_context)
        
        self.memory.append({
            "role": "system", 
            "content": self.prompt_manager.get_system_prompt()
        })

    def run(self):
        self.initialize()
        
        task_instruction = getattr(self.world.task, "instruction", "")
        last_obs = "None (Start of task)"

        while self.step < Config.MAX_STEPS:
            self.step += 1
            self.logger.info(f"--- Step {self.step} ---")

            user_msg = self.prompt_manager.get_step_prompt(self.step, task_instruction, last_obs)
            self.memory.append({"role": "user", "content": user_msg})

            try:
                response = self.client.chat.completions.create(
                    model=Config.OPENAI_MODEL,
                    messages=self.memory,
                    response_format={ "type": "json_object" },
                    temperature=0.0
                )
                content = response.choices[0].message.content
                self.memory.append({"role": "assistant", "content": content})
            except Exception as e:
                self.logger.error(f"LLM Call Error: {e}")
                last_obs = f"System Error: LLM call failed {e}"
                continue

            action_data = self._parse_json(content)
            if not action_data or "action" not in action_data:
                self.logger.error("Invalid JSON from LLM")
                last_obs = "Error: Invalid JSON format. Please return {'reasoning':..., 'action':{...}}"
                continue

            tool = action_data["action"].get("tool")
            args = action_data["action"].get("args", {})
            reasoning = action_data.get("reasoning", "No reasoning provided")
            
            self.logger.info(f"Reasoning: {reasoning}")
            self.logger.info(f"Executing: {tool} | Args: {args}")

            obs_data = self.executor.execute(tool, args)
            last_obs = json.dumps(obs_data, ensure_ascii=False)
            
            self.logger.log_turn(self.step, self.memory, action_data, obs_data)

            if self.world.task_completed():
                self.logger.info(">>> TASK COMPLETED SUCCESSFULLY <<<")
                return True

            time.sleep(0.5)

        self.logger.info("Max steps reached. Task failed.")
        return False

    def _parse_json(self, text):
        text = text.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(text)
        except:
            return None