import ast
import json
import re
from typing import List
from abc import ABC, abstractmethod

from decord import VideoReader

from .core import Action, Observation, Trajectory, TrajectoryStep
from .tools import DEFAULT_TOOL_REGISTRY
from .utils import call_llm_api, convert_to_free_form_text_representation, get_message_content, load_subtitles


def extract_actions_from_text(text: str, valid_tools: set) -> List[Action]:
    actions = []
    if not text:
        return actions

    # Pattern 1: JSON objects with "name" or "function"
    json_blocks = re.findall(r'(\{(?:[^{}]|(?:\{[^{}]*\}))*\})', text)
    for j_str in json_blocks:
        try:
            j_clean = re.sub(r'^.*?\{', '{', j_str)
            try:
                data = json.loads(j_clean)
            except json.JSONDecodeError:
                data = ast.literal_eval(j_clean)
            if isinstance(data, dict):
                fname = data.get("name") or data.get("function")
                if isinstance(fname, dict):
                    fname = fname.get("name")
                params = data.get("parameters") or data.get("arguments") or {}
                if isinstance(params, str):
                    try:
                        params = json.loads(params)
                    except Exception:
                        params = {}
                if fname in valid_tools:
                    actions.append(Action(function_name=fname, parameters=params, function_id=f"{fname}:0"))
        except Exception:
            pass

    if actions:
        return actions

    # Pattern 2: tool_call('overview') or tool_call('skim', start_time=0, end_time=180, query='...')
    tc_matches = re.findall(r"tool_call\s*\(\s*['\"]?(\w+)['\"]?\s*(?:,\s*(.*?))?\)", text, flags=re.DOTALL)
    for fname, args_str in tc_matches:
        if fname in valid_tools:
            params = {}
            if args_str:
                kw_matches = re.findall(r"(\w+)\s*=\s*('[^']*'|\"[^\"]*\"|\d+(?:\.\d+)?|True|False)", args_str)
                for k, v in kw_matches:
                    v_clean = v.strip("'\"")
                    if v_clean.isdigit():
                        params[k] = int(v_clean)
                    else:
                        try:
                            params[k] = float(v_clean)
                        except ValueError:
                            params[k] = v_clean
            actions.append(Action(function_name=fname, parameters=params, function_id=f"{fname}:0"))

    if actions:
        return actions

    # Pattern 3: Mention of tool names in prose
    for fname in ["overview", "skim", "focus", "answer"]:
        if fname in valid_tools and re.search(rf"\b(?:call|use|run)\s+(?:the\s+)?`?{fname}`?\b", text, re.I):
            actions.append(Action(function_name=fname, parameters={}, function_id=f"{fname}:0"))
            break

    return actions


def normalize_action_parameters(action: Action, tool_registry) -> bool:
    if not isinstance(action.parameters, dict):
        return False

    schema = tool_registry.get_tool(action.function_name)["function"]["parameters"]
    properties = schema.get("properties", {})
    for key, value in action.parameters.items():
        expected_type = properties.get(key, {}).get("type")
        if expected_type == "number" and isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                return False
            action.parameters[key] = value
        if expected_type == "number" and (
            isinstance(value, bool) or not isinstance(value, (int, float))
        ):
            return False

    return all(key in action.parameters for key in schema.get("required", []))

class BaseAgent(ABC):
    def __init__(self) -> None:
        self.messages: List[dict] = []
        self.final_answer = None
        self.question = None

    def reset(self) -> None:
        self.messages = self.construct_initial_messages()
        self.final_answer = None
        self.question = None

    @abstractmethod
    def construct_initial_messages(self) -> List[dict]:
        raise NotImplementedError

    @abstractmethod
    def run(self, question: str) -> Trajectory:
        raise NotImplementedError


class VideoSeekAgent(BaseAgent):
    def __init__(
        self,
        config: dict,
        video_path: str,
        subtitle_path: str,
        output_dir: str,
        tools: list,
        verbose: bool = False,
    ):
        super().__init__()
        self.config = config
        self.video_path = video_path
        self.vr = VideoReader(video_path)
        self.tool_registry = DEFAULT_TOOL_REGISTRY
        self.tools = self.tool_registry.resolve_tools(tools + ["answer"])
        self.output_dir = output_dir
        self.verbose = verbose

        self.duration = round(len(self.vr) / self.vr.get_avg_fps(), 2)
        self.subtitles = load_subtitles(subtitle_path)

        # LLM config
        self.model_name = config["model_name"]
        self.api_base = config["api_base"]
        self.api_key = config["api_key"]
        self.api_version = config["api_version"]
        self.max_steps = config["max_steps"]
        self.max_tokens = config["max_tokens"]
        self.reasoning_effort = config["reasoning_effort"]
        self.seed = config["seed"]
        self.temperature = config["temperature"]

        # initial messages
        self.messages = self.construct_initial_messages()
        # trajectory
        self.trajectory_steps: List[TrajectoryStep] = []

    def reset(self):
        """
        Reset the agent.
        """
        super().reset()
        self.trajectory_steps = []

    def construct_initial_messages(self) -> List[dict]:
        """
        Build the initial [system, user] messages.
        Subclasses should return a list of chat messages that may contain placeholders.
        """
        system_prompt = self.config["SYSTEM_PROMPT"].format(
            overview_num_frames=self.config["frame_sampling_factor"] * self.config["overview_base"],
            skim_num_frames=self.config["frame_sampling_factor"] * self.config["skim_base"],
            focus_num_frames=self.config["frame_sampling_factor"] * self.config["focus_base"],
        )
        return [{"role": "system", "content": system_prompt}]

    def __parse_actions(self, thought: str) -> List[Action]:
        """
        Parse the actions from the thought.
        """
        actions = []
        valid_tools = set(self.tool_registry.tools.keys())
        valid_tools.add("answer")

        if not thought:
            thought = "Analyze the video by invoking an appropriate perception tool."

        # 1. First attempt: parse tool calls directly embedded in thought text
        actions = extract_actions_from_text(thought, valid_tools)
        if actions:
            if all(
                normalize_action_parameters(action, self.tool_registry)
                for action in actions
            ):
                return actions
            actions = []

        # 2. Second attempt: API completion with tools
        answer_call_id = None
        try:
            response = call_llm_api(
                messages=[
                    {
                        "role": "user",
                        "content": f"Please call the appropriate tool(s) based on the following thought:\n{thought}",
                    }
                ],
                model_name=self.model_name,
                api_base=self.api_base,
                api_key=self.api_key,
                api_version=self.api_version,
                max_tokens=self.max_tokens,
                reasoning_effort=self.reasoning_effort,
                seed=self.seed,
                tool_choice="required",
                tools=self.tools,
                temperature=self.temperature,
            )

            msg_obj = response.choices[0].message if (response and getattr(response, "choices", None)) else None
            tool_calls = getattr(msg_obj, "tool_calls", None)
            if tool_calls is None and msg_obj is not None:
                if isinstance(msg_obj, dict):
                    tool_calls = msg_obj.get("tool_calls", [])
                elif hasattr(msg_obj, "json"):
                    try:
                        raw_j = msg_obj.json()
                        m_dict = json.loads(raw_j) if isinstance(raw_j, str) else raw_j
                        tool_calls = m_dict.get("tool_calls", [])
                    except Exception:
                        tool_calls = []
            if tool_calls is None:
                tool_calls = []

            for tool_idx, tool_call in enumerate(tool_calls):
                if isinstance(tool_call, dict):
                    func_data = tool_call.get("function", {})
                    function_name = func_data.get("name")
                    raw_args = func_data.get("arguments", "{}")
                    function_id = tool_call.get("id")
                else:
                    func_data = getattr(tool_call, "function", None)
                    function_name = getattr(func_data, "name", None) if func_data else None
                    raw_args = getattr(func_data, "arguments", "{}") if func_data else "{}"
                    function_id = getattr(tool_call, "id", None)

                parameters = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                if self.tool_registry.has_tool(function_name):
                    if function_name == "answer":
                        answer_call_id = tool_idx
                    action = Action(
                        function_name=function_name,
                        parameters=parameters,
                        function_id=function_id or f"{function_name}:0",
                    )
                    if normalize_action_parameters(action, self.tool_registry):
                        actions.append(action)

            if not actions and msg_obj:
                raw_text = get_message_content(response)
                actions = [
                    action
                    for action in extract_actions_from_text(raw_text, valid_tools)
                    if normalize_action_parameters(action, self.tool_registry)
                ]

        except Exception as e:
            err_msg = str(e)
            actions = extract_actions_from_text(err_msg, valid_tools)

        if len(actions) > 1 and answer_call_id is not None:
            actions.pop(answer_call_id)

        # 3. Fallback: Default to overview tool if no action detected
        if not actions:
            actions = [Action(function_name="overview", parameters={}, function_id="overview:0")]

        return actions

    def __exec_action(self, action: Action) -> str:
        """
        Execute an action.
        """
        function_name = getattr(action, "function_name", None) if action else None
        parameters = getattr(action, "parameters", {}) if action else {}

        if function_name == "answer":
            parameters = {"question": self.question, "messages": self.messages}
        else:
            parameters.update({"vr": self.vr, "subtitles": self.subtitles})
        
        if self.tool_registry.has_tool(function_name):
            outcome = self.tool_registry.get_function(function_name)(
                config=self.config, parameters=parameters
            )
            if outcome is None:
                outcome = "Tool execution failed."
            return outcome
        else:
            raise ValueError(f"Invalid function name: {function_name}")

    def run(self, question: str):
        self.reset()
        self.question = question
        subtitles_str = convert_to_free_form_text_representation(
            self.subtitles, content_type="subtitle"
        )

        ############################
        # Input
        ############################
        self.messages.append(
            {
                "role": "user",
                "content": (
                    f"Video Duration: {self.duration:.01f}s\n\n"
                    f"Video Subtitles:\n{subtitles_str}\n\n"
                    f"Question:\n{question}"
                ),
            }
        )

        video_id = self.video_path.split("/")[-1].split(".")[0]
        if self.verbose:
            print("--------------------------------")
            print(f"Video ID: {video_id} ({self.duration:.01f}s)")
            print("--------------------------------")
            print(f"Question:")
            print(question)
            print("--------------------------------")

        for step in range(self.max_steps):
            self.messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Step [{step + 1} / {self.max_steps}]: "
                        "Please follow the Thinking Policy to do **reasoning over the current state** "
                        "and **plan the next action(s) to take** following the Tool Calling Policy "
                        "and the Final Answer Policy. "
                        "No observation is needed to be provided in the response."
                    ),
                }
            )

            ############################################################
            # THOUGHT
            ############################################################
            response = call_llm_api(
                messages=self.messages,
                model_name=self.model_name,
                api_base=self.api_base,
                api_key=self.api_key,
                api_version=self.api_version,
                max_tokens=self.max_tokens,
                reasoning_effort=self.reasoning_effort,
                seed=self.seed,
                temperature=self.temperature,
            )
            thought = get_message_content(response)
            if not thought:
                thought = "I will proceed by running the overview tool to inspect the video timeline."
            self.messages.append({"role": "assistant", "content": thought})
            if self.verbose:
                print(f"[STEP {step+1} / {self.max_steps}] THOUGHT")
                print(thought)
                print("--------------------------------")

            ############################################################
            # ACTIONS
            ############################################################
            actions = self.__parse_actions(thought)
            if self.verbose:
                print(f"[STEP {step+1} / {self.max_steps}] ACTIONS")
                print([str(action) for action in actions])
                print("--------------------------------")
            if len(actions) != 0 and actions[0].function_name != "answer":
                self.messages[-1]["tool_calls"] = [
                    {
                        "id": action.function_id,
                        "type": "function",
                        "function": {
                            "name": action.function_name,
                            "arguments": json.dumps(action.parameters),
                        },
                    }
                    for action in actions
                ]

            ############################################################
            # OBSERVATIONS
            ############################################################
            for action in actions:
                try:
                    outcome = self.__exec_action(action)
                except Exception as e:
                    outcome = f"Tool execution failed: {type(e).__name__}: {e}"
                if self.verbose:
                    print(f"[STEP {step+1} / {self.max_steps}] OBSERVATION")
                    print(outcome)
                    print("--------------------------------")
                observation = Observation(action=action, outcome=outcome)
                if action.parameters is not None:
                    action.parameters.pop("vr", None)
                    action.parameters.pop("subtitles", None)
                self.trajectory_steps.append(
                    TrajectoryStep(
                        step_id=step + 1,
                        thought=thought,
                        action=action,
                        observation=observation,
                    )
                )
                if action.function_name == "answer":
                    self.final_answer = observation.outcome
                    break
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": action.function_id,
                        "content": f"Observation from `{str(action.to_dict())}`:\n{outcome}",
                    }
                )
            
            if len(actions) == 0:
                self.messages.append(
                    {
                        "role": "user",
                        "content": "There is no function call in your response. YOU MUST USE A FUNCTION CALL IN EACH RESPONSE.",
                    }
                )
                continue

            ############################################################
            # STOP IF FINAL ANSWER IS FOUND
            ############################################################
            if self.final_answer is not None:
                break

        ############################################################
        # REACH MAX STEPS BUT NO FINAL ANSWER IS FOUND
        ############################################################
        if self.final_answer is None:
            self.messages.append(
                {
                    "role": "user",
                    "content": (
                        "You have reached the maximum number of steps. "
                        f"Question:\n{question}\n\n"
                        "If the question is a multiple-choice question, please directly answer with the option's letter from the given choices without any additional text."
                    ),
                }
            )
            response = call_llm_api(
                messages=self.messages,
                model_name=self.model_name,
                api_base=self.api_base,
                api_key=self.api_key,
                api_version=self.api_version,
                max_tokens=self.max_tokens,
                reasoning_effort=self.reasoning_effort,
                seed=self.seed,
                temperature=self.temperature,
            )
            self.final_answer = response.choices[0].message.content
            self.messages.append({"role": "assistant", "content": self.final_answer})
            return Trajectory(
                question=question,
                steps=self.trajectory_steps,
                final_answer=self.final_answer,
                finish_reason="reach_max_steps",
            )

        return Trajectory(
            question=question,
            steps=self.trajectory_steps,
            final_answer=self.final_answer,
            finish_reason="stop",
        )