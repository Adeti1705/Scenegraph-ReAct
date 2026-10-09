import json
import math
import yaml
from pathlib import Path
from typing import List

from decord import VideoReader

from .agent import BaseAgent, extract_actions_from_text, normalize_action_parameters
from .core import Action, Observation, Trajectory, TrajectoryStep
from .tools import SCENEGRAPH_TOOL_REGISTRY
from .utils import call_llm_api, convert_to_free_form_text_representation, get_message_content, load_subtitles


CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

def load_scenegraph_prompts():
    prompt_path = CONFIG_DIR / "prompts_scenegraph.yaml"
    if prompt_path.exists():
        with open(prompt_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


class SceneGraphAgent(BaseAgent):
    def __init__(
        self,
        config: dict,
        video_path: str,
        subtitle_path: str = None,
        output_dir: str = "./output/scenegraph",
        tools: list = None,
        verbose: bool = False,
    ):
        super().__init__()
        self.config = config
        self.config.setdefault("scenegraph_skim_base", 4)
        self.config.setdefault("scenegraph_focus_base", 4)
        self.video_path = video_path
        self.vr = VideoReader(video_path)
        self.frame_count = len(self.vr)
        if self.frame_count == 0:
            raise ValueError(f"Video contains no frames: {video_path}")
        self.source_fps = float(self.vr.get_avg_fps())
        if not math.isfinite(self.source_fps) or self.source_fps <= 0:
            raise ValueError(f"Video has invalid source FPS {self.source_fps!r}: {video_path}")
        self.video_id = Path(video_path).stem
        self.tool_registry = SCENEGRAPH_TOOL_REGISTRY
        tool_names = (tools or ["overview", "skim", "focus"]) + ["answer"]
        self.tools = self.tool_registry.resolve_tools(tool_names)
        self.output_dir = output_dir
        self.verbose = verbose

        exact_duration = self.frame_count / self.source_fps
        self.duration = round(exact_duration, 2)
        annotation_fps = float(config.get("scenegraph_fps", 5))
        if not math.isfinite(annotation_fps) or annotation_fps <= 0:
            raise ValueError("scenegraph_fps must be a finite number greater than zero")
        height, width = self.vr[0].asnumpy().shape[:2]
        self.video_meta = {
            "height": height,
            "width": width,
            "fps": annotation_fps,
            "duration": self.duration,
            "num_frames": round(exact_duration * annotation_fps),
        }
        self.subtitles = load_subtitles(subtitle_path)

        # Prompts
        sg_prompts = load_scenegraph_prompts()
        self.system_prompt_template = sg_prompts.get("SCENEGRAPH_SYSTEM_PROMPT", "")

        # LLM config
        self.model_name = config["model_name"]
        self.api_base = config["api_base"]
        self.api_key = config["api_key"]
        self.api_version = config.get("api_version")
        self.max_steps = config.get("max_steps", 10)
        if self.max_steps < 3:
            raise ValueError("max_steps must be at least three for overview, skim, and answer")
        self.max_tokens = config.get("max_tokens", 4096)
        self.reasoning_effort = config.get("reasoning_effort")
        self.seed = config.get("seed", 42)
        self.temperature = config.get("temperature", 0.2)

        self.messages = self.construct_initial_messages()
        self.trajectory_steps: List[TrajectoryStep] = []

    def reset(self):
        super().reset()
        self.trajectory_steps = []

    def construct_initial_messages(self) -> List[dict]:
        return [{"role": "system", "content": self.system_prompt_template}]

    def __parse_actions(self, thought: str) -> List[Action]:
        actions = []
        valid_tools = set(self.tool_registry.tools.keys())
        valid_tools.add("answer")

        if not thought:
            thought = "Extract scene graph entities and relationships by running the overview tool."

        # 1. Parse from thought text
        actions = [
            action
            for action in extract_actions_from_text(thought, valid_tools)
            if normalize_action_parameters(action, self.tool_registry)
        ]
        if actions:
            return actions[:1]

        # 2. Try LLM tool completion
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

            for tool_call in tool_calls:
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

        return actions[:1]

    def __fallback_action(self) -> tuple[Action, str]:
        executed_tools = {step.action.function_name for step in self.trajectory_steps}
        if "overview" not in executed_tools:
            action = Action(function_name="overview", parameters={}, function_id="overview:0")
            thought = "Fallback: no valid tool call was extracted, so run overview to inventory the video."
        elif "skim" not in executed_tools:
            action = Action(
                function_name="skim",
                parameters={
                    "start_time": 0.0,
                    "end_time": self.duration,
                    "query": "Extract active relationship triplets across the video",
                },
                function_id="skim:0",
            )
            thought = "Fallback: no valid tool call was extracted, so skim the full video for relationships."
        elif "focus" not in executed_tools:
            action = Action(
                function_name="focus",
                parameters={
                    "start_time": 0.0,
                    "end_time": min(4.0, self.duration),
                    "query": "Verify a visually supported relationship in this short clip",
                },
                function_id="focus:0",
            )
            thought = "Fallback: no valid tool call was extracted, so focus on an initial short clip."
        else:
            action = Action(function_name="answer", parameters={}, function_id="answer:0")
            thought = "Fallback: no valid tool call was extracted, so finalize from the available observations."
        return action, thought

    def __scheduled_action(self, step: int) -> Action | None:
        if self.max_steps == 1:
            return Action(function_name="answer", parameters={}, function_id="answer:0")
        if step == 0:
            return Action(function_name="overview", parameters={}, function_id="overview:0")
        if step == self.max_steps - 1:
            return Action(function_name="answer", parameters={}, function_id="answer:0")

        refinement_slots = 1 if self.max_steps >= 4 else 0
        coverage_bins = self.max_steps - 2 - refinement_slots
        if coverage_bins > 0 and 1 <= step <= coverage_bins:
            bin_index = step - 1
            start_time = self.duration * bin_index / coverage_bins
            end_time = self.duration * (bin_index + 1) / coverage_bins
            return Action(
                function_name="skim",
                parameters={
                    "query": "Find all visually supported ontology-valid relationships and newly appearing objects in this interval. Report conservative relation start/end seconds.",
                    "start_time": start_time,
                    "end_time": end_time,
                },
                function_id=f"skim:coverage:{bin_index}",
            )
        return None

    def __exec_action(self, action: Action) -> str:
        function_name = getattr(action, "function_name", None) if action else None
        parameters = getattr(action, "parameters", {}) if action else {}

        if function_name == "answer":
            parameters = {
                "messages": self.messages,
                "video_id": self.video_id,
                "meta": self.video_meta,
                "object_categories": self.config.get("scenegraph_object_categories", {}),
                "predicates": self.config.get("scenegraph_predicates", []),
            }
        else:
            parameters.update({"vr": self.vr, "subtitles": self.subtitles})

        if self.tool_registry.has_tool(function_name):
            outcome = self.tool_registry.get_function(function_name)(
                config=self.config, parameters=parameters
            )
            if outcome is None:
                outcome = "Tool execution failed."
            return outcome
        raise ValueError(f"Invalid function name: {function_name}")

    def run(self, query_instruction: str = "Generate a complete spatio-temporal scene graph for this video.") -> Trajectory:
        self.reset()
        self.question = query_instruction
        subtitles_str = convert_to_free_form_text_representation(self.subtitles, content_type="subtitle")

        self.messages.append(
            {
                "role": "user",
                "content": (
                    f"Video Duration: {self.duration:.01f}s\n"
                    f"Annotation FPS: {self.video_meta['fps']}\n"
                    f"Annotation Frame Count: {self.video_meta['num_frames']}\n\n"
                    f"Video Subtitles:\n{subtitles_str}\n\n"
                    f"Task Goal:\n{query_instruction}"
                ),
            }
        )

        if self.verbose:
            print("========================================")
            print(f"SceneGraph Agent | Video: {self.video_id} ({self.duration:.1f}s)")
            print("========================================")

        for step in range(self.max_steps):
            scheduled_action = self.__scheduled_action(step)
            step_instruction = (
                f"Step [{step + 1} / {self.max_steps}]: "
                "Plan the next action to extract entities, relationships, or construct the final Scene Graph. "
                "Choose ONE tool: `overview`, `skim`, `focus`, or `answer`."
            )
            if scheduled_action is not None:
                step_instruction += f" Follow the required coverage action: {scheduled_action.function_name} {scheduled_action.parameters}."
            self.messages.append(
                {
                    "role": "user",
                    "content": step_instruction,
                }
            )

            executed_tools = [s.action.function_name for s in self.trajectory_steps]
            if scheduled_action is not None:
                if scheduled_action.function_name == "skim":
                    thought = (
                        "Required coverage scan: inspect the full interval "
                        f"{scheduled_action.parameters['start_time']:.2f}s to "
                        f"{scheduled_action.parameters['end_time']:.2f}s before finalizing."
                    )
                else:
                    thought = f"Required coverage action: call `{scheduled_action.function_name}`."
            else:
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

            if not thought and scheduled_action is None:
                if "overview" not in executed_tools:
                    thought = "I will call `overview` to build the global entity inventory."
                elif "skim" not in executed_tools:
                    thought = f"I will call `skim` to scan from 0.0s to {self.duration:.1f}s for relationship triplets."
                elif "focus" not in executed_tools:
                    thought = "I will call `focus` to verify fine-grained contact predicates."
                else:
                    thought = "I will call `answer` to consolidate the final spatio-temporal scene graph JSON."

            self.messages.append({"role": "assistant", "content": thought})

            if self.verbose:
                print(f"[STEP {step+1} / {self.max_steps}] THOUGHT\n{thought}\n" + "-"*40)

            if scheduled_action is not None:
                actions = [scheduled_action]
            else:
                actions = self.__parse_actions(thought)
                if not actions:
                    action, thought = self.__fallback_action()
                    self.messages[-1]["content"] = thought
                    actions = [action]

            if self.verbose:
                print(f"[STEP {step+1} / {self.max_steps}] ACTIONS\n{[str(a) for a in actions]}\n" + "-"*40)

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

            for action in actions:
                try:
                    outcome = self.__exec_action(action)
                except Exception as e:
                    outcome = f"Tool execution error: {e}"

                if self.verbose:
                    print(f"[STEP {step+1} / {self.max_steps}] OBSERVATION\n{outcome}\n" + "-"*40)

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
                    self.final_answer = outcome
                    break

                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": action.function_id,
                        "content": f"Observation from `{str(action.to_dict())}`:\n{outcome}",
                    }
                )

            if self.final_answer is not None:
                break

        return Trajectory(
            question=self.question,
            steps=self.trajectory_steps,
            final_answer=self.final_answer or "",
            finish_reason="answer_tool" if self.final_answer else "max_steps",
        )