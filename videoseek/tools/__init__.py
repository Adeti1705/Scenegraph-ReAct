from copy import deepcopy

from .overview import overview_tool, execute_overview
from .skim import skim_tool, execute_skim
from .focus import focus_tool, execute_focus
from .answer import answer_tool, execute_answer
from .registry import ToolRegistry


from .scenegraph_tools import (
    execute_overview_scenegraph,
    execute_skim_scenegraph,
    execute_focus_scenegraph,
    execute_answer_scenegraph,
)

TOOLS = {
    "overview": overview_tool,
    "skim": skim_tool,
    "focus": focus_tool,
    "answer": answer_tool,
}


TOOL_FUNCTIONS = {
    "overview": execute_overview,
    "skim": execute_skim,
    "focus": execute_focus,
    "answer": execute_answer,
}

SCENEGRAPH_TOOL_FUNCTIONS = {
    "overview": execute_overview_scenegraph,
    "skim": execute_skim_scenegraph,
    "focus": execute_focus_scenegraph,
    "answer": execute_answer_scenegraph,
}

SCENEGRAPH_TOOLS = deepcopy(TOOLS)
SCENEGRAPH_TOOLS["overview"]["function"]["description"] = (
    "Scan the full video to build a visually grounded scene-graph object inventory and scene context."
)
SCENEGRAPH_TOOLS["skim"]["function"]["description"] = (
    "Scan a video interval for object interactions and estimate relationship frame spans."
)
SCENEGRAPH_TOOLS["focus"]["function"]["description"] = (
    "Inspect a video interval of at most 4 seconds to verify object identities, predicates, and relationship boundaries."
)
SCENEGRAPH_TOOLS["answer"]["function"]["description"] = (
    "Consolidate video observations into a PVSG-format spatio-temporal scene graph."
)
for tool_name in ("skim", "focus"):
    properties = SCENEGRAPH_TOOLS[tool_name]["function"]["parameters"]["properties"]
    properties["query"]["description"] = (
        "A concise, visually verifiable object or relationship question for this interval."
    )
    properties["start_time"]["description"] = "Interval start in seconds from the beginning of the video."
    properties["end_time"]["description"] = "Interval end in seconds from the beginning of the video."

DEFAULT_TOOL_REGISTRY = ToolRegistry(tools=TOOLS, tool_functions=TOOL_FUNCTIONS)
SCENEGRAPH_TOOL_REGISTRY = ToolRegistry(tools=SCENEGRAPH_TOOLS, tool_functions=SCENEGRAPH_TOOL_FUNCTIONS)