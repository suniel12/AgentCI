"""
Lightweight mock tools for zero-API-key testing.

Developers define mock responses in YAML or Python.
The demo agent ships with these pre-configured.
"""

from typing import Any, Callable
from agentci.models import ToolCall


class MockTool:
    """
    A fake tool that returns predefined responses.
    
    Usage:
        search = MockTool(
            name="search_flights",
            responses={
                "default": {"flights": [{"id": 1, "price": 350}]},
                "no_results": {"flights": []},
            }
        )
        
        # In agent code, replace real tool with mock:
        result = search.call(origin="SFO", destination="JFK")
    """
    
    def __init__(
        self, 
        name: str, 
        responses: dict[str, Any] | None = None,
        handler: Callable[..., Any] | None = None,
        stateful: bool = False,
    ):
        self.name = name
        self.responses = responses or {"default": {}}
        self.handler = handler
        self.stateful = stateful
        self._state: dict[str, Any] = {}
        self._call_history: list[dict[str, Any]] = []
        self._scenario: str = "default"
    
    def set_scenario(self, scenario: str) -> None:
        """Switch to a named response scenario."""
        self._scenario = scenario
    
    def call(self, **kwargs) -> Any:
        """Execute the mock tool, recording the call."""
        self._call_history.append({"arguments": kwargs})
        
        if self.handler:
            assert self.handler is not None
            return self.handler(**kwargs, _state=self._state)
        
        return self.responses.get(self._scenario, self.responses["default"])
    
    @property
    def call_count(self) -> int:
        return len(self._call_history)
    
    def reset(self) -> None:
        self._call_history.clear()
        self._state.clear()
        self._scenario = "default"


class MockToolkit:
    """
    A collection of mock tools loaded from YAML.
    
    mocks.yaml:
        search_flights:
            default:
                flights:
                    - id: 1
                      price: 350
                      airline: "United"
            no_results:
                flights: []
        
        book_flight:
            default:
                confirmation: "ABC123"
                status: "confirmed"
    """
    
    def __init__(self):
        self.tools: dict[str, MockTool] = {}
    
    @classmethod
    def from_yaml(cls, path: str) -> "MockToolkit":
        # yaml is a dependency, so we import it at top level or here
        import yaml
        toolkit = cls()
        with open(path) as f:
            config = yaml.safe_load(f)
        
        for tool_name, responses in config.items():
            toolkit.tools[tool_name] = MockTool(
                name=tool_name,
                responses=responses,
            )
        
        return toolkit
    
    def get(self, name: str) -> MockTool:
        if name not in self.tools:
            raise KeyError(f"Mock tool '{name}' not found. Available: {list(self.tools)}")
        return self.tools[name]
    
    def set_all_scenarios(self, scenario: str) -> None:
        for tool in self.tools.values():
            tool.set_scenario(scenario)
    
    def reset_all(self) -> None:
        for tool in self.tools.values():
            tool.reset()
