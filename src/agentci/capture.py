"""
Trace capture via monkey-patching.

Strategy: Wrap the OpenAI/Anthropic client's .create() methods to
automatically record every LLM call and tool invocation into a Trace
object. The developer doesn't change their agent code at all.

Phase 1: Patch openai.ChatCompletion and anthropic.Messages
Phase 2: Add OTEL span emission for interop with Arize/Langfuse
"""

import time
import contextvars
from .models import Trace, Span, LLMCall, ToolCall, SpanKind
from .cost import compute_cost

# Global context var — allows nested agent calls to share a trace
_active_trace: contextvars.ContextVar[Trace | None] = contextvars.ContextVar(
    '_active_trace', default=None
)
_active_span: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
    '_active_span', default=None
)


class TraceContext:
    """
    Context manager that captures all LLM/tool activity into a Trace.
    
    Usage:
        with TraceContext(agent_name="booking_agent") as ctx:
            result = my_agent.run("Book a flight to NYC")
            trace = ctx.trace
    """
    
    def __init__(self, agent_name: str = "", test_name: str = ""):
        self.trace = Trace(agent_name=agent_name, test_name=test_name)
        self._patches = []
        self._start_time = 0.0
    
    def __enter__(self):
        # Create root span
        root_span = Span(kind=SpanKind.AGENT, name=self.trace.agent_name)
        self.trace.spans.append(root_span)
        
        # Set context vars
        _active_trace.set(self.trace)
        _active_span.set(root_span)
        
        # Apply monkey patches
        self._patch_openai()
        self._patch_anthropic()
        
        self._start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        # Compute duration
        self.trace.total_duration_ms = (time.perf_counter() - self._start_time) * 1000
        
        # Roll up metrics
        self.trace.compute_metrics()
        
        # Remove patches
        for restore_fn in self._patches:
            restore_fn()
        
        # Clear context
        _active_trace.set(None)
        _active_span.set(None)
    
    def _patch_openai(self):
        """Wrap openai.chat.completions.create to capture LLM calls."""
        try:
            import openai  # type: ignore
            original_create = openai.resources.chat.completions.Completions.create
            
            def patched_create(self_client, *args, **kwargs):
                start = time.perf_counter()
                response = original_create(self_client, *args, **kwargs)
                duration = (time.perf_counter() - start) * 1000
                
                span = _active_span.get()
                if span is not None:
                    model = kwargs.get('model', getattr(response, 'model', ''))
                    usage = getattr(response, 'usage', None)
                    tokens_in = getattr(usage, 'prompt_tokens', 0) if usage else 0
                    tokens_out = getattr(usage, 'completion_tokens', 0) if usage else 0
                    
                    llm_call = LLMCall(
                        model=model,
                        provider="openai",
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        cost_usd=compute_cost("openai", model, tokens_in, tokens_out),
                        duration_ms=duration,
                    )
                    span.llm_calls.append(llm_call)
                    
                    # Capture tool calls from response
                    choices = getattr(response, 'choices', [])
                    if choices:
                        message = choices[0].message
                        tool_calls = getattr(message, 'tool_calls', None)
                        if tool_calls:
                            for tc in tool_calls:
                                import json
                                span.tool_calls.append(ToolCall(
                                    tool_name=tc.function.name,
                                    arguments=json.loads(tc.function.arguments),
                                ))
                
                return response
            
            openai.resources.chat.completions.Completions.create = patched_create
            self._patches.append(
                lambda: setattr(
                    openai.resources.chat.completions.Completions, 
                    'create', 
                    original_create
                )
            )
        except ImportError:
            pass  # OpenAI not installed — skip silently
    
    def _patch_anthropic(self):
        """Wrap anthropic.messages.create to capture LLM calls."""
        try:
            import anthropic  # type: ignore
            original_create = anthropic.resources.messages.Messages.create
            
            def patched_create(self_client, *args, **kwargs):
                start = time.perf_counter()
                response = original_create(self_client, *args, **kwargs)
                duration = (time.perf_counter() - start) * 1000
                
                span = _active_span.get()
                if span is not None:
                    model = kwargs.get('model', getattr(response, 'model', ''))
                    usage = getattr(response, 'usage', None)
                    tokens_in = getattr(usage, 'input_tokens', 0) if usage else 0
                    tokens_out = getattr(usage, 'output_tokens', 0) if usage else 0
                    
                    llm_call = LLMCall(
                        model=model,
                        provider="anthropic",
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        cost_usd=compute_cost("anthropic", model, tokens_in, tokens_out),
                        duration_ms=duration,
                    )
                    span.llm_calls.append(llm_call)
                    
                    # Capture tool use blocks
                    for block in getattr(response, 'content', []):
                        if getattr(block, 'type', '') == 'tool_use':
                            span.tool_calls.append(ToolCall(
                                tool_name=block.name,
                                arguments=block.input if isinstance(block.input, dict) else {},
                            ))
                
                return response
            
            anthropic.resources.messages.Messages.create = patched_create
            self._patches.append(
                lambda: setattr(
                    anthropic.resources.messages.Messages, 
                    'create', 
                    original_create
                )
            )
        except ImportError:
            pass
