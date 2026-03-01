from agentci.models import Trace, Span, SpanKind
from agentci.capture import TraceContext
import time

def run(query: str) -> Trace:
    with TraceContext(test_name="demo", agent_name="dummy_agent") as ctx:
        # Manually construct a trace since we aren't using an actual LLM client that the patcher catches
        from agentci.models import ToolCall
        tool_span = Span(name="dummy_tool", kind=SpanKind.TOOL_CALL, input_data={"q": query})
        tool_span.tool_calls.append(ToolCall(tool_name="dummy_tool", arguments={"q": query}))
        ctx.trace.spans.append(tool_span)
        
        output_span = Span(name="output", kind=SpanKind.AGENT, output_data=f"Processed: {query}")
        ctx.trace.spans.append(output_span)
        
        ctx.trace.compute_metrics()
        
    return ctx.trace
