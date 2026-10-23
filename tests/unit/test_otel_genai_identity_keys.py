"""Tests that GenAI-standard identity keys are emitted next to TruLens keys."""

import unittest

from opentelemetry import baggage
from opentelemetry import context as context_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from trulens.experimental.otel_tracing.core.span import (
    set_general_span_attributes,
)
from trulens.otel.semconv.trace import GenAIAttributes
from trulens.otel.semconv.trace import ResourceAttributes
from trulens.otel.semconv.trace import SpanAttributes


class TestGenAIIdentityKeysOnSpans(unittest.TestCase):
    def setUp(self) -> None:
        self.exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        self.tracer = provider.get_tracer(__name__)

    def _span_attributes(self, **entries) -> dict:
        ctx = context_api.get_current()
        for key, value in entries.items():
            ctx = baggage.set_baggage(key, value, ctx)
        with self.tracer.start_as_current_span("s", context=ctx) as span:
            set_general_span_attributes(
                span, SpanAttributes.SpanType.UNKNOWN, ctx
            )
        return dict(self.exporter.get_finished_spans()[0].attributes)

    def test_both_key_sets_are_written(self) -> None:
        attrs = self._span_attributes(**{
            SpanAttributes.CONVERSATION_ID: "c1",
            GenAIAttributes.CONVERSATION.ID: "c1",
            ResourceAttributes.APP_NAME: "app",
            GenAIAttributes.AGENT.NAME: "app",
            ResourceAttributes.APP_VERSION: "v1",
            GenAIAttributes.AGENT.VERSION: "v1",
            ResourceAttributes.APP_ID: "id1",
            GenAIAttributes.AGENT.ID: "id1",
        })

        self.assertEqual("c1", attrs[SpanAttributes.CONVERSATION_ID])
        self.assertEqual("c1", attrs[GenAIAttributes.CONVERSATION.ID])
        self.assertEqual("app", attrs[ResourceAttributes.APP_NAME])
        self.assertEqual("app", attrs[GenAIAttributes.AGENT.NAME])
        self.assertEqual("v1", attrs[GenAIAttributes.AGENT.VERSION])
        self.assertEqual("id1", attrs[GenAIAttributes.AGENT.ID])

    def test_keys_are_absent_when_unset(self) -> None:
        attrs = self._span_attributes()

        for key in (
            GenAIAttributes.CONVERSATION.ID,
            GenAIAttributes.AGENT.NAME,
            GenAIAttributes.AGENT.VERSION,
            GenAIAttributes.AGENT.ID,
        ):
            self.assertNotIn(key, attrs)


if __name__ == "__main__":
    unittest.main()
