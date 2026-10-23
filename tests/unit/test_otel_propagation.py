"""Tests for W3C trace context propagation on outbound HTTP requests.

TruLens instruments the caller, but the provider runs on the far side of an
HTTP call, so without propagation the provider's server span starts a separate
trace. These cover the helper that writes the active span's context and the
sync and async httpx transports that apply it to every outbound request.
"""

import asyncio
import os
import unittest
from unittest import mock

import httpx
from opentelemetry import baggage
from opentelemetry import context as context_api
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.context import Context
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.textmap import TextMapPropagator
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import format_trace_id
from opentelemetry.trace import get_current_span
from trulens.core.otel import propagation


class _NoOpPropagator(TextMapPropagator):
    """A propagator that is valid but writes nothing, like propagation off."""

    @property
    def fields(self) -> set:
        return set()

    def extract(
        self, carrier: object, context: object = None, getter: object = None
    ) -> Context:
        return context or Context()

    def inject(
        self, carrier: object, context: object = None, setter: object = None
    ) -> None:
        return None


class _TracingTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = TracerProvider()
        self.provider.add_span_processor(
            SimpleSpanProcessor(InMemorySpanExporter())
        )
        self.tracer = self.provider.get_tracer(__name__)

    def current_trace_id(self) -> str:
        """Format the active span's trace id for asserting on headers."""
        return format_trace_id(get_current_span().get_span_context().trace_id)


class TestInjectTraceContext(_TracingTestCase):
    def test_injects_active_span_context(self) -> None:
        with self.tracer.start_as_current_span("outbound"):
            headers = {}
            trace_id = self.current_trace_id()
            self.assertTrue(propagation.inject_trace_context(headers))

        self.assertIn("traceparent", headers)
        self.assertIn(trace_id, headers["traceparent"])

    def test_injects_nothing_without_an_active_span(self) -> None:
        headers = {}
        self.assertFalse(propagation.inject_trace_context(headers))
        self.assertEqual({}, headers)

    def test_preserves_existing_headers(self) -> None:
        with self.tracer.start_as_current_span("outbound"):
            headers = {"authorization": "Bearer test"}
            propagation.inject_trace_context(headers)

        self.assertEqual("Bearer test", headers["authorization"])


_TRACE_ID = "07f8c88f000f3854aedaf55fa46a8b71"
_SPAN_ID = "e9e48039ccad0456"
_TRACEPARENT = f"00-{_TRACE_ID}-{_SPAN_ID}-01"


class TestExtractTraceContext(_TracingTestCase):
    def test_extracts_a_remote_context(self) -> None:
        context = propagation.extract_trace_context({
            "traceparent": _TRACEPARENT
        })

        self.assertIsNotNone(context)
        self.assertEqual(_TRACE_ID, format_trace_id(context.trace_id))
        self.assertEqual(int(_SPAN_ID, 16), context.span_id)

    def test_marks_the_context_remote_so_it_can_parent_local_spans(
        self,
    ) -> None:
        context = propagation.extract_trace_context({
            "traceparent": _TRACEPARENT
        })

        self.assertTrue(context.is_remote)

    def test_carries_the_sampled_flag_through(self) -> None:
        sampled = propagation.extract_trace_context({
            "traceparent": _TRACEPARENT
        })
        unsampled = propagation.extract_trace_context({
            "traceparent": f"00-{_TRACE_ID}-{_SPAN_ID}-00"
        })

        self.assertTrue(sampled.trace_flags.sampled)
        self.assertFalse(unsampled.trace_flags.sampled)

    def test_returns_none_for_an_empty_carrier(self) -> None:
        self.assertIsNone(propagation.extract_trace_context({}))

    def test_returns_none_for_malformed_values(self) -> None:
        for value in (
            "not-a-traceparent",
            f"00-{_TRACE_ID}",
            f"00-07f8c88f-{_SPAN_ID}-01",
            "",
        ):
            with self.subTest(traceparent=value):
                self.assertIsNone(
                    propagation.extract_trace_context({"traceparent": value})
                )

    def test_returns_none_for_all_zero_ids(self) -> None:
        for value in (
            f"00-{'0' * 32}-{_SPAN_ID}-01",
            f"00-{_TRACE_ID}-{'0' * 16}-01",
        ):
            with self.subTest(traceparent=value):
                self.assertIsNone(
                    propagation.extract_trace_context({"traceparent": value})
                )

    def test_ignores_an_uppercase_key(self) -> None:
        # Environment variable names have to be lowercased by the caller,
        # because the W3C propagator only looks for the lowercase header name.
        self.assertIsNone(
            propagation.extract_trace_context({"TRACEPARENT": _TRACEPARENT})
        )

    def test_does_not_fall_through_to_the_ambient_span(self) -> None:
        # Extraction reads the carrier and nothing else. A carrier with no
        # usable context must not report the locally active span as inherited.
        with self.tracer.start_as_current_span("local"):
            self.assertIsNone(propagation.extract_trace_context({}))

    def test_extracts_nothing_when_propagation_is_disabled(self) -> None:
        from opentelemetry.propagate import get_global_textmap

        self.assertIsNotNone(
            propagation.extract_trace_context({"traceparent": _TRACEPARENT})
        )

        original = get_global_textmap()
        set_global_textmap(_NoOpPropagator())
        self.addCleanup(set_global_textmap, original)

        self.assertIsNone(
            propagation.extract_trace_context({"traceparent": _TRACEPARENT})
        )

    def test_reports_no_injection_when_propagator_writes_nothing(self) -> None:
        # A valid span context is not enough on its own: with a no-op
        # propagator installed the header mapping stays untouched, and the
        # return value has to say so.
        from opentelemetry.propagate import get_global_textmap

        original = get_global_textmap()
        set_global_textmap(_NoOpPropagator())
        self.addCleanup(set_global_textmap, original)

        with self.tracer.start_as_current_span("outbound"):
            headers = {}
            self.assertFalse(propagation.inject_trace_context(headers))

        self.assertEqual({}, headers)


class TestTracingTransport(_TracingTestCase):
    def _client_recording_headers(self) -> tuple:
        recorded = {}

        def handler(request: httpx.Request) -> httpx.Response:
            recorded.update(request.headers)
            return httpx.Response(200, json={"ok": True})

        transport = propagation.tracing_transport(httpx.MockTransport(handler))
        return httpx.Client(transport=transport), recorded

    def test_transport_injects_on_outbound_request(self) -> None:
        client, recorded = self._client_recording_headers()
        with self.tracer.start_as_current_span("outbound"):
            trace_id = self.current_trace_id()
            client.get("https://example.invalid/v1/models")

        self.assertIn("traceparent", recorded)
        self.assertIn(trace_id, recorded["traceparent"])

    def test_transport_sends_request_without_an_active_span(self) -> None:
        # A client with the transport must still work outside a trace.
        client, recorded = self._client_recording_headers()
        response = client.get("https://example.invalid/v1/models")

        self.assertEqual(200, response.status_code)
        self.assertNotIn("traceparent", recorded)

    def test_close_is_delegated_to_the_inner_transport(self) -> None:
        # The inner transport owns the connection pool, so closing the wrapper
        # has to close the inner one too.
        inner = _CloseSpy()
        transport = propagation.tracing_transport(inner)
        transport.close()

        self.assertTrue(inner.closed)

    def test_close_reaches_the_inner_transport_through_client_close(
        self,
    ) -> None:
        inner = _CloseSpy()
        client = httpx.Client(transport=propagation.tracing_transport(inner))
        client.close()

        self.assertTrue(inner.closed)


class TestAsyncTracingTransport(_TracingTestCase):
    def _client_recording_headers(self) -> tuple:
        recorded = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            recorded.update(request.headers)
            return httpx.Response(200, json={"ok": True})

        transport = propagation.async_tracing_transport(
            httpx.MockTransport(handler)
        )
        return httpx.AsyncClient(transport=transport), recorded

    def test_transport_injects_on_outbound_request(self) -> None:
        client, recorded = self._client_recording_headers()

        async def request() -> None:
            with self.tracer.start_as_current_span("outbound"):
                trace_id = self.current_trace_id()
                await client.get("https://example.invalid/v1/models")
            self.recorded_trace_id = trace_id

        asyncio.run(request())
        asyncio.run(client.aclose())

        self.assertIn("traceparent", recorded)
        self.assertIn(self.recorded_trace_id, recorded["traceparent"])

    def test_transport_sends_request_without_an_active_span(self) -> None:
        client, recorded = self._client_recording_headers()

        async def request() -> httpx.Response:
            return await client.get("https://example.invalid/v1/models")

        response = asyncio.run(request())
        asyncio.run(client.aclose())

        self.assertEqual(200, response.status_code)
        self.assertNotIn("traceparent", recorded)

    def test_aclose_is_delegated_to_the_inner_transport(self) -> None:
        inner = _CloseSpy()
        transport = propagation.async_tracing_transport(inner)
        asyncio.run(transport.aclose())

        self.assertTrue(inner.closed)


_CONV = "gen_ai.conversation.id"
_AGENT = "gen_ai.agent.name"


class TestBaggagePropagation(_TracingTestCase):
    """Baggage leaves the process only to allowed hosts and only as allowlisted
    GenAI keys."""

    def setUp(self) -> None:
        super().setUp()
        env = mock.patch.dict(os.environ, clear=False)
        env.start()
        self.addCleanup(env.stop)
        for name in (
            propagation.PROPAGATION_HOSTS_ENV_VAR,
            *propagation._DEFAULT_HOST_ENV_VARS,
        ):
            os.environ.pop(name, None)

    def inject(self, url, **entries) -> dict:
        ctx = context_api.get_current()
        for key, value in entries.items():
            ctx = baggage.set_baggage(key.replace("__", "."), value, ctx)
        token = context_api.attach(ctx)
        try:
            with self.tracer.start_as_current_span("outbound"):
                headers = {}
                propagation.inject_trace_context(headers, url)
        finally:
            context_api.detach(token)
        return headers

    def test_no_baggage_by_default(self) -> None:
        headers = self.inject(
            "https://api.example.com/v1", gen_ai__conversation__id="c"
        )
        self.assertIn("traceparent", headers)
        self.assertNotIn("baggage", headers)

    def test_no_baggage_without_a_target_url(self) -> None:
        os.environ["TRULENS_PROPAGATION_HOSTS"] = "api.example.com"
        headers = self.inject(None, gen_ai__conversation__id="c")
        self.assertNotIn("baggage", headers)

    def test_allowed_host_gets_only_allowlisted_keys(self) -> None:
        os.environ["TRULENS_PROPAGATION_HOSTS"] = "api.example.com"
        headers = self.inject(
            "https://api.example.com/v1",
            gen_ai__conversation__id="support-42",
            gen_ai__agent__name="bot",
            ai__observability__record_id="secret",
            other="x",
        )
        entries = dict(p.split("=", 1) for p in headers["baggage"].split(","))
        self.assertEqual({_CONV: "support-42", _AGENT: "bot"}, entries)

    def test_other_host_gets_no_baggage(self) -> None:
        os.environ["TRULENS_PROPAGATION_HOSTS"] = "api.example.com"
        for url in (
            "https://evil.example.org/v1",
            "https://api.example.com.evil.org/v1",
            "https://sub.api.example.com/v1",
        ):
            with self.subTest(url=url):
                headers = self.inject(url, gen_ai__conversation__id="c")
                self.assertNotIn("baggage", headers)

    def test_host_match_is_case_insensitive_and_ignores_port_and_scheme(
        self,
    ) -> None:
        os.environ["TRULENS_PROPAGATION_HOSTS"] = (
            "API.Example.com, https://other.test:8443"
        )
        for url in ("https://api.example.COM:8080/x", "http://other.test/x"):
            with self.subTest(url=url):
                headers = self.inject(url, gen_ai__conversation__id="c")
                self.assertIn("baggage", headers)

    def test_defaults_to_otlp_endpoint_and_provider_base_urls(self) -> None:
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://collector.test:4318"
        os.environ["OPENAI_BASE_URL"] = "https://gateway.test/v1"
        for url in (
            "http://collector.test:4318/v1/traces",
            "https://gateway.test/v1/chat",
        ):
            with self.subTest(url=url):
                headers = self.inject(url, gen_ai__conversation__id="c")
                self.assertIn("baggage", headers)
        self.assertNotIn(
            "baggage",
            self.inject(
                "https://elsewhere.test/", gen_ai__conversation__id="c"
            ),
        )

    def test_explicit_hosts_replace_the_defaults(self) -> None:
        os.environ["OPENAI_BASE_URL"] = "https://gateway.test/v1"
        os.environ["TRULENS_PROPAGATION_HOSTS"] = "api.example.com"
        headers = self.inject(
            "https://gateway.test/v1", gen_ai__conversation__id="c"
        )
        self.assertNotIn("baggage", headers)

    def test_empty_hosts_value_disables_baggage(self) -> None:
        os.environ["OPENAI_BASE_URL"] = "https://gateway.test/v1"
        os.environ["TRULENS_PROPAGATION_HOSTS"] = ""
        headers = self.inject(
            "https://gateway.test/v1", gen_ai__conversation__id="c"
        )
        self.assertNotIn("baggage", headers)

    def test_values_are_length_limited(self) -> None:
        os.environ["TRULENS_PROPAGATION_HOSTS"] = "api.example.com"
        headers = self.inject(
            "https://api.example.com/", gen_ai__conversation__id="a" * 10_000
        )
        value = headers["baggage"].split("=", 1)[1]
        self.assertEqual("a" * propagation.MAX_BAGGAGE_VALUE_LEN, value)

    def test_values_are_percent_encoded(self) -> None:
        os.environ["TRULENS_PROPAGATION_HOSTS"] = "api.example.com"
        headers = self.inject(
            "https://api.example.com/",
            gen_ai__agent__name="my app, v2;x=y\r\nX: 1",
        )
        header = headers["baggage"]
        self.assertEqual(1, header.count("="))
        for bad in (",", ";", "\r", "\n", " "):
            self.assertNotIn(bad, header)
        parsed = baggage.get_all(
            W3CBaggagePropagator().extract({"baggage": header})
        )
        self.assertEqual("my app, v2;x=y\r\nX: 1", parsed[_AGENT])

    def test_traceparent_is_unaffected_by_baggage_filtering(self) -> None:
        headers = self.inject(
            "https://elsewhere.test/", gen_ai__conversation__id="c"
        )
        self.assertIn("traceparent", headers)

    def test_transport_applies_the_host_check(self) -> None:
        os.environ["TRULENS_PROPAGATION_HOSTS"] = "allowed.test"
        recorded = []

        def handler(request: httpx.Request) -> httpx.Response:
            recorded.append(dict(request.headers))
            return httpx.Response(200)

        client = httpx.Client(
            transport=propagation.tracing_transport(
                httpx.MockTransport(handler)
            )
        )
        token = context_api.attach(
            baggage.set_baggage(_CONV, "c", context_api.get_current())
        )
        try:
            with self.tracer.start_as_current_span("outbound"):
                client.get("https://allowed.test/x")
                client.get("https://denied.test/x")
        finally:
            context_api.detach(token)

        self.assertEqual(f"{_CONV}=c", recorded[0]["baggage"])
        self.assertNotIn("baggage", recorded[1])


class _CloseSpy(httpx.MockTransport):
    """MockTransport that records whether it was closed."""

    def __init__(self) -> None:
        super().__init__(lambda request: httpx.Response(200))
        self.closed = False

    def close(self) -> None:
        self.closed = True

    async def aclose(self) -> None:
        self.closed = True


if __name__ == "__main__":
    unittest.main()
