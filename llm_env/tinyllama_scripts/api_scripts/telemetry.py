from opentelemetry import trace, metrics
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

# -------------------------------------------------------------------
# RESOURCE
# -------------------------------------------------------------------

resource = Resource.create(
    attributes={
        "service.name": "fastapi-llm-service",
    }
)

# -------------------------------------------------------------------
# TRACING
# -------------------------------------------------------------------

trace.set_tracer_provider(TracerProvider(resource=resource))
tracer = trace.get_tracer(__name__)

span_exporter = OTLPSpanExporter(
    endpoint="http://192.168.1.7:4318/v1/traces"
)

span_processor = BatchSpanProcessor(span_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# -------------------------------------------------------------------
# METRICS
# -------------------------------------------------------------------

metric_reader = PeriodicExportingMetricReader(
    OTLPMetricExporter(
        endpoint="http://192.168.1.7:4318/v1/metrics"
    ),
    export_interval_millis=5000,
)

metrics.set_meter_provider(
    MeterProvider(
        resource=resource,
        metric_readers=[metric_reader],
    )
)

meter = metrics.get_meter(__name__)
