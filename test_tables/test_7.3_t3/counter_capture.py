"""Read one fixed XProf counter window over gRPC, without exporting files."""

from functools import lru_cache


# JAX 0.10.2's pinned wire protocol and server's emit_xspace handling:
# https://github.com/openxla/xla/blob/5a9e73cbd92530cac2ac36f4736a774b2412afe2/third_party/tsl/tsl/profiler/protobuf/profiler_options.proto
# https://github.com/openxla/xla/blob/5a9e73cbd92530cac2ac36f4736a774b2412afe2/third_party/tsl/tsl/profiler/protobuf/profiler_service.proto
# https://github.com/openxla/xla/blob/5a9e73cbd92530cac2ac36f4736a774b2412afe2/xla/tsl/profiler/rpc/profiler_service_impl.cc
# https://github.com/openxla/xla/blob/5a9e73cbd92530cac2ac36f4736a774b2412afe2/xla/tsl/profiler/rpc/client/capture_profile.cc
_RPC = "/tensorflow.ProfilerService/Profile"
_DURATION_MS = 5000
_MESSAGE_LIMIT = 512 << 20


@lru_cache(maxsize=1)
def _schema():
    # A private pool avoids TensorFlow imports and collisions with installed
    # profiler protos. Message/package names do not affect wire compatibility.
    try:
        from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
    except ImportError as error:
        raise RuntimeError("The existing environment must provide protobuf.") from error
    field_type = descriptor_pb2.FieldDescriptorProto
    file = descriptor_pb2.FileDescriptorProto(
        name="sectile_table3_capture.proto", package="sectile.table3", syntax="proto3"
    )

    def message(name):
        return file.message_type.add(name=name)

    def field(message, name, number, kind, *, repeated=False, type_name=None,
              oneof=None):
        value = message.field.add(
            name=name, number=number, type=kind,
            label=field_type.LABEL_REPEATED if repeated else field_type.LABEL_OPTIONAL,
        )
        if type_name is not None:
            value.type_name = ".sectile.table3." + type_name
        if oneof is not None:
            value.oneof_index = oneof

    config = message("AdvancedConfigValue")
    config.oneof_decl.add(name="value")
    field(config, "string_value", 1, field_type.TYPE_STRING, oneof=0)
    field(config, "bool_value", 2, field_type.TYPE_BOOL, oneof=0)
    field(config, "int64_value", 3, field_type.TYPE_INT64, oneof=0)

    options = message("ProfileOptions")
    field(options, "include_dataset_ops", 1, field_type.TYPE_BOOL)
    field(options, "host_tracer_level", 2, field_type.TYPE_UINT32)
    field(options, "device_tracer_level", 3, field_type.TYPE_UINT32)
    field(options, "python_tracer_level", 4, field_type.TYPE_UINT32)
    field(options, "version", 5, field_type.TYPE_UINT32)
    field(options, "enable_hlo_proto", 7, field_type.TYPE_BOOL)
    field(options, "duration_ms", 9, field_type.TYPE_UINT64)
    field(options, "repository_path", 10, field_type.TYPE_STRING)
    entry = options.nested_type.add(name="AdvancedConfigurationEntry")
    entry.options.map_entry = True
    field(entry, "key", 1, field_type.TYPE_STRING)
    field(entry, "value", 2, field_type.TYPE_MESSAGE, type_name="AdvancedConfigValue")
    field(options, "advanced_configuration", 12, field_type.TYPE_MESSAGE,
          repeated=True, type_name="ProfileOptions.AdvancedConfigurationEntry")

    request = message("ProfileRequest")
    field(request, "duration_ms", 1, field_type.TYPE_UINT64)
    field(request, "max_events", 2, field_type.TYPE_UINT64)
    field(request, "tools", 3, field_type.TYPE_STRING, repeated=True)
    field(request, "opts", 4, field_type.TYPE_MESSAGE, type_name="ProfileOptions")
    field(request, "repository_root", 5, field_type.TYPE_STRING)
    field(request, "emit_xspace", 9, field_type.TYPE_BOOL)

    response = message("ProfileResponse")
    field(response, "empty_trace", 7, field_type.TYPE_BOOL)
    # The official field is a nested XSpace message. Declaring its identical
    # length-delimited wire representation as bytes preserves the raw payload.
    field(response, "xspace", 8, field_type.TYPE_BYTES)
    status = message("XSpaceStatus")
    field(status, "errors", 2, field_type.TYPE_STRING, repeated=True)
    field(status, "warnings", 3, field_type.TYPE_STRING, repeated=True)
    pool = descriptor_pool.DescriptorPool()
    pool.Add(file)
    return tuple(
        message_factory.GetMessageClass(pool.FindMessageTypeByName("sectile.table3." + name))
        for name in ("ProfileRequest", "ProfileResponse", "XSpaceStatus")
    )


def preflight():
    """Check installed capture dependencies without contacting a TPU/server."""
    try:
        import grpc
    except ImportError as error:
        raise RuntimeError("The existing environment must provide grpcio.") from error
    if not callable(getattr(grpc, "insecure_channel", None)):
        raise RuntimeError("grpcio does not expose insecure_channel.")
    if not isinstance(getattr(grpc, "RpcError", None), type):
        raise RuntimeError("grpcio does not expose RpcError.")
    _schema()


def capture_window(address: str, options: dict, duration_ms=5000) -> bytes:
    """Capture exactly 5000 ms and return XSpace bytes; never call trace/export."""
    if type(duration_ms) is not int or duration_ms != _DURATION_MS:
        raise ValueError("Table 3 requires independent 5000 ms capture windows.")
    if not isinstance(address, str) or not address.strip():
        raise ValueError("A profiler server host:port is required.")
    if not isinstance(options, dict):
        raise TypeError("Profiler options must be a dictionary.")
    preflight()
    import grpc
    request_type, response_type, status_type = _schema()
    request = request_type(
        duration_ms=duration_ms, max_events=1_000_000,
        repository_root="", emit_xspace=True,
    )
    request.tools.extend(("trace_viewer", "op_profile", "input_pipeline", "kernel_stats",
                          "memory_viewer", "memory_profile", "overview_page", "pod_viewer",
                          "tensorflow_stats", "xplane.pb"))
    # version=0 makes this server replace custom options with defaults.
    request.opts.version = 1
    request.opts.include_dataset_ops = True
    request.opts.enable_hlo_proto = True
    request.opts.duration_ms = duration_ms
    request.opts.repository_path = ""
    for name, default in (("host_tracer_level", 1), ("device_tracer_level", 1),
                          ("python_tracer_level", 0)):
        value = options.get(name, default)
        if type(value) is not int or value < 0:
            raise ValueError(f"Invalid profiler option {name}: {value!r}")
        setattr(request.opts, name, value)
    for name, value in options.items():
        if name in {"host_tracer_level", "device_tracer_level", "python_tracer_level"}:
            continue
        if not isinstance(name, str) or not name.startswith("tpu_"):
            raise ValueError(f"Unsupported counter option: {name!r}")
        config = request.opts.advanced_configuration[name]
        if isinstance(value, bool):
            config.bool_value = value
        elif isinstance(value, int):
            config.int64_value = value
        elif isinstance(value, str):
            config.string_value = value
        else:
            raise TypeError(f"Unsupported value for counter option {name!r}")
    limits = (("grpc.max_receive_message_length", _MESSAGE_LIMIT),
              ("grpc.max_send_message_length", _MESSAGE_LIMIT))
    try:
        with grpc.insecure_channel(address, options=limits) as channel:
            rpc = channel.unary_unary(
                _RPC, request_serializer=lambda message: message.SerializeToString(),
                response_deserializer=response_type.FromString,
            )
            response = rpc(request, timeout=60.0)
    except grpc.RpcError as error:
        raise RuntimeError(f"XProf capture RPC failed: {error}") from error
    raw = response.xspace
    if response.empty_trace or not isinstance(raw, bytes) or not raw:
        raise RuntimeError("XProf returned an empty capture or no raw XSpace.")
    status = status_type.FromString(raw)
    # The paper's counter gate classifies warnings after checking the complete
    # required counter vector; trace-buffer warnings alone are not a rejection.
    if status.errors:
        raise RuntimeError(f"XProf reported errors: {list(status.errors)}")
    return raw
