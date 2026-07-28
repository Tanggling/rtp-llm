load("@arch_config//:arch_select.bzl", "requirement")

requirement(["torch"])

py_test(
    name = "qwen3_next_qkvz_ba_fusion_test",
    srcs = ["qwen3_next_qkvz_ba_fusion_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["H20"],
    exec_properties = {'gpu': 'H20', 'gpu_count': '1'},
)

py_test(
    name = "log2phy_dispatch_test",
    srcs = ["log2phy_dispatch_test.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        ":torch",
    ],
    visibility = ["//visibility:public"],
)
