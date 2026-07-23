load("@arch_config//:arch_select.bzl", "requirement")

requirement(["torch"])

py_test(
    name = "eplb_algo_test",
    srcs = ["eplb_algo_test.py"],
    deps = [
        "//rtp_llm:eplb",
    ] + [":torch"],
)
