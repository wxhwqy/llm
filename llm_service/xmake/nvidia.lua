local nccl_include = os.getenv("NCCL_INCLUDE") or "/usr/include"
local nccl_lib = os.getenv("NCCL_LIB") or "/usr/lib/x86_64-linux-gnu"

target("llaisys-device-nvidia")
    set_kind("static")
    add_rules("cuda")
    add_cugencodes("native")
    set_languages("cxx17")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cuflags("-Xcompiler=-fPIC", {force = true})
    end

    add_includedirs("../include")
    add_includedirs(nccl_include)
    add_files("../src/device/nvidia/*.cu")

    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    add_deps("llaisys-tensor")
    add_rules("cuda")
    add_cugencodes("native")
    set_languages("cxx17")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cuflags("-Xcompiler=-fPIC", {force = true})
    end

    add_includedirs("../include")
    add_links("cublas")
    add_files("../src/ops/*/nvidia/*.cu")

    on_install(function (target) end)
target_end()
