add_rules("mode.debug", "mode.release")
set_encodings("utf-8")

add_includedirs("include")

local nccl_include = os.getenv("NCCL_INCLUDE") or "/usr/include"
local nccl_lib = os.getenv("NCCL_LIB") or "/usr/lib/x86_64-linux-gnu"

-- CPU --
includes("xmake/cpu.lua")

-- NVIDIA --
option("nv-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Nvidia GPU")
option_end()

if has_config("nv-gpu") then
    add_defines("ENABLE_NVIDIA_API")
    includes("xmake/nvidia.lua")
end

target("llaisys-utils")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    add_files("src/utils/*.cpp")
    on_install(function (target) end)
target_end()

target("llaisys-device")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device-cpu")
    if has_config("nv-gpu") then
        add_deps("llaisys-device-nvidia")
    end
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    add_files("src/device/*.cpp")
    on_install(function (target) end)
target_end()

target("llaisys-core")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    add_files("src/core/*/*.cpp")
    on_install(function (target) end)
target_end()

target("llaisys-tensor")
    set_kind("static")
    add_deps("llaisys-core")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    add_files("src/tensor/*.cpp")
    on_install(function (target) end)
target_end()

target("llaisys-ops")
    set_kind("static")
    add_deps("llaisys-ops-cpu")
    if has_config("nv-gpu") then
        add_deps("llaisys-ops-nvidia")
    end
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    add_files("src/ops/*/*.cpp")
    on_install(function (target) end)
target_end()

target("llaisys-models")
    set_kind("static")
    add_deps("llaisys-tensor")
    add_deps("llaisys-ops")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    if has_config("nv-gpu") then
        add_includedirs(nccl_include)
        add_includedirs("/usr/local/cuda/include")
    end
    add_files("src/models/*.cpp")
    on_install(function (target) end)
target_end()

target("llaisys")
    set_kind("shared")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")
    add_deps("llaisys-core")
    add_deps("llaisys-tensor")
    add_deps("llaisys-ops")
    add_deps("llaisys-models")
    set_languages("cxx17")
    set_warnings("all", "error")
    if has_config("nv-gpu") then
        add_includedirs(nccl_include)
        add_includedirs("/usr/local/cuda/include")
    end
    add_files("src/llaisys/*.cc")
    if has_config("nv-gpu") then
        add_rules("cuda")
        add_cugencodes("native")
        add_cuflags("-Xcompiler=-fPIC", {force = true})
        add_files("src/llaisys/cuda_link.cu")
        add_links("cudart", "cublas", "nvToolsExt")
        add_linkdirs("/usr/local/cuda/lib64")
        add_ldflags("-Xlinker=-rpath=/usr/local/cuda/lib64", {force = true})
        add_ldflags("-Xlinker=-rpath=" .. nccl_lib, {force = true})
        add_ldflags(nccl_lib .. "/libnccl.so.2", {force = true})
    end
    set_installdir(".")

    after_build(function (target)
        local so = target:targetfile()
        os.cp(so, "python/llaisys/libllaisys/")
        print("Copied " .. so .. " -> python/llaisys/libllaisys/")
    end)
target_end()
