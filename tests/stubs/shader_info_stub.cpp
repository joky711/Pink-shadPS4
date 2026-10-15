// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// info.cpp reads sharp tables through the memory manager; the tests build IR without guest memory and with
// dynamic_tsharp_array_size off. The SRT walker and its context come from flatten_extended_userdata_pass.cpp

#include "shader_recompiler/info.h"

namespace Shader {

void Info::RefreshSharpTables() {}

u32 DynamicTsharpArraySize() {
    return 0;
}

} // namespace Shader
