// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// SRT walker coverage for scalar loads whose offset comes from a Phi. The guest SRT is a host
// array: the walker masks pointers to 48 bits, which holds for user space addresses

#include <array>
#include <cstring>

#include <gtest/gtest.h>

#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/basic_block.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/post_order.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/recompiler.h"

namespace {

using namespace Shader;

constexpr u32 SrtBase = 0xA000u;
constexpr u32 PtrSgpr = 2; // s[2:3] holds the SRT pointer

struct SrtFixture {
    alignas(16) std::array<u32, 64> srt{};
    alignas(16) std::array<u32, 64> child{};
    std::array<u32, NUM_USER_DATA_REGS> user_data{};
    Pools pools{};
    Info info{};
    IR::Program program{info};

    SrtFixture() {
        for (u32 i = 0; i < srt.size(); ++i) {
            srt[i] = SrtBase + i;
            child[i] = 0xC000u + i;
        }
        SetPointer(user_data.data() + PtrSgpr, srt.data());
        info.hw_stage = HwStage::Compute;
        info.sw_stage = SwStage::Compute;
        info.user_data = user_data;
    }

    static void SetPointer(u32* dst, const void* ptr) {
        const u64 address = reinterpret_cast<u64>(ptr);
        dst[0] = static_cast<u32>(address);
        dst[1] = static_cast<u32>(address >> 32);
    }

    IR::Block* NewBlock() {
        IR::Block* block = pools.block_pool.Create(pools.inst_pool);
        program.blocks.push_back(block);
        return block;
    }

    static IR::Value Pointer(IR::IREmitter& ir) {
        return ir.CompositeConstruct(ir.GetUserData(IR::ScalarReg(PtrSgpr)),
                                     ir.GetUserData(IR::ScalarReg(PtrSgpr + 1)));
    }

    /// Offset as S_LOAD_DWORD builds it from a byte offset sgpr
    static IR::U32 DwordOffset(IR::IREmitter& ir, const IR::U32& byte_offset) {
        return ir.IAdd(ir.ShiftRightLogical(byte_offset, ir.Imm32(2u)), ir.Imm32(0u));
    }

    static IR::Inst* NewPhi(IR::Block* block) {
        IR::Inst* phi = &*block->PrependNewInst(block->begin(), IR::Opcode::Phi);
        phi->SetFlags(IR::Type::U32);
        return phi;
    }

    void Run(IR::Block* entry) {
        program.post_order_blocks = IR::PostOrder(entry);
        Optimization::FlattenExtendedUserdataPass(program);
    }

    /// Value the shader sees for a flattened scalar load
    u32 Flat(const IR::Inst* read_const) const {
        const u32 off = read_const->Flags<u16>();
        EXPECT_NE(off, 0u) << "scalar load was not flattened";
        EXPECT_LT(off, info.flattened_ud_buf.size());
        return off < info.flattened_ud_buf.size() ? info.flattened_ud_buf[off] : 0u;
    }
};

/// [entry] -> [then] / [else] -> [merge]. The byte offset is 16 on one side and 32 on the other,
/// merged by a Phi in [merge] and used as the offset of one s_load_dword
struct BranchOffsetProgram : SrtFixture {
    IR::Block* entry;
    IR::Block* then_block;
    IR::Block* else_block;
    IR::Block* merge;
    IR::Inst* consumer; // IAdd(load, 1)

    BranchOffsetProgram() {
        entry = NewBlock();
        then_block = NewBlock();
        else_block = NewBlock();
        merge = NewBlock();
        entry->AddBranch(then_block);
        entry->AddBranch(else_block);
        then_block->AddBranch(merge);
        else_block->AddBranch(merge);

        IR::IREmitter entry_ir{*entry};
        (void)entry_ir.GetUserData(IR::ScalarReg(0));

        IR::Inst* phi = NewPhi(merge);
        phi->AddPhiOperand(then_block, IR::Value{16u});
        phi->AddPhiOperand(else_block, IR::Value{32u});

        IR::IREmitter ir{*merge};
        const IR::U32 load = ir.ReadConst(Pointer(ir), DwordOffset(ir, IR::U32{IR::Value{phi}}));
        consumer = IR::Value{ir.IAdd(load, ir.Imm32(1u))}.Inst();
    }
};

TEST(SrtWalkerPhi, BranchOffsetIsFlattenedPerPredecessor) {
    BranchOffsetProgram p;
    p.Run(p.entry);

    // The load is now one flattened load per predecessor, merged by a Phi
    const IR::Inst* merged = p.consumer->Arg(0).Inst();
    ASSERT_EQ(merged->GetOpcode(), IR::Opcode::Phi);
    ASSERT_EQ(merged->NumArgs(), 2u);
    for (size_t i = 0; i < merged->NumArgs(); ++i) {
        const IR::Inst* load = merged->Arg(i).Inst();
        ASSERT_EQ(load->GetOpcode(), IR::Opcode::ReadConst);
        EXPECT_EQ(load->GetParent(), merged->PhiBlock(i));
        const u32 expected = merged->PhiBlock(i) == p.then_block ? SrtBase + 4 : SrtBase + 8;
        EXPECT_EQ(p.Flat(load), expected);
    }
}

/// [entry] -> [body] -> [merge] and [entry] -> [merge]: an if without else, so one predecessor of
/// [merge] has two successors. The load reads through a table pointer that is itself a flattened
/// load (s_load_dwordx2 at dword 10 of the root)
TEST(SrtWalkerPhi, IfWithoutElseThroughNestedPointer) {
    SrtFixture p;
    SrtFixture::SetPointer(p.srt.data() + 10, p.child.data());
    IR::Block* entry = p.NewBlock();
    IR::Block* body = p.NewBlock();
    IR::Block* merge = p.NewBlock();
    entry->AddBranch(body);
    entry->AddBranch(merge);
    body->AddBranch(merge);

    IR::IREmitter entry_ir{*entry};
    const IR::Value root = SrtFixture::Pointer(entry_ir);
    const IR::U32 table_lo = entry_ir.ReadConst(root, entry_ir.Imm32(10u));
    const IR::U32 table_hi = entry_ir.ReadConst(root, entry_ir.Imm32(11u));
    const IR::Value table = entry_ir.CompositeConstruct(table_lo, table_hi);

    IR::Inst* phi = SrtFixture::NewPhi(merge);
    phi->AddPhiOperand(entry, IR::Value{8u});
    phi->AddPhiOperand(body, IR::Value{24u});
    IR::IREmitter ir{*merge};
    const IR::U32 load = ir.ReadConst(table, SrtFixture::DwordOffset(ir, IR::U32{IR::Value{phi}}));
    const IR::Inst* consumer = IR::Value{ir.IAdd(load, ir.Imm32(1u))}.Inst();

    p.Run(entry);

    const IR::Inst* merged = consumer->Arg(0).Inst();
    ASSERT_EQ(merged->GetOpcode(), IR::Opcode::Phi);
    for (size_t i = 0; i < merged->NumArgs(); ++i) {
        const IR::Inst* split = merged->Arg(i).Inst();
        ASSERT_EQ(split->GetOpcode(), IR::Opcode::ReadConst);
        const u32 expected = merged->PhiBlock(i) == entry ? 0xC000u + 2 : 0xC000u + 6;
        EXPECT_EQ(p.Flat(split), expected);
    }
}

/// [entry] -> [header] <-> [latch], [header] -> [exit]. The byte offset is a loop counter (0, 4, 8,
/// ...), so no single value exists before the draw: the load must stay unflattened and the pass
/// must not crash
TEST(SrtWalkerPhi, LoopCounterOffsetStaysDynamic) {
    SrtFixture p;
    IR::Block* entry = p.NewBlock();
    IR::Block* header = p.NewBlock();
    IR::Block* latch = p.NewBlock();
    IR::Block* exit = p.NewBlock();
    entry->AddBranch(header);
    header->AddBranch(latch);
    header->AddBranch(exit);
    latch->AddBranch(header);

    IR::Inst* counter = SrtFixture::NewPhi(header);
    IR::IREmitter ir{*header};
    const IR::U32 load = ir.ReadConst(SrtFixture::Pointer(ir),
                                      SrtFixture::DwordOffset(ir, IR::U32{IR::Value{counter}}));
    IR::IREmitter latch_ir{*latch};
    const IR::U32 next = latch_ir.IAdd(IR::U32{IR::Value{counter}}, latch_ir.Imm32(4u));
    counter->AddPhiOperand(entry, IR::Value{0u});
    counter->AddPhiOperand(latch, next);

    p.Run(entry);

    const IR::Inst* load_inst = IR::Value{load}.Inst();
    EXPECT_EQ(load_inst->GetOpcode(), IR::Opcode::ReadConst);
    EXPECT_EQ(load_inst->Flags<u16>(), 0u);
}

/// A pointer (s_load_dwordx2) read at a Phi offset, then dereferenced at an immediate offset. Only
/// scalar data loads are split per predecessor, so the pointer chain is left as it was and the pass
/// must not crash
TEST(SrtWalkerPhi, PointerAtBranchOffsetIsLeftAlone) {
    BranchOffsetProgram p;
    // Child table pointer at dwords 4..5 and 8..9
    SrtFixture::SetPointer(p.srt.data() + 4, p.child.data());
    SrtFixture::SetPointer(p.srt.data() + 8, p.child.data());

    IR::Inst* phi = &*p.merge->begin();
    IR::IREmitter ir{*p.merge, IR::Block::InstructionList::s_iterator_to(*p.consumer)};
    const IR::U32 dw = SrtFixture::DwordOffset(ir, IR::U32{IR::Value{phi}});
    const IR::Value base = SrtFixture::Pointer(ir);
    const IR::U32 lo = ir.ReadConst(base, dw);
    const IR::U32 hi = ir.ReadConst(base, ir.IAdd(dw, ir.Imm32(1u)));
    const IR::U32 child_value = ir.ReadConst(ir.CompositeConstruct(lo, hi), ir.Imm32(3u));
    (void)ir.IAdd(child_value, ir.Imm32(1u));

    p.Run(p.entry);

    EXPECT_EQ(IR::Value{child_value}.Inst()->GetOpcode(), IR::Opcode::ReadConst);
    EXPECT_EQ(IR::Value{lo}.Inst()->GetOpcode(), IR::Opcode::ReadConst);
}

} // namespace
