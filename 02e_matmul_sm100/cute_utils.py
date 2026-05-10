from cutlass import Boolean, Float32, Int32, Uint32, Uint64, cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def tcgen05_mma_f16(
    d_tmem,
    a_desc,
    b_desc,
    idesc,
    enable_input_d,
    cta_group: nvvm.Tcgen05GroupKind = nvvm.Tcgen05GroupKind.CTA_1,
    *,
    loc=None,
    ip=None,
) -> None:
    nvvm.tcgen05_mma(
        nvvm.Tcgen05MMAKind.F16,
        cta_group,
        llvm.inttoptr(
            llvm.PointerType.get(cute.AddressSpace.tmem.value),
            Int32(d_tmem).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        ),
        Uint64(a_desc).ir_value(loc=loc, ip=ip),
        Uint64(b_desc).ir_value(loc=loc, ip=ip),
        Int32(idesc & 0xFFFF_FFFF).ir_value(loc=loc, ip=ip),
        Boolean(enable_input_d).ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tcgen05_ld(taddr, shape, num, *, loc=None, ip=None):
    if shape == nvvm.Tcgen05LdStShape.SHAPE_32X32B:
        num_regs = num
    elif shape == nvvm.Tcgen05LdStShape.SHAPE_16X128B:
        num_regs = num * 2
    elif shape == nvvm.Tcgen05LdStShape.SHAPE_16X256B:
        num_regs = num * 4
    else:
        raise ValueError

    tmem_ptr_ty = llvm.PointerType.get(cute.AddressSpace.tmem.value)
    tmem_ptr = llvm.inttoptr(tmem_ptr_ty, Int32(taddr).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)

    if num_regs == 1:
        reg = nvvm.tcgen05_ld(Int32.mlir_type, shape, num, tmem_ptr, loc=loc, ip=ip)
        reg_f32 = llvm.bitcast(Float32.mlir_type, reg, loc=loc, ip=ip)
        return Float32(reg_f32)

    else:
        vec_i32_ty = ir.VectorType.get([num_regs], Int32.mlir_type, loc=loc)
        vec_f32_ty = ir.VectorType.get([num_regs], Float32.mlir_type, loc=loc)
        regs = nvvm.tcgen05_ld(vec_i32_ty, shape, num, tmem_ptr, loc=loc, ip=ip)
        regs_f32 = llvm.bitcast(vec_f32_ty, regs, loc=loc, ip=ip)
        return cute.TensorSSA(regs_f32, (num_regs,), Float32)


@dsl_user_op
def tcgen05_dealloc(
    cta_group: nvvm.Tcgen05GroupKind = nvvm.Tcgen05GroupKind.CTA_1,
    *,
    loc=None,
    ip=None,
) -> None:
    tmem_ptr_ty = llvm.PointerType.get(cute.AddressSpace.tmem.value)
    nvvm.tcgen05_dealloc(
        llvm.inttoptr(tmem_ptr_ty, Int32(0).ir_value(loc=loc, ip=ip), loc=loc, ip=ip),
        Int32(512).ir_value(loc=loc, ip=ip),
        group=cta_group,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _fp32x2_to_bf16x2(a: Float32, b: Float32, *, loc=None, ip=None) -> Uint32:
    out = llvm.inline_asm(
        T.i32(),
        [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
        "cvt.rn.bf16x2.f32 $0, $2, $1;",
        "=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
    )
    return Uint32(out)
