"""Verify that v_ldexp_f32 lowers to ldexp32 with valid float/integer operands and refuses invalid modifiers.

Refs #2618.
"""
from pathlib import Path
import struct
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))

import gfx2hlsl as G  # noqa: E402
import startup_gfx2hlsl_compute_check as C  # noqa: E402


def vop3a(op, vdst, src0, src1, src2=0, *, abs_bits=0, clamp=0, omod=0, neg=0, lit=None):
    """Encode one synthetic gfx1013 VOP3A instruction."""
    w0 = 0xd4000000 | (op << 16) | (clamp << 15) | (abs_bits << 8) | vdst
    w1 = (neg << 29) | (omod << 27) | (src2 << 18) | (src1 << 9) | src0
    out = struct.pack('<2I', w0, w1)
    if lit is not None:
        out += struct.pack('<I', lit & 0xffffffff)
    return out


ENDPGM = struct.pack('<I', 0xbf800000)


def translate_cs(code):
    return G.ComputePipeline(code + ENDPGM, C.root_blob(), C.CASES['A'][2], (2, 3, 4)).hlsl_cs()


def check_translations():
    # 1. Preamble contains ldexp32
    hlsl_base = translate_cs(vop3a(0x362, 5, 240, 130))
    assert 'float ldexp32(float value, int delta) {' in hlsl_base, 'preamble missing ldexp32'
    assert 'v5 = U(ldexp32(0.5, asint(0x00000002u)));' in hlsl_base, f'unexpected hlsl: {hlsl_base}'

    # 2. Float input modifiers on src0:
    # Negation on inline float
    hlsl_neg = translate_cs(vop3a(0x362, 5, 240, 130, neg=1))
    assert 'v5 = U(ldexp32(-0.5, asint(0x00000002u)));' in hlsl_neg, f'unexpected neg: {hlsl_neg}'

    # Negation on VGPR src0
    hlsl_vneg = translate_cs(vop3a(0x362, 5, 257, 130, neg=1))
    assert 'v5 = U(ldexp32((-F(v1)), asint(0x00000002u)));' in hlsl_vneg, f'unexpected vneg: {hlsl_vneg}'

    # Absolute on VGPR src0
    hlsl_vabs = translate_cs(vop3a(0x362, 5, 257, 130, abs_bits=1))
    assert 'v5 = U(ldexp32(abs(F(v1)), asint(0x00000002u)));' in hlsl_vabs, f'unexpected vabs: {hlsl_vabs}'

    # Combined -|src0| on VGPR
    hlsl_neg_abs = translate_cs(vop3a(0x362, 5, 257, 130, neg=1, abs_bits=1))
    assert 'v5 = U(ldexp32((-abs(F(v1))), asint(0x00000002u)));' in hlsl_neg_abs, f'unexpected neg abs: {hlsl_neg_abs}'

    # 3. Exponent operand types:
    # Negative inline integer constant -2 (encoding 194)
    hlsl_neg_exp = translate_cs(vop3a(0x362, 5, 240, 194))
    assert 'v5 = U(ldexp32(0.5, asint(0xfffffffeu)));' in hlsl_neg_exp, f'unexpected neg exp: {hlsl_neg_exp}'

    # Exponent from SGPR s4 (user data ud0, encoding 4)
    hlsl_sreg = translate_cs(vop3a(0x362, 5, 240, 4))
    assert 'v5 = U(ldexp32(0.5, asint(ud0)));' in hlsl_sreg, f'unexpected sreg: {hlsl_sreg}'

    # Exponent from VGPR v2 (encoding 258)
    hlsl_vreg = translate_cs(vop3a(0x362, 5, 240, 258))
    assert 'v5 = U(ldexp32(0.5, asint(v2)));' in hlsl_vreg, f'unexpected vreg: {hlsl_vreg}'

    # 4. Output modifiers:
    # clamp (generates saturate)
    hlsl_clamp = translate_cs(vop3a(0x362, 5, 240, 130, clamp=1))
    assert 'v5 = U(saturate(ldexp32(0.5, asint(0x00000002u))));' in hlsl_clamp, f'unexpected clamp: {hlsl_clamp}'

    # omod mul:2
    hlsl_mul2 = translate_cs(vop3a(0x362, 5, 240, 130, omod=1))
    assert 'v5 = U((ldexp32(0.5, asint(0x00000002u))) * 2.0);' in hlsl_mul2, f'unexpected mul2: {hlsl_mul2}'


def make_stage(kind='cs'):
    st = G.Stage.__new__(G.Stage)
    st.kind = kind
    st.lines = []
    st.sym = {}
    st.used_s, st.used_v, st.uniform_v = set(), set(), set()
    st.varying_s = set()
    st.scc_uniform = True
    st.uniform_expr = {}
    st.past_entry = False
    st.has_branches = False
    st.exec_known = True
    st.masks = set()
    st.exports = {}
    st.low16_only_v = set()
    st.pinned_low16_v = set()
    st.whole_read_v = set()
    st.byte_float_v = set()
    st.bad_sgprs = set()
    st.defined_uniform_s = None
    st.wave_reductions = []
    st.emitted_reductions = []
    st.wave_reduce_used_exec = False
    return st


def check_refusals():
    # Helper to test Stage.one directly
    def lower_inst(asm_line):
        name = asm_line.split()[0]
        st = make_stage('cs')
        return st.one(name, asm_line)

    # 1. Float modifier on integer exponent operand:
    bad_modifiers = (
        ('v_ldexp_f32 v0, v1, -s4', 'v_ldexp_f32 exponent -s4 has float modifier'),
        ('v_ldexp_f32 v0, v1, -v2', 'v_ldexp_f32 exponent -v2 has float modifier'),
        ('v_ldexp_f32 v0, v1, |s4|', 'v_ldexp_f32 exponent |s4| has float modifier'),
        ('v_ldexp_f32 v0, v1, abs(s4)', 'v_ldexp_f32 exponent abs(s4) has float modifier'),
    )
    for asm, expected in bad_modifiers:
        try:
            lower_inst(asm)
            raise AssertionError(f'expected refusal for {asm}, but succeeded')
        except G.Unsupported as e:
            assert expected in str(e), f'unexpected message for {asm}: {e}'

    # 2. Exponent out of 32-bit range:
    try:
        lower_inst('v_ldexp_f32 v0, v1, 0x100000000')
        raise AssertionError('expected refusal for 0x100000000, but succeeded')
    except G.Unsupported as e:
        assert 'v_ldexp_f32 source 0x100000000' in str(e), f'unexpected message: {e}'

    # 3. Operand counts:
    for asm in ('v_ldexp_f32 v0, v1', 'v_ldexp_f32 v0, v1, v2, v3'):
        try:
            lower_inst(asm)
            raise AssertionError(f'expected refusal for {asm}, but succeeded')
        except G.Unsupported as e:
            assert 'v_ldexp_f32 with' in str(e), f'unexpected message: {e}'

    # 4. Destination not a VGPR:
    try:
        lower_inst('v_ldexp_f32 s0, v1, v2')
        raise AssertionError('expected refusal for destination s0, but succeeded')
    except G.Unsupported as e:
        assert 'v_ldexp_f32 destination s0' in str(e), f'unexpected message: {e}'

    # 5. Disallowed instruction forms:
    for asm, expected in (
        ('v_ldexp_f32_e32 v0, v1, v2', 'v_ldexp_f32_e32'),
        ('v_ldexp_f32_sdwa v0, v1, v2', 'SDWA form of v_ldexp_f32'),
        ('v_ldexp_f32_dpp v0, v1, v2', 'v_ldexp_f32_dpp'),
    ):
        try:
            lower_inst(asm)
            raise AssertionError(f'expected refusal for {asm}, but succeeded')
        except G.Unsupported as e:
            assert expected in str(e), f'unexpected message: {e}'

    # 6. Machine-code VOP3 check_encoding refusals:
    # NEG bit on src1 exponent (w1[30])
    try:
        G.decode(vop3a(0x362, 0, 240, 130, neg=2))
        raise AssertionError('expected decode refusal for NEG on exponent')
    except G.Unsupported as e:
        assert 'v_ldexp_f32 with neg/abs on exponent or unused source' in str(e), f'unexpected: {e}'

    # ABS bit on src1 exponent (w0[9])
    try:
        G.decode(vop3a(0x362, 0, 240, 130, abs_bits=2))
        raise AssertionError('expected decode refusal for ABS on exponent')
    except G.Unsupported as e:
        assert 'v_ldexp_f32 with neg/abs on exponent or unused source' in str(e), f'unexpected: {e}'

    # Unused src2 field set (w1[26:18])
    try:
        G.decode(vop3a(0x362, 0, 240, 130, src2=1))
        raise AssertionError('expected decode refusal for src2 set')
    except G.Unsupported as e:
        assert 'v_ldexp_f32 with unused src2 field set' in str(e), f'unexpected: {e}'


def check_negative_controls():
    # Negative control 1: pre-fix mutant where op_v_ldexp_f32 is missing
    real_handler = getattr(G.Stage, 'op_v_ldexp_f32', None)
    try:
        delattr(G.Stage, 'op_v_ldexp_f32')
        try:
            translate_cs(vop3a(0x362, 5, 240, 130))
            raise AssertionError('expected translation failure under mutant')
        except G.Unsupported as e:
            assert 'v_ldexp_f32' in str(e), f'unexpected mutant error: {e}'
    finally:
        if real_handler:
            setattr(G.Stage, 'op_v_ldexp_f32', real_handler)

    # Negative control 2: mutant that treats exponent as float (accepting float modifiers)
    real_op = G.Stage.op_v_ldexp_f32

    def mutant_op(self, ops, mods):
        dst = G.reg_range(ops[0])
        src0 = self.f(ops[1])
        src1 = self.f(ops[2])
        self.set_vf(dst[1], 'ldexp32(%s, %s)' % (src0, src1), mods)

    try:
        G.Stage.op_v_ldexp_f32 = mutant_op
        # Under this mutant, float modifier on exponent would escape refusal
        st = make_stage('cs')
        st.lower('v_ldexp_f32', 'v_ldexp_f32 v0, v1, -s4', ['v0', 'v1', '-s4'], [])
        assert any('(-s4)' in line or 'F(s4)' in line for line in st.lines), 'mutant failed to reproduce bypass'
    finally:
        G.Stage.op_v_ldexp_f32 = real_op


def check_real_cache_sample():
    # If the real request file from the shader cache exists locally, verify it decodes and translates
    req_a = HERE.parent / 'out' / 'startup' / 'shader_cache' / 'req_12265051f3da290f.a'
    if req_a.exists():
        code = req_a.read_bytes()
        ins = G.decode(code)
        ldexp_found = any('v_ldexp_f32' in item[2] for item in ins)
        assert ldexp_found, 'req_12265051f3da290f.a missing v_ldexp_f32'
        st = make_stage('vs')
        for _, _, base, asm in ins:
            if 'v_ldexp_f32' in base:
                name, ops, mods = G.parse(asm)
                st.lower(base, asm, ops, mods)
        assert any('ldexp32' in line for line in st.lines), 'ldexp32 missing from translated real shader stage'


def main():
    check_translations()
    check_refusals()
    check_negative_controls()
    check_real_cache_sample()
    print('gfx2hlsl v_ldexp_f32 lowering and refusal checks passed')


if __name__ == '__main__':
    main()
