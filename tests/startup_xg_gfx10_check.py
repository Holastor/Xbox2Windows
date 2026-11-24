"""Verify GFX10 texture computer, micro-tile detiling and mip-tail layout."""
import ctypes as ct
import struct
import sys

DESC = struct.Struct("<I4xQQIHHIIIII4x")
LAYOUT_BYTES = 0x18 + 4 * 0x6C8 + 8
ENTRY = struct.Struct("<IHHI")

PTR = ct.c_void_p
UINT = ct.c_uint32
UINT64 = ct.c_uint64
HRESULT = ct.c_int32

LAYOUT = ct.WINFUNCTYPE(HRESULT, PTR, PTR)
SIZE = ct.WINFUNCTYPE(UINT64, PTR)
ALIGN = ct.WINFUNCTYPE(UINT64, PTR)
OFFSET = ct.WINFUNCTYPE(UINT64, PTR, UINT, UINT)
ELEMENT = ct.WINFUNCTYPE(UINT64, PTR, UINT, UINT, UINT64, UINT, UINT, UINT, PTR)
COPY_INTO = ct.WINFUNCTYPE(HRESULT, PTR, PTR, UINT, UINT, PTR, UINT, UINT)
RELEASE = ct.WINFUNCTYPE(UINT64, PTR)
TILING = ct.WINFUNCTYPE(HRESULT, PTR, PTR, PTR, PTR, PTR, UINT, PTR)


def test_format(create, name, fmt, width, height, mips, slices, layout_code, bpe, block_dim):
    desc = ct.create_string_buffer(
        DESC.pack(3, 0, width, height, slices, mips, fmt, 1, 0, layout_code, 0)
    )
    computer = PTR()
    hr = create(desc, ct.byref(computer), 0, 0)
    assert hr == 0 and computer, f"Failed to create computer for {name}"
    slots = ct.cast(computer, ct.POINTER(ct.POINTER(PTR)))[0]

    try:
        # Slot 3 (size) & Slot 4 (alignment)
        total_size = SIZE(slots[3])(computer)
        alignment = ALIGN(slots[4])(computer)
        assert total_size > 0, f"{name}: size must be > 0"
        expected_align = 65536 if (layout_code & 0xF) in (9, 10) else (4096 if (layout_code & 0xF) in (5, 6) else 256)
        assert alignment == expected_align, f"{name}: align={alignment}, expected={expected_align}"

        # Slot 2 (layout)
        layout_buf = ct.create_string_buffer(LAYOUT_BYTES)
        assert LAYOUT(slots[2])(computer, layout_buf) == 0
        l_size, l_align, l_mips, l_planes = struct.unpack_from("<QQII", layout_buf)
        assert l_size == total_size
        assert l_align == alignment
        assert l_mips == mips
        assert l_planes == 1

        # Slot 10 (tiling)
        tiles = UINT()
        packed = ct.create_string_buffer(12)
        shape = (UINT * 3)()
        count = UINT(mips * slices)
        entries = ct.create_string_buffer(count.value * ENTRY.size)
        assert TILING(slots[10])(computer, ct.byref(tiles), packed, shape, ct.byref(count), 0, entries) == 0
        num_standard, num_packed = struct.unpack_from("<BB", packed)
        assert num_standard + num_packed == mips, f"{name}: standard={num_standard} + packed={num_packed} != {mips}"

        # Test subresource tile & detile roundtrip across mips
        tiled_buf = ct.create_string_buffer(total_size)
        for s in range(slices):
            for m in range(mips):
                mw = (max(1, width >> m) + block_dim - 1) // block_dim
                mh = (max(1, height >> m) + block_dim - 1) // block_dim
                row_pitch = mw * bpe + 16  # padded pitch

                # Create synthetic linear content
                linear_src = bytearray(mh * row_pitch)
                for y in range(mh):
                    for x in range(mw):
                        for b in range(bpe):
                            linear_src[y * row_pitch + x * bpe + b] = (m * 41 + s * 17 + x * 7 + y * 13 + b) & 0xFF

                src_buf = ct.create_string_buffer(bytes(linear_src), len(linear_src))
                subresource = s * mips + m
                # Slot 8: CopyIntoSubresource (linear -> tiled)
                rc = COPY_INTO(slots[8])(computer, tiled_buf, 0, subresource, src_buf, row_pitch, len(linear_src))
                assert rc == 0, f"{name} mip {m} CopyIntoSubresource failed: 0x{rc & 0xFFFFFFFF:08x}"

                # Slot 9: CopyFromSubresource (tiled -> linear)
                dest_buf = ct.create_string_buffer(b"\xaa" * len(linear_src), len(linear_src))
                rc = COPY_INTO(slots[9])(computer, dest_buf, 0, subresource, tiled_buf, row_pitch, len(linear_src))
                assert rc == 0, f"{name} mip {m} CopyFromSubresource failed: 0x{rc & 0xFFFFFFFF:08x}"

                # Verify exact texel match for every row
                for y in range(mh):
                    row_orig = linear_src[y * row_pitch: y * row_pitch + mw * bpe]
                    row_detiled = dest_buf.raw[y * row_pitch: y * row_pitch + mw * bpe]
                    assert row_detiled == row_orig, f"{name} slice {s} mip {m} row {y} mismatch after detile!"

                # Verify Slot 6: element address matches
                bit = UINT(99)
                first_addr = ELEMENT(slots[6])(computer, 0, m, 0, 0, s, 0, ct.byref(bit))
                assert bit.value == 0
                last_addr = ELEMENT(slots[6])(computer, 0, m, mw - 1, mh - 1, s, 0, ct.byref(bit))
                assert last_addr < total_size, f"{name} element address out of bounds: {last_addr} >= {total_size}"

    finally:
        RELEASE(slots[1])(computer)


def main(dll_path):
    create = ct.WinDLL(dll_path).XGCreateTextureComputer
    create.argtypes = (PTR, ct.POINTER(PTR), ct.c_uint64, ct.c_uint64)
    create.restype = HRESULT

    cases = [
        # BC1 (fmt 71): 8 bytes/block, 64KB_S (0x109)
        ("BC1 256x256 5-mip 64KB_S", 71, 256, 256, 5, 1, 0x109, 8, 4),
        # BC1 (fmt 71): 8 bytes/block, 4KB_S (0x105)
        ("BC1 128x128 4-mip 4KB_S", 71, 128, 128, 4, 1, 0x105, 8, 4),
        # BC4 (fmt 80): 8 bytes/block, 64KB_S (0x109)
        ("BC4 512x256 6-mip 64KB_S", 80, 512, 256, 6, 1, 0x109, 8, 4),
        # BC5 (fmt 83): 16 bytes/block, 64KB_D (0x10A)
        ("BC5 256x256 4-mip 64KB_D", 83, 256, 256, 4, 1, 0x10A, 16, 4),
        # BC7 (fmt 98): 16 bytes/block, 4KB_S (0x105)
        ("BC7 64x64 7-mip 4KB_S", 98, 64, 64, 7, 1, 0x105, 16, 4),
        # BC7 (fmt 98): 16 bytes/block, 4KB_D (0x106)
        ("BC7 81x69 1-mip 4KB_D", 98, 81, 69, 1, 1, 0x106, 16, 4),
        # Uncompressed RGBA8 (fmt 28): 4 bytes/texel, 64KB_S (0x109)
        ("RGBA8 128x128 4-mip 64KB_S", 28, 128, 128, 4, 1, 0x109, 4, 1),
        # Uncompressed R8 (fmt 61): 1 byte/texel, 64KB_S (0x109)
        ("R8 256x256 4-mip 64KB_S", 61, 256, 256, 4, 1, 0x109, 1, 1),
        # Uncompressed R16 (fmt 56): 2 bytes/texel, 4KB_S (0x105)
        ("R16 64x64 3-mip 4KB_S", 56, 64, 64, 3, 1, 0x105, 2, 1),
        # Array of 2 slices: BC7 128x128 3-mip 64KB_S
        ("BC7 128x128 3-mip 2-slice 64KB_S", 98, 128, 128, 3, 2, 0x109, 16, 4),
        # 256B_S swizzle (0x101): R8 64x64 1-mip
        ("R8 64x64 1-mip 256B_S", 61, 64, 64, 1, 1, 0x101, 1, 1),
        # Linear (0x100): RGBA8 64x64 1-mip
        ("RGBA8 64x64 1-mip LINEAR", 28, 64, 64, 1, 1, 0x100, 4, 1),
        # Non-square BC7 128x64 8-mip 4KB_S: mip 7 triggers height >> 7 == 0
        ("BC7 128x64 8-mip 4KB_S", 98, 128, 64, 8, 1, 0x105, 16, 4),
    ]

    for name, fmt, w, h, mips, slices, layout, bpe, bdim in cases:
        test_format(create, name, fmt, w, h, mips, slices, layout, bpe, bdim)
        print(f"  {name}: PASS (layout, tiling, slot 8 tile, slot 9 detile roundtrip exact)")

    test_negative_controls(create)

    print(f"All {len(cases)} GFX10 texture computer test cases and negative controls passed successfully!")
    return 0


def test_negative_controls(create):
    # Valid base descriptor: BC7 64x64, 1-mip, 1-slice, 4KB_S (0x105)
    desc = ct.create_string_buffer(
        DESC.pack(3, 0, 64, 64, 1, 1, 98, 1, 0, 0x105, 0)
    )
    computer = PTR()
    hr = create(desc, ct.byref(computer), 0, 0)
    assert hr == 0 and computer, "Failed to create computer for negative controls"
    slots = ct.cast(computer, ct.POINTER(ct.POINTER(PTR)))[0]

    try:
        total_size = SIZE(slots[3])(computer)
        tiled_buf = ct.create_string_buffer(total_size)
        linear_src = ct.create_string_buffer(16 * 16 * 16)
        row_pitch = 16 * 16  # 256 bytes

        # 1. pitch < row_bytes: row_bytes is 256, pass pitch = 255
        rc = COPY_INTO(slots[8])(computer, tiled_buf, 0, 0, linear_src, row_pitch - 1, len(linear_src))
        assert (rc & 0xFFFFFFFF) == 0x80070057, f"pitch < row_bytes expected E_INVALIDARG, got 0x{rc & 0xFFFFFFFF:08x}"
        rc = COPY_INTO(slots[9])(computer, linear_src, 0, 0, tiled_buf, row_pitch - 1, len(linear_src))
        assert (rc & 0xFFFFFFFF) == 0x80070057, f"pitch < row_bytes expected E_INVALIDARG, got 0x{rc & 0xFFFFFFFF:08x}"

        # 2. Out-of-range plane (plane = 1 on single-plane resource)
        rc = COPY_INTO(slots[8])(computer, tiled_buf, 1, 0, linear_src, row_pitch, len(linear_src))
        assert (rc & 0xFFFFFFFF) == 0x80070057, f"plane > 0 expected E_INVALIDARG, got 0x{rc & 0xFFFFFFFF:08x}"
        rc = COPY_INTO(slots[9])(computer, linear_src, 1, 0, tiled_buf, row_pitch, len(linear_src))
        assert (rc & 0xFFFFFFFF) == 0x80070057, f"plane > 0 expected E_INVALIDARG, got 0x{rc & 0xFFFFFFFF:08x}"

        # 3. Out-of-range subresource (subresource = 1 for 1-mip 1-slice resource)
        rc = COPY_INTO(slots[8])(computer, tiled_buf, 0, 1, linear_src, row_pitch, len(linear_src))
        assert (rc & 0xFFFFFFFF) == 0x80070057, f"subresource >= mips*slices expected E_INVALIDARG, got 0x{rc & 0xFFFFFFFF:08x}"
        rc = COPY_INTO(slots[9])(computer, linear_src, 0, 1, tiled_buf, row_pitch, len(linear_src))
        assert (rc & 0xFFFFFFFF) == 0x80070057, f"subresource >= mips*slices expected E_INVALIDARG, got 0x{rc & 0xFFFFFFFF:08x}"

        # 4. Overlapping source and destination buffers
        overlap_buf = ct.create_string_buffer(total_size)
        rc = COPY_INTO(slots[8])(computer, overlap_buf, 0, 0, overlap_buf, row_pitch, total_size)
        assert (rc & 0xFFFFFFFF) == 0x80070057, f"overlapping src/dst expected E_INVALIDARG, got 0x{rc & 0xFFFFFFFF:08x}"
        rc = COPY_INTO(slots[9])(computer, overlap_buf, 0, 0, overlap_buf, row_pitch, total_size)
        assert (rc & 0xFFFFFFFF) == 0x80070057, f"overlapping src/dst expected E_INVALIDARG, got 0x{rc & 0xFFFFFFFF:08x}"

        # 5. Null pointers
        rc = COPY_INTO(slots[8])(computer, None, 0, 0, linear_src, row_pitch, len(linear_src))
        assert (rc & 0xFFFFFFFF) == 0x80070057, f"null dst expected E_INVALIDARG, got 0x{rc & 0xFFFFFFFF:08x}"
        rc = COPY_INTO(slots[8])(computer, tiled_buf, 0, 0, None, row_pitch, len(linear_src))
        assert (rc & 0xFFFFFFFF) == 0x80070057, f"null src expected E_INVALIDARG, got 0x{rc & 0xFFFFFFFF:08x}"

    finally:
        RELEASE(slots[1])(computer)

    # 6. Descriptor with MipLevels > 16 rejected from GFX10 tiling
    desc_over_mips = ct.create_string_buffer(
        DESC.pack(3, 0, 64, 64, 1, 17, 98, 1, 0, 0x109, 0)
    )
    comp_over = PTR()
    hr = create(desc_over_mips, ct.byref(comp_over), 0, 0)
    if hr == 0 and comp_over:
        s = ct.cast(comp_over, ct.POINTER(ct.POINTER(PTR)))[0]
        align_val = ALIGN(s[4])(comp_over)
        assert align_val != 65536, f"MipLevels=17 unexpectedly accepted by GFX10 computer (align={align_val})"
        RELEASE(s[1])(comp_over)
    print("  Negative controls: PASS (pitch < row_bytes, plane/subresource out of range, overlap, null, mips > 16)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: startup_xg_gfx10_check.py <path_to_xg_xs.dll>")
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
