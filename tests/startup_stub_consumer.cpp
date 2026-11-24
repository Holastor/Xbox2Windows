// Source-built consumer for the startup trap stubs. Imports by name from chad.dll, xg_xs.dll and
// xmem.dll and calls the export the first argument selects. (XMemTransferMemory and
// D3D12ConfigureVirtualMemoryX have real implementations in the shims and their own tests.)
//   none     -> returns 0 without calling a stub (control)
//   chad     -> XDspActivate(0x11, 0x22, 0x33, 0x44)
//   xg_xs    -> XGSuggestSwizzleMode(0x55, 0, 0, 0)
//   physical -> XMemAllocatePhysicalPages/XMemFreePhysicalPages round trip (no trap expected)
//   dstorage -> DirectStorage multi-page RFC 1950 zlib request and failure controls (Refs #1144)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using u64 = std::uint64_t;

extern "C" __declspec(dllimport) u64 XDspActivate(u64, u64, u64, u64);
extern "C" __declspec(dllimport) u64 XGSuggestSwizzleMode(u64, u64, u64, u64);
extern "C" __declspec(dllimport) int XMemAllocatePhysicalPages(u64 flags, std::size_t* count, u64* pages);
extern "C" __declspec(dllimport) int XMemFreePhysicalPages(std::size_t count, u64* pages);
extern "C" __declspec(dllimport) void* WINAPI XMemVirtualAlloc(void* address, SIZE_T bytes, DWORD allocation_type,
                                                              ULONGLONG attributes, DWORD protection);
extern "C" __declspec(dllimport) void* WINAPI XMemMapPhysicalPages(void* base, ULONG_PTR count,
                                                                   const ULONG_PTR* tokens);

namespace {

uint32_t adler32_calc(const unsigned char* data, size_t size) {
    uint32_t a = 1, b = 0;
    while (size) {
        size_t chunk = size < 5552 ? size : 5552;
        size -= chunk;
        while (chunk--) {
            a += *data++;
            b += a;
        }
        a %= 65521;
        b %= 65521;
    }
    return (b << 16) | a;
}

std::vector<unsigned char> make_stored_zlib(const unsigned char* data, size_t size) {
    std::vector<unsigned char> out;
    out.push_back(0x78);
    out.push_back(0x01);
    size_t pos = 0;
    while (pos < size) {
        size_t chunk = (size - pos > 32768) ? 32768 : (size - pos);
        bool last = (pos + chunk == size);
        out.push_back(last ? 0x01 : 0x00);
        out.push_back(static_cast<unsigned char>(chunk & 0xFF));
        out.push_back(static_cast<unsigned char>((chunk >> 8) & 0xFF));
        unsigned short check = static_cast<unsigned short>(chunk ^ 0xFFFF);
        out.push_back(static_cast<unsigned char>(check & 0xFF));
        out.push_back(static_cast<unsigned char>((check >> 8) & 0xFF));
        out.insert(out.end(), data + pos, data + pos + chunk);
        pos += chunk;
    }
    uint32_t adler = adler32_calc(data, size);
    out.push_back(static_cast<unsigned char>((adler >> 24) & 0xFF));
    out.push_back(static_cast<unsigned char>((adler >> 16) & 0xFF));
    out.push_back(static_cast<unsigned char>((adler >> 8) & 0xFF));
    out.push_back(static_cast<unsigned char>(adler & 0xFF));
    return out;
}

struct Object {
    void** vtable;
};

using DStorageGetFactoryFn = long(__stdcall*)(const void*, void**);
using CreateStatusArrayFn = long(__stdcall*)(void*, unsigned, const char*, const void*, void**);
using CreateQueueFn = long(__stdcall*)(void*, const void*, const void*, void**);
using EnqueueRequestFn = void(__stdcall*)(void*, const void*);
using EnqueueStatusFn = void(__stdcall*)(void*, void*, unsigned);
using SubmitFn = void(__stdcall*)(void*);
using IsCompleteFn = bool(__stdcall*)(void*, unsigned);
using GetHResultFn = long(__stdcall*)(void*, unsigned);

bool dstorage_checks() {
    _putenv("X2W_XMEM_PERMISSIVE=1");

    HMODULE xcrdapi = LoadLibraryA("XCRDAPI.dll");
    if (!xcrdapi) return false;
    auto get_factory = reinterpret_cast<DStorageGetFactoryFn>(GetProcAddress(xcrdapi, "DStorageGetFactory"));
    if (!get_factory) return false;

    void* factory_ptr = nullptr;
    if (get_factory(nullptr, &factory_ptr) != 0 || !factory_ptr) return false;
    auto* factory = static_cast<Object*>(factory_ptr);

    const size_t plain_size = 128 * 1024;
    std::vector<unsigned char> plain(plain_size);
    for (size_t i = 0; i < plain_size; ++i) plain[i] = static_cast<unsigned char>((i % 251) ^ 0x5a);
    const auto comp = make_stored_zlib(plain.data(), plain_size);

    const char* cases[] = { "multipage", "short", "missing_grant", "linear_short" };
    for (const char* mode : cases) {
        void* status_ptr = nullptr;
        if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, 1, "Status", nullptr, &status_ptr) != 0 || !status_ptr)
            return false;
        auto* status = static_cast<Object*>(status_ptr);

        void* queue_ptr = nullptr;
        unsigned char desc[0x20] = {};
        *reinterpret_cast<uint16_t*>(desc + 8) = 16;
        if (reinterpret_cast<CreateQueueFn>(factory->vtable[10])(factory, desc, nullptr, &queue_ptr) != 0 || !queue_ptr)
            return false;
        auto* queue = static_cast<Object*>(queue_ptr);

        if (std::strcmp(mode, "linear_short") == 0) {
            std::vector<unsigned char> dest(64 * 1024, 0xAA);
            unsigned char request[0x48] = {};
            *reinterpret_cast<u64*>(request + 0x00) = 1 | 0x400000;
            *reinterpret_cast<void**>(request + 0x08) = dest.data();
            *reinterpret_cast<u64*>(request + 0x10) = 0;
            *reinterpret_cast<unsigned*>(request + 0x18) = static_cast<unsigned>(dest.size());
            *reinterpret_cast<const unsigned char**>(request + 0x20) = comp.data();
            *reinterpret_cast<u64*>(request + 0x28) = 0;
            *reinterpret_cast<unsigned*>(request + 0x30) = static_cast<unsigned>(comp.size());
            *reinterpret_cast<unsigned*>(request + 0x38) = static_cast<unsigned>(plain_size);

            reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, request);
            reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status, 0);
            reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);

            for (int i = 0; i < 200; ++i) {
                if (reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, 0)) break;
                Sleep(10);
            }
            if (!reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, 0)) return false;
            long hr = reinterpret_cast<GetHResultFn>(status->vtable[4])(status, 0);
            if (hr == 0) return false;
            for (unsigned char b : dest) {
                if (b != 0xAA) return false;
            }
        } else {
            size_t page_count = 2;
            u64 pages[2] = {};
            if (!XMemAllocatePhysicalPages(0x20400000ull, &page_count, pages) || page_count != 2) return false;

            ULONG_PTR tokens[2] = { pages[0], pages[1] };
            unsigned destination_size = 128 * 1024;

            void* va = XMemVirtualAlloc(nullptr, 2 * 65536, 0x20402000, 0x21, PAGE_READWRITE);
            if (!va) return false;
            if (XMemMapPhysicalPages(va, 2, tokens) != va) return false;
            std::memset(va, 0xAA, 128 * 1024);

            if (std::strcmp(mode, "short") == 0) {
                destination_size = 64 * 1024;
            } else if (std::strcmp(mode, "missing_grant") == 0) {
                tokens[1] = 0;
            }

            unsigned char request[0x48] = {};
            *reinterpret_cast<u64*>(request + 0x00) = 1 | (1ull << 21) | 0x400000;
            *reinterpret_cast<ULONG_PTR**>(request + 0x08) = tokens;
            *reinterpret_cast<u64*>(request + 0x10) = 0;
            *reinterpret_cast<unsigned*>(request + 0x18) = destination_size;
            *reinterpret_cast<const unsigned char**>(request + 0x20) = comp.data();
            *reinterpret_cast<u64*>(request + 0x28) = 0;
            *reinterpret_cast<unsigned*>(request + 0x30) = static_cast<unsigned>(comp.size());
            *reinterpret_cast<unsigned*>(request + 0x38) = static_cast<unsigned>(plain_size);

            reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, request);
            reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status, 0);
            reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);

            for (int i = 0; i < 200; ++i) {
                if (reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, 0)) break;
                Sleep(10);
            }
            if (!reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, 0)) return false;
            long hr = reinterpret_cast<GetHResultFn>(status->vtable[4])(status, 0);

            if (std::strcmp(mode, "multipage") == 0) {
                if (hr != 0) return false;
                if (std::memcmp(va, plain.data(), plain_size) != 0) return false;
            } else {
                if (hr == 0) return false;
                const auto* bytes = static_cast<const unsigned char*>(va);
                for (size_t i = 0; i < 128 * 1024; ++i) {
                    if (bytes[i] != 0xAA) return false;
                }
            }

            XMemMapPhysicalPages(va, 2, nullptr);
            VirtualFree(va, 0, MEM_RELEASE);
            XMemFreePhysicalPages(2, pages);
        }
    }

    // Test chunked streaming reads across 4 requests (Refs #1136)
    {
        const size_t total_chunks = 4;
        const size_t chunk_dest_size = 64 * 1024;
        const size_t total_plain_size = total_chunks * chunk_dest_size; // 256 KiB
        std::vector<unsigned char> full_plain(total_plain_size);
        for (size_t i = 0; i < total_plain_size; ++i) full_plain[i] = static_cast<unsigned char>((i % 239) ^ 0xa5);
        const auto full_comp = make_stored_zlib(full_plain.data(), total_plain_size);

        void* status_ptr = nullptr;
        if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, static_cast<unsigned>(total_chunks), "ChunkedStatus", nullptr, &status_ptr) != 0 || !status_ptr)
            return false;
        auto* status = static_cast<Object*>(status_ptr);

        void* queue_ptr = nullptr;
        unsigned char desc[0x20] = {};
        *reinterpret_cast<uint16_t*>(desc + 8) = 16;
        if (reinterpret_cast<CreateQueueFn>(factory->vtable[10])(factory, desc, nullptr, &queue_ptr) != 0 || !queue_ptr)
            return false;
        auto* queue = static_cast<Object*>(queue_ptr);

        std::vector<std::vector<unsigned char>> chunks(total_chunks, std::vector<unsigned char>(chunk_dest_size, 0));
        for (size_t k = 0; k < total_chunks; ++k) {
            unsigned char req[0x48] = {};
            *reinterpret_cast<u64*>(req + 0x00) = 1 | 0x400000; // zlib | memory_source
            *reinterpret_cast<void**>(req + 0x08) = chunks[k].data();
            *reinterpret_cast<u64*>(req + 0x10) = 0;
            *reinterpret_cast<unsigned*>(req + 0x18) = static_cast<unsigned>(chunk_dest_size);
            *reinterpret_cast<const unsigned char**>(req + 0x20) = full_comp.data();
            *reinterpret_cast<u64*>(req + 0x28) = k * chunk_dest_size; // offset stepped by destination size
            *reinterpret_cast<unsigned*>(req + 0x30) = static_cast<unsigned>(full_comp.size());
            *reinterpret_cast<unsigned*>(req + 0x38) = static_cast<unsigned>(total_plain_size);

            reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, req);
            reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status, static_cast<unsigned>(k));
        }
        reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);

        for (int i = 0; i < 200; ++i) {
            if (reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, static_cast<unsigned>(total_chunks - 1))) break;
            Sleep(10);
        }
        for (size_t k = 0; k < total_chunks; ++k) {
            if (!reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, static_cast<unsigned>(k))) return false;
            long hr = reinterpret_cast<GetHResultFn>(status->vtable[4])(status, static_cast<unsigned>(k));
            if (hr != 0) return false;
            if (std::memcmp(chunks[k].data(), full_plain.data() + k * chunk_dest_size, chunk_dest_size) != 0) return false;
        }
    }

    // Negative control: chunk request with out-of-bounds offset must fail and leave destination untouched
    {
        const size_t total_plain_size = 128 * 1024;
        std::vector<unsigned char> test_plain(total_plain_size, 0x55);
        const auto test_comp = make_stored_zlib(test_plain.data(), total_plain_size);

        void* status_ptr = nullptr;
        if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, 2, "ChunkNegStatus", nullptr, &status_ptr) != 0 || !status_ptr)
            return false;
        auto* status = static_cast<Object*>(status_ptr);

        void* queue_ptr = nullptr;
        unsigned char desc[0x20] = {};
        *reinterpret_cast<uint16_t*>(desc + 8) = 16;
        if (reinterpret_cast<CreateQueueFn>(factory->vtable[10])(factory, desc, nullptr, &queue_ptr) != 0 || !queue_ptr)
            return false;
        auto* queue = static_cast<Object*>(queue_ptr);

        std::vector<unsigned char> dest0(64 * 1024, 0);
        std::vector<unsigned char> dest1(64 * 1024, 0xEE); // sentinel

        // Piece 0
        unsigned char req0[0x48] = {};
        *reinterpret_cast<u64*>(req0 + 0x00) = 1 | 0x400000;
        *reinterpret_cast<void**>(req0 + 0x08) = dest0.data();
        *reinterpret_cast<unsigned*>(req0 + 0x18) = 64 * 1024;
        *reinterpret_cast<const unsigned char**>(req0 + 0x20) = test_comp.data();
        *reinterpret_cast<u64*>(req0 + 0x28) = 0;
        *reinterpret_cast<unsigned*>(req0 + 0x30) = static_cast<unsigned>(test_comp.size());
        *reinterpret_cast<unsigned*>(req0 + 0x38) = static_cast<unsigned>(total_plain_size);

        // Piece 1 with invalid offset (500 KiB > 128 KiB)
        unsigned char req1[0x48] = {};
        *reinterpret_cast<u64*>(req1 + 0x00) = 1 | 0x400000;
        *reinterpret_cast<void**>(req1 + 0x08) = dest1.data();
        *reinterpret_cast<unsigned*>(req1 + 0x18) = 64 * 1024;
        *reinterpret_cast<const unsigned char**>(req1 + 0x20) = test_comp.data();
        *reinterpret_cast<u64*>(req1 + 0x28) = 500 * 1024;
        *reinterpret_cast<unsigned*>(req1 + 0x30) = static_cast<unsigned>(test_comp.size());
        *reinterpret_cast<unsigned*>(req1 + 0x38) = static_cast<unsigned>(total_plain_size);

        reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, req0);
        reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status, 0);
        reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, req1);
        reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status, 1);
        reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);

        for (int i = 0; i < 200; ++i) {
            if (reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, 1)) break;
            Sleep(10);
        }
        if (!reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, 1)) return false;
        long hr1 = reinterpret_cast<GetHResultFn>(status->vtable[4])(status, 1);
        if (hr1 == 0) return false;
        for (unsigned char b : dest1) {
            if (b != 0xEE) return false;
        }
    }

    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 2) return 2;
    const char* which = argv[1];
    if (!std::strcmp(which, "none")) return 0;
    if (!std::strcmp(which, "chad")) return static_cast<int>(XDspActivate(0x11, 0x22, 0x33, 0x44));
    if (!std::strcmp(which, "xg_xs")) return static_cast<int>(XGSuggestSwizzleMode(0x55, 0, 0, 0));
    if (!std::strcmp(which, "physical")) {
        std::size_t count = 4;
        u64 pages[4] = {};
        if (!XMemAllocatePhysicalPages(0x20400000ull /* 64 KiB pages */, &count, pages) || count != 4) return 3;
        for (u64 page : pages)
            if (!page) return 4;
        if (!XMemFreePhysicalPages(count, pages)) return 5;
        if (XMemFreePhysicalPages(count, pages)) return 6;  // double free must be refused
        std::puts("physical ok");
        return 0;
    }
    if (!std::strcmp(which, "dstorage")) {
        if (!dstorage_checks()) return 7;
        std::puts("dstorage ok");
        return 0;
    }
    return 2;
}
