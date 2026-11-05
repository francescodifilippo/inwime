#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#ifdef BENCH_CUPTI
#include "activity_trace.hpp"
#endif
#ifdef BENCH_NVTX
#include <nvToolsExt.h>
#endif
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/resource.h>
#include <unistd.h>
#include <vector>

using Clock = std::chrono::steady_clock;
void range_begin(const char* name) {
#ifdef BENCH_NVTX
    nvtxRangePushA(name);
#else
    (void)name;
#endif
}
void range_end() {
#ifdef BENCH_NVTX
    nvtxRangePop();
#endif
}
double ms(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}
void ck(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e));
}
void cb(cublasStatus_t e) {
    if (e != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS status " + std::to_string(e));
}
void sysfail(const std::string& s) { throw std::runtime_error(s + ": " + std::strerror(errno)); }
struct Fd {
    int value;
    explicit Fd(int v): value(v) { if (v < 0) sysfail("open"); }
    ~Fd() { close(value); }
};
struct Buffer {
    inline static size_t pinned_bytes=0, peak_pinned_bytes=0;
    void* p = nullptr;
    bool pinned = false;
    explicit Buffer(size_t n, bool pin = false) {
        if (posix_memalign(&p, 4096, n)) throw std::runtime_error("posix_memalign failed");
        if (pin) {
            auto e = cudaHostRegister(p, n, cudaHostRegisterDefault);
            if (e != cudaSuccess) { free(p); p = nullptr; ck(e); }
            pinned = true;
            pinned_bytes+=n; peak_pinned_bytes=std::max(peak_pinned_bytes,pinned_bytes);
            registered_bytes=n;
        }
    }
    size_t registered_bytes=0;
    ~Buffer() { if (pinned) { cudaHostUnregister(p); pinned_bytes-=registered_bytes; } free(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};
struct Config {
    int m=1024, k=4096, n=16, blocks=64, repeats=3, threads=8;
    std::string file="data/weights.bf16", mode="all";
    bool prepare=false, full=false, profile=false;
    size_t cache_mib=0;
    size_t elems() const { return size_t(m)*k; }
    size_t bytes() const { return elems()*2; }
    size_t total() const { return bytes()*blocks; }
};
int positive(const std::string& s) {
    size_t used=0;
    long long v=std::stoll(s,&used);
    if (used!=s.size() || v<1 || v>1048576) throw std::runtime_error("invalid positive integer: "+s);
    return int(v);
}
Config args(int argc, char** argv) {
    Config c;
    for (int i=1;i<argc;++i) {
        std::string a=argv[i];
        if (a=="--prepare") { c.prepare=true; continue; }
        if (a=="--full-check") { c.full=true; continue; }
        if (a=="--profile") { c.profile=true; continue; }
        if (a=="--help") {
            std::cout << "bf16_stream [--prepare] [--file PATH] [--mode all|cpu|sync|async|io]\n"
                         "  [--m 1024] [--k 4096] [--n 16] [--blocks 64]\n"
                         "  [--repeats 3] [--threads 8] [--full-check]\n"
                         "  [--cache-mib 0] [--profile (single GPU pass; cudaProfilerApi)]\n"
                         "Preparation creates a new file exclusively. Block bytes must be divisible by 4096.\n";
            std::exit(0);
        }
        if (++i==argc) throw std::runtime_error("missing value for "+a);
        std::string v=argv[i];
        if (a=="--file") c.file=v;
        else if (a=="--mode") c.mode=v;
        else if (a=="--m") c.m=positive(v);
        else if (a=="--k") c.k=positive(v);
        else if (a=="--n") c.n=positive(v);
        else if (a=="--blocks") c.blocks=positive(v);
        else if (a=="--repeats") c.repeats=positive(v);
        else if (a=="--threads") c.threads=positive(v);
        else if (a=="--cache-mib") c.cache_mib=v=="0"?0:positive(v);
        else throw std::runtime_error("unknown option: "+a);
    }
    if (c.mode!="all" && c.mode!="cpu" && c.mode!="sync" && c.mode!="async" && c.mode!="io")
        throw std::runtime_error("invalid mode");
    if(c.cache_mib>2048) throw std::runtime_error("cache limit is 2048 MiB");
    if(c.profile && (c.repeats!=1 || (c.mode!="sync" && c.mode!="async")))
        throw std::runtime_error("--profile requires one GPU-only repetition");
    // Bound all allocations and file offsets before doing any I/O.
    if (c.bytes()%4096 || c.bytes()>256ULL*1024*1024 || size_t(c.k)*c.n>16*1024*1024 ||
        size_t(c.m)*c.n>16*1024*1024 || c.total()>1024ULL*1024*1024*1024)
        throw std::runtime_error("unaligned or excessive dimensions (block <=256 MiB, file <=1 TiB)");
    return c;
}
float decode(uint16_t b) { uint32_t u=uint32_t(b)<<16; float f; std::memcpy(&f,&u,4); return f; }
uint16_t encode(float f) { uint32_t u; std::memcpy(&u,&f,4); return uint16_t(u>>16); }
// Values are exact BF16 multiples of 1/16. No quantization pipeline or model is used.
float value(uint64_t i) {
    i += 0x9e3779b97f4a7c15ULL;
    i = (i^(i>>30))*0xbf58476d1ce4e5b9ULL;
    i = (i^(i>>27))*0x94d049bb133111ebULL;
    i ^= i>>31;
    return (int(i%33)-16)/16.0f;
}
float weight(const Config& c,int block,size_t j) { return value(uint64_t(block)*c.elems()+j); }
float activation(size_t j) { return value(j ^ 0xd1b54a32d192ed03ULL); }
void read_block(int fd, void* p, size_t n, off_t offset) {
    range_begin("pread");
#ifdef BENCH_CUPTI
    uint64_t read_start=activity_trace::active?activity_trace::timestamp():0;
#endif
    // O_DIRECT short reads are rejected: silently retrying a misaligned remainder is unsafe.
    ssize_t got;
    do { got=pread(fd,p,n,offset); } while (got<0 && errno==EINTR);
    range_end();
#ifdef BENCH_CUPTI
    if(activity_trace::active) activity_trace::record("pread",read_start,activity_trace::timestamp(),n,0);
#endif
    if (got<0) sysfail("pread O_DIRECT");
    if (size_t(got)!=n) throw std::runtime_error("short read or truncated dataset");
}
void prepare(const Config& c) {
    Fd fd(open(c.file.c_str(),O_WRONLY|O_CREAT|O_EXCL,0644));
    Buffer a(c.bytes());
    auto* p=static_cast<uint16_t*>(a.p);
    for (int b=0;b<c.blocks;++b) {
        #pragma omp parallel for
        for (size_t j=0;j<c.elems();++j) p[j]=encode(weight(c,b,j));
        size_t done=0;
        while (done<c.bytes()) {
            ssize_t got=write(fd.value,static_cast<char*>(a.p)+done,c.bytes()-done);
            if (got<0 && errno==EINTR) continue;
            if (got<=0) sysfail("write");
            done+=size_t(got);
        }
    }
    if (fsync(fd.value)) sysfail("fsync");
    std::cerr << "Prepared " << c.total() << " bytes at " << c.file << '\n';
}
struct Metrics {
    double read=0, h2d=0, compute=0, d2h=0, total=0, error=0;
    size_t checked=0, read_bytes=0, h2d_bytes=0, hits=0, device_bytes=0;
    size_t gpu_used_bytes=0;
};
struct Sample { int block; size_t index; float actual; };
std::vector<size_t> indices(const Config& c) {
    size_t count=size_t(c.m)*c.n;
    std::vector<size_t> ids;
    if (c.full) { for(size_t i=0;i<count;++i) ids.push_back(i); }
    else {
        ids={0,count-1,count/2};
        for(size_t i=0;i<29;++i) ids.push_back((i*2654435761ULL+17)%count);
        std::sort(ids.begin(),ids.end()); ids.erase(std::unique(ids.begin(),ids.end()),ids.end());
    }
    return ids;
}
void capture(std::vector<Sample>& samples,const std::vector<size_t>& ids,int b,const float* out) {
    for(auto i:ids) samples.push_back({b,i,out[i]});
}
void validate(const Config& c,const std::vector<Sample>& samples,Metrics& t) {
    for (const auto& s:samples) {
        size_t row=s.index%c.m, col=s.index/c.m;
        double expected=0;
        for (int k=0;k<c.k;++k) expected+=double(weight(c,s.block,row*c.k+k))*activation(col*c.k+k);
        double err=std::abs(double(s.actual)-expected);
        if (!std::isfinite(s.actual) || err>1e-4+1e-5*std::abs(expected))
            throw std::runtime_error("validation failed at block "+std::to_string(s.block)+" index "+std::to_string(s.index));
        t.error=std::max(t.error,err); ++t.checked;
    }
}
void cpu_gemm(const Config& c,const uint16_t* a,const float* b,float* out) {
    #pragma omp parallel for collapse(2)
    for(int col=0;col<c.n;++col) for(int row=0;row<c.m;++row) {
        float sum=0;
        #pragma omp simd reduction(+:sum)
        for(int k=0;k<c.k;++k) sum+=decode(a[size_t(row)*c.k+k])*b[size_t(col)*c.k+k];
        out[size_t(col)*c.m+row]=sum;
    }
}
Metrics cpu(const Config& c,int fd,bool io) {
    Buffer a(c.bytes());
    std::vector<float> b(size_t(c.k)*c.n), out(size_t(c.m)*c.n);
    for(size_t i=0;i<b.size();++i) b[i]=activation(i);
    auto ids=indices(c); std::vector<Sample> samples;
    samples.reserve(size_t(c.blocks)*ids.size());
    // Warm up CPU threads and GEMM outside timing.
    if(!io) { read_block(fd,a.p,c.bytes(),0); cpu_gemm(c,static_cast<uint16_t*>(a.p),b.data(),out.data()); }
    Metrics t; auto start=Clock::now();
    for(int block=0;block<c.blocks;++block) {
        auto r=Clock::now(); read_block(fd,a.p,c.bytes(),off_t(block)*c.bytes()); t.read+=ms(r);
        t.read_bytes+=c.bytes();
        if(!io) {
            auto g=Clock::now(); cpu_gemm(c,static_cast<uint16_t*>(a.p),b.data(),out.data()); t.compute+=ms(g);
            capture(samples,ids,block,out.data());
        }
    }
    t.total=ms(start); validate(c,samples,t); return t;
}
struct Slot {
    Buffer host, output;
    void* device=nullptr; float* result=nullptr;
    cudaStream_t stream=nullptr; cublasHandle_t handle=nullptr;
    std::array<cudaEvent_t,4> event{};
    int block=-1;
    explicit Slot(const Config& c):host(c.bytes(),true),output(size_t(c.m)*c.n*4,true) {}
    void init(const Config& c) {
        ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        cb(cublasCreate(&handle)); cb(cublasSetStream(handle,stream));
        cb(cublasSetMathMode(handle,CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
        ck(cudaMalloc(&device,c.bytes())); ck(cudaMalloc(&result,size_t(c.m)*c.n*4));
        for(auto& e:event) ck(cudaEventCreate(&e));
    }
    ~Slot() {
        if(stream) cudaStreamSynchronize(stream);
        if(handle) cublasDestroy(handle);
        for(auto e:event) if(e) cudaEventDestroy(e);
        if(device) cudaFree(device);
        if(result) cudaFree(result);
        if(stream) cudaStreamDestroy(stream);
    }
};
struct DeviceB {
    void* p=nullptr;
    ~DeviceB(){ if(p) cudaFree(p); }
};
struct WeightCache {
    DeviceB storage;
    size_t bytes=0;
    bool initialized=false;
};
void submit(const Config& c,Slot& s,const void* b,const void* cached=nullptr) {
    ck(cudaEventRecord(s.event[0],s.stream));
    if(!cached) ck(cudaMemcpyAsync(s.device,s.host.p,c.bytes(),cudaMemcpyHostToDevice,s.stream));
    ck(cudaEventRecord(s.event[1],s.stream));
    float alpha=1, beta=0;
    // A stored [m][k]; interpreted column-major k x m, then transposed.
    cb(cublasGemmEx(s.handle,CUBLAS_OP_T,CUBLAS_OP_N,c.m,c.n,c.k,&alpha,
        cached?cached:s.device,CUDA_R_16BF,c.k,b,CUDA_R_16BF,c.k,&beta,s.result,CUDA_R_32F,c.m,
        CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
    ck(cudaEventRecord(s.event[2],s.stream));
    ck(cudaMemcpyAsync(s.output.p,s.result,size_t(c.m)*c.n*4,cudaMemcpyDeviceToHost,s.stream));
    ck(cudaEventRecord(s.event[3],s.stream));
}
void finish(Slot& s,Metrics& t,const std::vector<size_t>& ids,std::vector<Sample>& samples) {
    if(s.block<0) return;
    ck(cudaEventSynchronize(s.event[3]));
    float elapsed;
    ck(cudaEventElapsedTime(&elapsed,s.event[0],s.event[1])); t.h2d+=elapsed;
    ck(cudaEventElapsedTime(&elapsed,s.event[1],s.event[2])); t.compute+=elapsed;
    ck(cudaEventElapsedTime(&elapsed,s.event[2],s.event[3])); t.d2h+=elapsed;
    capture(samples,ids,s.block,static_cast<float*>(s.output.p)); s.block=-1;
}
Metrics gpu(const Config& c,int fd,bool async,WeightCache& cache) {
    size_t free_bytes,total_bytes; ck(cudaMemGetInfo(&free_bytes,&total_bytes));
    size_t cache_bytes=std::min(c.total(),(c.cache_mib*1024*1024/c.bytes())*c.bytes());
    size_t required=2*(c.bytes()+size_t(c.m)*c.n*4)+size_t(c.k)*c.n*2;
    if(!cache.initialized) required+=cache_bytes;
    if(required+256ULL*1024*1024>free_bytes) throw std::runtime_error("insufficient free VRAM");
    DeviceB b;
    std::vector<uint16_t> hb(size_t(c.k)*c.n);
    for(size_t i=0;i<hb.size();++i) hb[i]=encode(activation(i));
    ck(cudaMalloc(&b.p,hb.size()*2)); ck(cudaMemcpy(b.p,hb.data(),hb.size()*2,cudaMemcpyHostToDevice));
    // Both modes allocate identical resources. Sync waits after every submitted block.
    std::array<std::unique_ptr<Slot>,2> slots;
    for(auto& s:slots) {
        s=std::make_unique<Slot>(c); s->init(c);
        read_block(fd,s->host.p,c.bytes(),0); submit(c,*s,b.p); ck(cudaStreamSynchronize(s->stream));
    }
    if(!cache.initialized) {
        auto begin=Clock::now();
        cache.bytes=cache_bytes;
        if(cache.bytes) {
            ck(cudaMalloc(&cache.storage.p,cache.bytes));
            for(size_t offset=0;offset<cache.bytes;offset+=c.bytes()) {
                read_block(fd,slots[0]->host.p,c.bytes(),off_t(offset));
                ck(cudaMemcpy(static_cast<char*>(cache.storage.p)+offset,slots[0]->host.p,
                              c.bytes(),cudaMemcpyHostToDevice));
            }
        }
        cache.initialized=true;
        std::cerr << "cache_setup_bytes=" << cache.bytes << " cache_setup_ms=" << ms(begin) << '\n';
    }
    auto ids=indices(c); std::vector<Sample> samples;
    samples.reserve(size_t(c.blocks)*ids.size());
    Metrics t; t.device_bytes=2*(c.bytes()+size_t(c.m)*c.n*4)+hb.size()*2+cache.bytes;
    ck(cudaMemGetInfo(&free_bytes,&total_bytes)); t.gpu_used_bytes=total_bytes-free_bytes;
    if(c.profile) {
#ifdef BENCH_CUPTI
        activity_trace::start();
#endif
        ck(cudaProfilerStart());
    }
    range_begin("pipeline");
    auto start=Clock::now();
    for(int block=0;block<c.blocks;++block) {
        Slot& s=*slots[block%2];
        finish(s,t,ids,samples); // Never overwrite a buffer still used by DMA or GEMM.
        const void* cached=nullptr;
        if(size_t(block)*c.bytes()<cache.bytes) {
            cached=static_cast<char*>(cache.storage.p)+size_t(block)*c.bytes(); ++t.hits;
        } else {
            auto r=Clock::now(); read_block(fd,s.host.p,c.bytes(),off_t(block)*c.bytes()); t.read+=ms(r);
            t.read_bytes+=c.bytes(); t.h2d_bytes+=c.bytes();
        }
        s.block=block; submit(c,s,b.p,cached);
        if(!async) finish(s,t,ids,samples);
    }
    for(auto& s:slots) finish(*s,t,ids,samples);
    t.total=ms(start); range_end();
    if(c.profile) {
        ck(cudaProfilerStop());
#ifdef BENCH_CUPTI
        activity_trace::stop();
#endif
    }
    validate(c,samples,t); return t;
}
void report(const Config& c,const std::string& mode,int repeat,const Metrics& t) {
    double gib=double(c.total())/(1024*1024*1024);
    double ops=2.0*c.m*c.k*c.n*c.blocks;
    std::cout << std::fixed << std::setprecision(6) << mode << ',' << repeat << ',' << c.total() << ','
        << t.read << ',' << (t.read?double(t.read_bytes)/(1ULL<<30)*1000/t.read:0) << ',' << t.h2d << ','
        << (t.h2d?double(t.h2d_bytes)/(1ULL<<30)*1000/t.h2d:0) << ',' << t.compute << ',' << (t.compute?ops/t.compute/1e9:0) << ','
        << t.d2h << ',' << t.total << ',' << gib*1000/t.total << ',' << t.checked << ',' << t.error;
    struct rusage usage{}; if(getrusage(RUSAGE_SELF,&usage)) sysfail("getrusage");
    std::cout << ',' << c.m << ',' << c.k << ',' << c.n << ',' << c.bytes() << ',' << c.cache_mib
              << ',' << t.read_bytes << ',' << t.h2d_bytes << ',' << t.hits
              << ',' << Buffer::peak_pinned_bytes << ',' << t.device_bytes << ',' << t.gpu_used_bytes
              << ',' << usage.ru_maxrss*1024ULL << '\n';
    std::cout.flush();
}
int main(int argc,char** argv) {
    try {
        Config c=args(argc,argv); omp_set_dynamic(0); omp_set_num_threads(c.threads);
        if(c.prepare) { prepare(c); return 0; }
        Fd fd(open(c.file.c_str(),O_RDONLY|O_DIRECT));
        struct stat st{}; if(fstat(fd.value,&st)) sysfail("fstat");
        if(!S_ISREG(st.st_mode) || uint64_t(st.st_size)!=c.total()) throw std::runtime_error("dataset size/type mismatch");
        std::cerr << "m=" << c.m << " k=" << c.k << " n=" << c.n << " blocks=" << c.blocks
                  << " threads=" << c.threads << " direct_io=1 block_bytes=" << c.bytes() << '\n';
        if(c.mode=="all" || c.mode=="sync" || c.mode=="async") {
            int count=0; ck(cudaGetDeviceCount(&count));
            if(!count) throw std::runtime_error("no CUDA device");
            cudaDeviceProp p{}; ck(cudaGetDeviceProperties(&p,0));
            if(p.major<8) throw std::runtime_error("native BF16 requires compute capability >=8");
            std::cerr << "GPU=" << p.name << " cc=" << p.major << '.' << p.minor
                      << " async_engines=" << p.asyncEngineCount << '\n';
            int driver=0,runtime=0;
            ck(cudaDriverGetVersion(&driver)); ck(cudaRuntimeGetVersion(&runtime));
            size_t available=0,capacity=0; ck(cudaMemGetInfo(&available,&capacity));
            std::cerr << "driver_api=" << driver << " runtime=" << runtime
                      << " free_vram_bytes=" << available << " total_vram_bytes=" << capacity << '\n';
        }
        std::cout << "mode,repeat,bytes,read_ms,read_GiB_s,h2d_ms,h2d_GiB_s,compute_ms,compute_TFLOP_s,d2h_ms,total_ms,effective_GiB_s,checked,max_abs_error,m,k,n,block_bytes,cache_mib,read_bytes,h2d_bytes,cache_hits,pinned_peak_bytes,device_alloc_bytes,gpu_used_bytes,rss_peak_bytes\n";
        WeightCache cache;
        for(int r=0;r<c.repeats;++r) {
            if(c.mode=="all" || c.mode=="io") report(c,"io",r,cpu(c,fd.value,true));
            // Rotate mode order between repeats to reduce systematic order effects.
            std::array<std::string,3> modes={"cpu","sync","async"};
            for(int j=0;j<3;++j) {
                auto mode=modes[(j+r)%3];
                if(c.mode!="all" && c.mode!=mode) continue;
                report(c,mode,r,mode=="cpu"?cpu(c,fd.value,false):gpu(c,fd.value,mode=="async",cache));
            }
        }
        return 0;
    } catch(const std::exception& e) { std::cerr << "ERROR: " << e.what() << '\n'; return 1; }
}
