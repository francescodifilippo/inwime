// Optional CUPTI activity tracing, compiled only into bf16_stream_trace.
#pragma once
#include <cupti.h>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>

namespace activity_trace {
inline FILE* output=nullptr;
inline bool active=false;
inline std::mutex lock;
inline void check(CUptiResult result) {
    if(result!=CUPTI_SUCCESS) {
        const char* message="unknown CUPTI error";
        cuptiGetResultString(result,&message);
        throw std::runtime_error(std::string("CUPTI: ")+message);
    }
}
inline uint64_t timestamp() { uint64_t t=0; check(cuptiGetTimestamp(&t)); return t; }
inline void record(const char* kind,uint64_t begin,uint64_t end,uint64_t bytes,uint32_t stream) {
    std::lock_guard<std::mutex> guard(lock);
    std::fprintf(output,"%s,%llu,%llu,%llu,%u\n",kind,
                 (unsigned long long)begin,(unsigned long long)end,(unsigned long long)bytes,stream);
}
inline void CUPTIAPI request(uint8_t** buffer,size_t* size,size_t* max_records) {
    *size=8*1024*1024; *buffer=static_cast<uint8_t*>(std::malloc(*size)); *max_records=0;
    if(!*buffer) std::abort();
}
inline void CUPTIAPI complete(CUcontext context,uint32_t stream,uint8_t* buffer,size_t,size_t valid) {
    CUpti_Activity* item=nullptr;
    CUptiResult status;
    while((status=cuptiActivityGetNextRecord(buffer,valid,&item))==CUPTI_SUCCESS) {
        if(item->kind==CUPTI_ACTIVITY_KIND_MEMCPY) {
            auto* r=reinterpret_cast<CUpti_ActivityMemcpy5*>(item);
            record(r->copyKind==CUPTI_ACTIVITY_MEMCPY_KIND_HTOD?"h2d":
                   r->copyKind==CUPTI_ACTIVITY_MEMCPY_KIND_DTOH?"d2h":"copy",
                   r->start,r->end,r->bytes,r->streamId);
        } else if(item->kind==CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL || item->kind==CUPTI_ACTIVITY_KIND_KERNEL) {
            auto* r=reinterpret_cast<CUpti_ActivityKernel9*>(item);
            record("kernel",r->start,r->end,0,r->streamId);
        }
    }
    size_t dropped=0;
    auto result=cuptiActivityGetNumDroppedRecords(context,stream,&dropped);
    if(status!=CUPTI_ERROR_MAX_LIMIT_REACHED || result!=CUPTI_SUCCESS || dropped)
        std::fprintf(stderr,"CUPTI callback status=%d dropped=%zu query_status=%d\n",int(status),dropped,int(result));
    std::free(buffer);
}
inline void start() {
    const char* path=std::getenv("BENCH_TRACE_FILE");
    if(!path) throw std::runtime_error("BENCH_TRACE_FILE is required for CUPTI profiling");
    output=std::fopen(path,"wx");
    if(!output) throw std::runtime_error("cannot exclusively create trace output");
    std::fprintf(output,"kind,start_ns,end_ns,bytes,stream\n");
    check(cuptiActivityRegisterCallbacks(request,complete));
    check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
    check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    active=true;
}
inline void stop() {
    active=false;
    check(cuptiActivityFlushAll(0));
    check(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    check(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY));
    std::fclose(output); output=nullptr;
}
}
