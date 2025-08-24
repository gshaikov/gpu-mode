// matmul.mm
// Objective-C++ implementation for Metal-accelerated matrix multiplication

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <dlfcn.h>

extern "C" int matmul_metal(const float* A, const float* B, float* C, int M, int N, int K) {
	@autoreleasepool {
		id<MTLDevice> device = MTLCreateSystemDefaultDevice();
		if (!device) {
			std::cerr << "No Metal device found, falling back to CPU" << std::endl;
			return 1;
		}
		// Find the path to the shared library
		Dl_info info;
		if (dladdr((const void*)&matmul_metal, &info) == 0) {
			std::cerr << "Failed to get shared library path" << std::endl;
			return 2;
		}
		std::string lib_path = info.dli_fname;
		std::string::size_type slash = lib_path.find_last_of("/");
		std::string dir = (slash == std::string::npos) ? "." : lib_path.substr(0, slash);
		std::string metal_path = dir + "/cpp/matmul.metal";
		std::ifstream file(metal_path);
		if (!file.is_open()) {
			std::cerr << "Failed to open matmul.metal at " << metal_path << std::endl;
			return 3;
		}
		std::stringstream buffer;
		buffer << file.rdbuf();
		NSString* src = [NSString stringWithUTF8String:buffer.str().c_str()];
		NSError* error = nil;
		id<MTLLibrary> library = [device newLibraryWithSource:src options:nil error:&error];
		if (!library) {
			std::cerr << "Failed to compile Metal shader: " << [[error localizedDescription] UTF8String] << std::endl;
			return 4;
		}
		id<MTLFunction> kernel = [library newFunctionWithName:@"matmul"];
		if (!kernel) {
			std::cerr << "Failed to get matmul kernel" << std::endl;
			return 5;
		}
		id<MTLCommandQueue> queue = [device newCommandQueue];
		id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:kernel error:&error];
		if (!pipeline) {
			std::cerr << "Failed to create pipeline: " << [[error localizedDescription] UTF8String] << std::endl;
			return 6;
		}
		size_t a_bytes = M * K * sizeof(float);
		size_t b_bytes = K * N * sizeof(float);
		size_t c_bytes = M * N * sizeof(float);
		id<MTLBuffer> a_buf = [device newBufferWithBytes:A length:a_bytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> b_buf = [device newBufferWithBytes:B length:b_bytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> c_buf = [device newBufferWithLength:c_bytes options:MTLResourceStorageModeShared];
		uint32_t m = M, n = N, k = K;
		id<MTLBuffer> m_buf = [device newBufferWithBytes:&m length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
		id<MTLBuffer> n_buf = [device newBufferWithBytes:&n length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
		id<MTLBuffer> k_buf = [device newBufferWithBytes:&k length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
		id<MTLCommandBuffer> cmd = [queue commandBuffer];
		id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
		[encoder setComputePipelineState:pipeline];
		[encoder setBuffer:a_buf offset:0 atIndex:0];
		[encoder setBuffer:b_buf offset:0 atIndex:1];
		[encoder setBuffer:c_buf offset:0 atIndex:2];
		[encoder setBuffer:m_buf offset:0 atIndex:3];
		[encoder setBuffer:n_buf offset:0 atIndex:4];
		[encoder setBuffer:k_buf offset:0 atIndex:5];
		MTLSize grid = MTLSizeMake(N, M, 1);
		MTLSize threadgroup = MTLSizeMake(8, 8, 1);
		[encoder dispatchThreads:grid threadsPerThreadgroup:threadgroup];
		[encoder endEncoding];
		[cmd commit];
		[cmd waitUntilCompleted];
		std::memcpy(C, [c_buf contents], c_bytes);
		return 0;
	}
}
