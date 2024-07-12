#include "cc3d_mps_impl.h"

@implementation MetalCCL {
    id<MTLDevice> _device;
    id<MTLCommandQueue> _commandQueue;
    id<MTLLibrary> _library;
    id<MTLComputePipelineState> _initPipeline;
    id<MTLComputePipelineState> _propagatePipeline;
    id<MTLComputePipelineState> _checkPipeline;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        NSError *error = nil;

        _device = MTLCreateSystemDefaultDevice();
        if (!_device) {
            @throw [NSException exceptionWithName:@"MetalCCLInitException" reason:@"Metal device not found." userInfo:nil];
        }

        // Create a default library
        id<MTLLibrary> defaultLibrary = [_device newDefaultLibrary];
        if (!defaultLibrary) {
            @throw [NSException exceptionWithName:@"MetalCCLInitException" reason:@"Failed to create default library." userInfo:nil];
        }

        id<MTLFunction> initFunction = [_library newFunctionWithName:@"init_labels"];
        id<MTLFunction> propagateFunction = [_library newFunctionWithName:@"propagate_labels"];
        id<MTLFunction> checkFunction = [_library newFunctionWithName:@"check_convergence"];
        
        _initPipeline = [_device newComputePipelineStateWithFunction:initFunction error:&error];
        if (error) {
            @throw [NSException exceptionWithName:@"MetalCCLInitException" reason:@"Failed to create 'init_labels' pipeline state." userInfo:nil];
        }
        
        _propagatePipeline = [_device newComputePipelineStateWithFunction:propagateFunction error:&error];
        if (error) {
            @throw [NSException exceptionWithName:@"MetalCCLInitException" reason:@"Failed to create 'propagate_labels' pipeline state." userInfo:nil];
        }

        _checkPipeline = [_device newComputePipelineStateWithFunction:checkFunction error:&error];
        if (error) {
            @throw [NSException exceptionWithName:@"MetalCCLInitException" reason:@"Failed to create 'check_convergence' pipeline state." userInfo:nil];
        }

        _commandQueue = [_device newCommandQueue];
    }
    return self;
}

- (void)connectedComponents4WithLabels:(uint64_t)labelsAddr
                                  sx:(NSUInteger)sx
                                  sy:(NSUInteger)sy
                                output:(uint64_t)outputAddr
                                  osx:(NSUInteger)osx
                                  osy:(NSUInteger)osy {
    

    uint8_t* labels = reinterpret_cast<uint8_t*>(labelsAddr);
    uint32_t* output = reinterpret_cast<uint32_t*>(outputAddr);

    id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    
    // Set input and output buffers
    id<MTLBuffer> labelsBuffer = [_device newBufferWithBytes:labels length:sx * sy * sizeof(uint8_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> outputBuffer = [_device newBufferWithBytes:output length:osx * osy * sizeof(uint32_t) options:MTLResourceStorageModeShared];
    
    [computeEncoder setBuffer:labelsBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:outputBuffer offset:0 atIndex:1];

    // Encode compute command
    MTLSize gridSize = MTLSizeMake(sx, sy, 1);
    MTLSize threadsPerGroup = MTLSizeMake(16, 16, 1); // Adjust as per your shader requirements
    MTLSize threadGroups = MTLSizeMake((sy + threadsPerGroup.width - 1) / threadsPerGroup.width, (sx + threadsPerGroup.height - 1) / threadsPerGroup.height, 1);
    [computeEncoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadGroups];

    // Initialization
    [computeEncoder setComputePipelineState:_initPipeline];
    [computeEncoder setBuffer:outputBuffer offset:0 atIndex:0];
    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroups];
    [computeEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    BOOL converged = NO;
    while (!converged) {
        commandBuffer = [_commandQueue commandBuffer];
        computeEncoder = [commandBuffer computeCommandEncoder];
        
        // Propagation
        [computeEncoder setComputePipelineState:_propagatePipeline];
        [computeEncoder setBuffer:labelsBuffer offset:0 atIndex:0];
        [computeEncoder setBuffer:outputBuffer offset:0 atIndex:1];
        [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroups];
        [computeEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Check Convergence
        converged = YES;
        commandBuffer = [_commandQueue commandBuffer];
        computeEncoder = [commandBuffer computeCommandEncoder];
        
        id<MTLBuffer> convergedBuffer = [_device newBufferWithLength:sizeof(uint) options:MTLResourceStorageModeShared];
        uint *convergedPtr = (uint *)[convergedBuffer contents];
        *convergedPtr = 1;

        [computeEncoder setComputePipelineState:_checkPipeline];
        [computeEncoder setBuffer:labelsBuffer offset:0 atIndex:0];
        [computeEncoder setBuffer:outputBuffer offset:0 atIndex:1];
        [computeEncoder setBuffer:convergedBuffer offset:0 atIndex:2];
        [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroups];
        [computeEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        if (*convergedPtr == 0) {
            converged = NO;
        }
    }
}

@end

void connected_components_4_mps(
    uint64_t labelsAddr,
    uint64_t sx, uint64_t sy,
    uint64_t outputAddr,
    uint64_t osx, uint64_t osy
) {

    if (!MTL) {
        MTL = [[MetalCCL alloc] init];
    }

    [MTL connectedComponents4WithLabels:labelsAddr sx:sx sy:sy output:outputAddr osx:osx osy:osy];
}


void cleanup () {
    [MTL dealloc];
}
