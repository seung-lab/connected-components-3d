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

        // NSLog(@"Hello world!");

        _device = MTLCreateSystemDefaultDevice();
        if (!_device) {
            @throw [NSException exceptionWithName:@"MetalCCLInitException" reason:@"Metal device not found." userInfo:nil];
        }


        // NSLog(@"device init");

        // Create a default library
        _library = [_device newLibraryWithFile:@"/Users/wms/code/connected-components-3d/src/gpu/default.metallib" error:&error];
        // NSLog(@"fucked");
        if (_library == nil) {
            @throw [NSException exceptionWithName:@"MetalCCLInitException" reason:@"Failed to create default library." userInfo:nil];
        }

        // NSLog(@"default lib init");

        id<MTLFunction> initFunction = [_library newFunctionWithName:@"init_labels"];
        // id<MTLFunction> propagateFunction = [_library newFunctionWithName:@"propagate_labels"];
        // id<MTLFunction> checkFunction = [_library newFunctionWithName:@"check_convergence"];
        
        _initPipeline = [_device newComputePipelineStateWithFunction:initFunction error:&error];
        if (error) {
            @throw [NSException exceptionWithName:@"MetalCCLInitException" reason:@"Failed to create 'init_labels' pipeline state." userInfo:nil];
        }

        // NSLog(@"init pipeline init");
        
        // _propagatePipeline = [_device newComputePipelineStateWithFunction:propagateFunction error:&error];
        // if (error) {
        //     @throw [NSException exceptionWithName:@"MetalCCLInitException" reason:@"Failed to create 'propagate_labels' pipeline state." userInfo:nil];
        // }

        // NSLog(@"propagate init");

        // _checkPipeline = [_device newComputePipelineStateWithFunction:checkFunction error:&error];
        // if (error) {
        //     @throw [NSException exceptionWithName:@"MetalCCLInitException" reason:@"Failed to create 'check_convergence' pipeline state." userInfo:nil];
        // }

        // NSLog(@"check pipeline init");

        _commandQueue = [_device newCommandQueue];

        // NSLog(@"command queue init");
    }

    // NSLog(@"successful init!");

    return self;
}

- (void)connectedComponents4WithLabels:(uint64_t)labelsAddr
                                  sx:(NSUInteger)sx
                                  sy:(NSUInteger)sy
                                output:(uint64_t)outputAddr {
    

    // WARNING!!! labels and output are pointers to the address
    // on the GPU but if you access them here they will be treated
    // as on the CPU. DON'T DO THAT. It will segfault.
    int8_t* labels = reinterpret_cast<int8_t*>(labelsAddr);
    int32_t* output = reinterpret_cast<int32_t*>(outputAddr);

    id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    
    // Create the MTLBuffer without making a copy
    id<MTLBuffer> labelsBuffer = [_device newBufferWithBytesNoCopy:labels
                                                          length:sx * sy * sizeof(int8_t)
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];

    id<MTLBuffer> outputBuffer = [_device newBufferWithBytesNoCopy:output
                                                          length:sx * sy * sizeof(int32_t)
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];
    // // Encode compute command
    MTLSize gridSize = MTLSizeMake(sx, sy, 1);
    MTLSize threadsPerGroup = MTLSizeMake(16, 16, 1); // Adjust as per your shader requirements
    MTLSize threadGroups = MTLSizeMake(
        (sx + threadsPerGroup.width - 1) / threadsPerGroup.width,
        (sy + threadsPerGroup.height - 1) / threadsPerGroup.height,
        1
    );

    // Initialization
    [computeEncoder setComputePipelineState:_initPipeline];
    [computeEncoder setBuffer:outputBuffer offset:0 atIndex:0];
    [computeEncoder setBytes:&sx length:sizeof(int32_t) atIndex:1];
    [computeEncoder setBytes:&sy length:sizeof(int32_t) atIndex:2];
    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroups];
    [computeEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

@end

void connected_components_4_mps(
    uint64_t labelsAddr,
    uint64_t sx, uint64_t sy,
    uint64_t outputAddr
) {
    if (!MTL) {
        MTL = [[MetalCCL alloc] init];
    }

    [MTL connectedComponents4WithLabels:labelsAddr sx:sx sy:sy output:outputAddr];
}


void cleanup () {
    [MTL dealloc];
}
