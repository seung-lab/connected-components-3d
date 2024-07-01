#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Foundation/Foundation.h>

@interface MetalCCL : NSObject

- (instancetype)initWithDevice:(id<MTLDevice>)device;
- (void)connectedComponentsWithInput:(id<MTLTexture>)inputTexture output:(id<MTLTexture>)outputTexture;

@end

@implementation MetalCCL {
    id<MTLDevice> _device;
    id<MTLCommandQueue> _commandQueue;
    id<MTLLibrary> _library;
    id<MTLComputePipelineState> _initPipeline;
    id<MTLComputePipelineState> _propagatePipeline;
    id<MTLComputePipelineState> _checkPipeline;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device {
    self = [super init];
    if (self) {
        _device = device;
        _commandQueue = [_device newCommandQueue];

        NSError *error = nil;
        NSString *librarySource = [NSString stringWithContentsOfFile:@"connected_components.metal" encoding:NSUTF8StringEncoding error:&error];
        _library = [_device newLibraryWithSource:librarySource options:nil error:&error];

        id<MTLFunction> initFunction = [_library newFunctionWithName:@"init_labels"];
        id<MTLFunction> propagateFunction = [_library newFunctionWithName:@"propagate_labels"];
        id<MTLFunction> checkFunction = [_library newFunctionWithName:@"check_convergence"];
        
        _initPipeline = [_device newComputePipelineStateWithFunction:initFunction error:&error];
        _propagatePipeline = [_device newComputePipelineStateWithFunction:propagateFunction error:&error];
        _checkPipeline = [_device newComputePipelineStateWithFunction:checkFunction error:&error];
    }
    return self;
}

- (void)connectedComponentsWithInput:(id<MTLTexture>)inputTexture output:(id<MTLTexture>)outputTexture {
    NSUInteger width = inputTexture.width;
    NSUInteger height = inputTexture.height;
    
    id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
    
    MTLSize gridSize = MTLSizeMake(width, height, 1);
    MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);

    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    
    // Initialization
    [computeEncoder setComputePipelineState:_initPipeline];
    [computeEncoder setTexture:outputTexture atIndex:0];
    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    [computeEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    BOOL converged = NO;
    while (!converged) {
        commandBuffer = [_commandQueue commandBuffer];
        computeEncoder = [commandBuffer computeCommandEncoder];
        
        // Propagation
        [computeEncoder setComputePipelineState:_propagatePipeline];
        [computeEncoder setTexture:inputTexture atIndex:0];
        [computeEncoder setTexture:outputTexture atIndex:1];
        [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
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
        [computeEncoder setTexture:inputTexture atIndex:0];
        [computeEncoder setTexture:outputTexture atIndex:1];
        [computeEncoder setBuffer:convergedBuffer offset:0 atIndex:0];
        [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [computeEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        if (*convergedPtr == 0) {
            converged = NO;
        }
    }
}

@end
