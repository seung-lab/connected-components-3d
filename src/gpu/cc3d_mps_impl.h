#ifndef CC3D_MPS_H_
#define CC3D_MPS_H_

#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <Foundation/Foundation.h>

@interface MetalCCL : NSObject

- (instancetype)init;
- (void)connectedComponents4WithLabels:(uint64_t)labelsAddr
                                  sx:(NSUInteger)sx
                                  sy:(NSUInteger)sy
                                output:(uint64_t)outputAddr
                                  osx:(NSUInteger)osx
                                  osy:(NSUInteger)osy;
@end

static MetalCCL* MTL;

void connected_components_4_mps(
    uint64_t labelsAddr,
    uint64_t sx, uint64_t sy,
    uint64_t outputAddr,
    uint64_t osx, uint64_t osy
);

void cleanup();


#endif