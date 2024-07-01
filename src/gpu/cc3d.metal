#include <metal_stdlib>
using namespace metal;

kernel void init_labels(
    texture2d<uint, access::read_write> labelTexture [[texture(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= labelTexture.get_width() || gid.y >= labelTexture.get_height()) return;
    
    uint label = gid.y * labelTexture.get_width() + gid.x + 1; // +1 to avoid label 0
    labelTexture.write(label, gid);
}


kernel void propagate_labels(
    texture2d<uint, access::read> inputTexture [[texture(0)]],
    texture2d<uint, access::read_write> labelTexture [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= labelTexture.get_width() || gid.y >= labelTexture.get_height()) return;

    uint currentLabel = labelTexture.read(gid);
    uint minLabel = currentLabel;

    if (gid.x > 0) {
        uint leftLabel = labelTexture.read(uint2(gid.x - 1, gid.y));
        minLabel = min(minLabel, leftLabel);
    }
    if (gid.y > 0) {
        uint topLabel = labelTexture.read(uint2(gid.x, gid.y - 1));
        minLabel = min(minLabel, topLabel);
    }

    if (minLabel < currentLabel) {
        labelTexture.write(minLabel, gid);
    }
}

kernel void check_convergence(
    texture2d<uint, access::read> inputTexture [[texture(0)]],
    texture2d<uint, access::read> labelTexture [[texture(1)]],
    device uint *converged [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= labelTexture.get_width() || gid.y >= labelTexture.get_height()) return;

    uint inputLabel = inputTexture.read(gid);
    uint currentLabel = labelTexture.read(gid);

    if (inputLabel != currentLabel) {
        atomic_store_explicit(converged, 0, memory_order_relaxed);
    }
}



