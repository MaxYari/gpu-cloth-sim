#[compute]
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 1, std430) restrict buffer Predicted        { vec4 predicted[];   };
layout(set = 0, binding = 3, std430) restrict readonly buffer Constraints { vec4 constraints[]; };

layout(push_constant, std430) uniform Params {
    float dt;
    float gravity;
    uint  particle_count;
    uint  constraint_count;
    float damping;
    float max_speed;
    uint  collider_count;
    uint  constraint_offset;
    float pad3, pad4, pad5, pad6;
    float pad7, pad8, pad9, pad10;
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= constraint_count) return;

    uint cidx = constraint_offset + idx;

    uint a = uint(constraints[cidx].x);
    uint b = uint(constraints[cidx].y);
    float rest = constraints[cidx].z;
    float stiffness = constraints[cidx].w;

    vec3 pa = predicted[a].xyz;
    vec3 pb = predicted[b].xyz;
    float wa = predicted[a].w;
    float wb = predicted[b].w;

    vec3 delta = pb - pa;
    float dist = length(delta);

    if (dist < 1e-7) return;

    float w_sum = wa + wb;
    if (w_sum < 1e-7) return;

    float err = (dist - rest) / dist;
    vec3 correction = delta * err * stiffness;

    predicted[a] = vec4(pa + correction * (wa / w_sum), wa);
    predicted[b] = vec4(pb - correction * (wb / w_sum), wb);
}
