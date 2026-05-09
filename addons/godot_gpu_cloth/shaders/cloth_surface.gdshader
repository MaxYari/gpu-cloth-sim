shader_type spatial;
render_mode cull_disabled;

// Positions and normals are written by the compute pipeline every frame.
// The vertex stage reads them directly — no CPU readback involved.
uniform sampler2D positions_tex : filter_nearest;
uniform sampler2D normals_tex   : filter_nearest;
uniform int       tex_width     = 4096;

uniform sampler2D albedo_texture : source_color, filter_linear_mipmap;
uniform vec4  color_tint : source_color = vec4(1.0, 1.0, 1.0, 1.0);
uniform float roughness  : hint_range(0.0, 1.0) = 0.9;
uniform float metallic   : hint_range(0.0, 1.0) = 0.0;
uniform float specular   : hint_range(0.0, 1.0) = 0.1;

void vertex() {
	ivec2 coord = ivec2(int(VERTEX_ID) % tex_width, int(VERTEX_ID) / tex_width);
	VERTEX = texelFetch(positions_tex, coord, 0).xyz;
	NORMAL = texelFetch(normals_tex,   coord, 0).xyz;
}

void fragment() {
	vec4 tex         = texture(albedo_texture, UV);
	vec4 final_color = tex * color_tint;
	ALBEDO    = final_color.rgb;
	ROUGHNESS = roughness;
	METALLIC  = metallic;
	SPECULAR  = specular;
	if (!FRONT_FACING) {
		NORMAL = -NORMAL;
		DEPTH  = FRAGCOORD.z + 0.0001;
	}
}
