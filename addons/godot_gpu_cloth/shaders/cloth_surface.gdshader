shader_type spatial;
render_mode cull_disabled;

uniform sampler2D albedo_texture : source_color, filter_linear_mipmap;
uniform vec4 color_tint : source_color = vec4(1.0, 1.0, 1.0, 1.0);
uniform float roughness : hint_range(0.0, 1.0) = 0.9;
uniform float metallic : hint_range(0.0, 1.0) = 0.0;
uniform float specular : hint_range(0.0, 1.0) = 0.1;

void fragment() {
	vec4 tex = texture(albedo_texture, UV);
	vec4 final_color = tex * color_tint;
	ALBEDO = final_color.rgb;
	ROUGHNESS = roughness;
	METALLIC = metallic;
	SPECULAR = specular;
	if (!FRONT_FACING) {
		NORMAL = -NORMAL;
		DEPTH = FRAGCOORD.z + 0.0001;
	}
}
