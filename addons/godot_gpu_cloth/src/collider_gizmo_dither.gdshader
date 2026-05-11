shader_type spatial;
render_mode unshaded, cull_disabled, blend_mix;

uniform vec4 albedo : source_color = vec4(0.0, 0.85, 1.0, 0.15);

void fragment() {
	ALBEDO = albedo.rgb;
	ALPHA = albedo.a;
}
