/*
 * d2co - Direct Directional Chamfer Optimization
 *
 *  Copyright (c) 2020, Alberto Pretto <alberto.pretto@flexsight.eu>
 *                      Marco Imperoli <marco.imperoli@flexsight.eu>
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

/* Most of the code here has been taken from:
 * 
 * https://github.com/SonarSystems/Modern-OpenGL-Tutorials
 * 
 */

#define OBJECT_VERTEX_SHADER_CODE(VERSION) \
"\
#version "+VERSION+"\
\n\
in vec3 vertex_pos; \
\
uniform mat4 model;\
uniform mat4 view;\
uniform mat4 projection;\
\
void main()\
{\
  gl_Position = projection * view *  model * vec4(vertex_pos, 1.0f); \
}\
"

#define COLORED_OBJECT_VERTEX_SHADER_CODE(VERSION) \
"\
#version "+VERSION+"\
\n\
in vec3 vertex_pos; \
in vec3 vertex_color; \
\
out vec3 frag_color; \
\
uniform mat4 model;\
uniform mat4 view;\
uniform mat4 projection;\
\
void main()\
{\
  gl_Position = projection * view *  model * vec4(vertex_pos, 1.0f); \
  frag_color = vertex_color; \
}\
"

#define SHADED_OBJECT_VERTEX_SHADER_CODE(VERSION) \
"\
#version "+VERSION+"\
\n in vec3 vertex_pos; \
in vec3 vertex_normal; \
in vec3 vertex_color; \
\
out vec3 frag_pos; \
out vec3 frag_normal; \
out vec3 frag_color; \
\
uniform mat4 model;\
uniform mat4 view;\
uniform mat4 projection;\
\
void main()\
{\
  gl_Position = projection * view *  model * vec4(vertex_pos, 1.0f); \
  frag_pos = vec3(model * vec4(vertex_pos, 1.0f));\
  frag_normal = mat3(transpose(inverse(model))) * vertex_normal;\
  frag_color = vertex_color; \
}\
"

#define COLORED_OBJECT_FRAGMENT_SHADER_CODE(VERSION) \
"\
#version "+VERSION+"\
\n\
in vec3 frag_color; \
\
out vec4 frag_out; \
\
void main()\
{\
  frag_out = vec4(frag_color, 1);\
}\
"



#define SHADED_OBJECT_FRAGMENT_SHADER_CODE(VERSION) \
"\
#version "+VERSION+"\
\n\
#define NUMBER_OF_POINT_LIGHTS 2\
\n\
#define NUMBER_OF_DIRECTIONAL_LIGHTS 1\
\n\
struct Material \
{\
  sampler2D diffuse;\
  sampler2D specular;\
  float shininess;\
};\
\
struct DirLight\
{\
  vec3 direction;\
\
  vec3 ambient;\
  vec3 diffuse;\
  vec3 specular;\
};\
\
struct PointLight\
{\
  vec3 position;\
\
  float constant;\
  float linear;\
  float quadratic;\
\
  vec3 ambient;\
  vec3 diffuse;\
  vec3 specular;\
};\
\
in vec3 frag_pos;\
in vec3 frag_normal;\
in vec3 frag_color;\
\
out vec4 frag_out;\
\
uniform vec3 view_pos;\
uniform DirLight dir_lights[NUMBER_OF_DIRECTIONAL_LIGHTS];\
uniform PointLight point_lights[NUMBER_OF_POINT_LIGHTS];\
uniform Material material;\
\
vec3 Calcdir_light( DirLight light, vec3 normal, vec3 view_dir );\
vec3 CalcPointLight( PointLight light, vec3 normal, vec3 frag_pos, vec3 view_dir );\
\
void main()\
{\
  vec3 norm = normalize( frag_normal );\
  vec3 view_dir = normalize( view_pos - frag_pos );\
\
  vec3 result = vec3(0.0, 0.0, 0.0 );\
\
  for ( int i = 0; i < NUMBER_OF_DIRECTIONAL_LIGHTS; i++ )\
  {\
    result += Calcdir_light( dir_lights[i], norm, view_dir );\
  }\
\
  for ( int i = 0; i < NUMBER_OF_POINT_LIGHTS; i++ )\
  {\
    result += CalcPointLight( point_lights[i], norm, frag_pos, view_dir );\
  }\
\
  frag_out = vec4( result, 1.0 );\
}\
\
vec3 Calcdir_light( DirLight light, vec3 normal, vec3 view_dir )\
{\
  vec3 light_dir = normalize( -light.direction );\
\
  float diff = max( dot( normal, light_dir ), 0.0 );\
\
  vec3 reflect_dir = reflect( -light_dir, normal );\
  float spec = pow( max( dot( view_dir, reflect_dir ), 0.0 ), material.shininess );\
\
  vec3 ambient = light.ambient * frag_color;\
  vec3 diffuse = light.diffuse * diff * frag_color;\
  vec3 specular = light.specular * spec * frag_color;\
\
  return ( ambient + diffuse + specular );\
}\
\
vec3 CalcPointLight( PointLight light, vec3 normal, vec3 frag_pos, vec3 view_dir )\
{\
  vec3 light_dir = normalize( light.position - frag_pos );\
\
  float diff = max( dot( normal, light_dir ), 0.0 );\
\
  vec3 reflect_dir = reflect( -light_dir, normal );\
  float spec = pow( max( dot( view_dir, reflect_dir ), 0.0 ), material.shininess );\
\
  float distance = length( light.position - frag_pos );\
  float attenuation = 1.0f / ( light.constant + light.linear * distance + light.quadratic * ( distance * distance ) );\
\
  vec3 ambient = light.ambient * frag_color;\
  vec3 diffuse = light.diffuse * diff * frag_color;\
  vec3 specular = light.specular * spec * frag_color;\
\
  ambient *= attenuation;\
  diffuse *= attenuation;\
  specular *= attenuation;\
\
  return ( ambient + diffuse + specular );\
}\
"

// struct SpotLight
// {
//   vec3 position;
//   vec3 direction;
//   float cutOff;
//   float outerCutOff;
// 
//   float constant;
//   float linear;
//   float quadratic;
//   
//   vec3 ambient;
//   vec3 diffuse;
//   vec3 specular;
// };
// 
// uniform SpotLight spot_light;
// 
// vec3 Calcspot_light( SpotLight light, vec3 normal, vec3 frag_pos, vec3 view_dir )
// {  
//   vec3 light_dir = normalize( light.position - frag_pos );
//   
//   // Diffuse shading
//   float diff = max( dot( normal, light_dir ), 0.0 );
// 
//   // Specular shading
//   vec3 reflect_dir = reflect( -light_dir, normal );
//   float spec = pow( max( dot( view_dir, reflect_dir ), 0.0 ), material.shininess );
// 
//   // Attenuation
//   float distance = length( light.position - frag_pos );
//   float attenuation = 1.0f / ( light.constant + light.linear * distance + light.quadratic * ( distance * distance ) );
//   
//   // SpotLight intensity
//   float theta = dot( light_dir, normalize( -light.direction ) );
//   float epsilon = light.cutOff - light.outerCutOff;
//   float intensity = clamp( ( theta - light.outerCutOff ) / epsilon, 0.0, 1.0 );
// 
//   // Combine results
//   vec3 ambient = light.ambient * frag_color );
//   vec3 diffuse = light.diffuse * diff * frag_color );
//   vec3 specular = light.specular * spec * frag_color );
// 
//   ambient *= attenuation * intensity;
//   diffuse *= attenuation * intensity;
//   specular *= attenuation * intensity;
// 
//   return ( ambient + diffuse + specular );
// }
