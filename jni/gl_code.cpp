/*
 * Copyright (C) 2009 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// OpenGL ES 2.0 code

#include <jni.h>
#include <android/log.h>

#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>



#define  LOG_TAG    "libgl2jni"
#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)
#define TEST_YUV_WIDTH   1920
#define TEST_YUV_HEIGHT  1080

enum {
	UNIFORM_PROJ_MATRIX = 0,
	UNIFORM_ROTATION,
	UNIFORM_TEXTURE_Y,
	UNIFORM_TEXTURE_U,
	UNIFORM_TEXTURE_V,
	NUM_UNIFORMS
};

enum {
	Y,
	U,
	V
};

enum {
	ATTRIB_VERTEX = 0,
	ATTRIB_UV,
	NUM_ATTRIBS
};


static char* s_pYBuffer = NULL;
static char* s_pUBuffer = NULL;
static char* s_pVBuffer = NULL;
GLuint gProgram;
GLuint gvPositionHandle;
GLuint spriteTexture[3];
GLint uniforms[NUM_UNIFORMS];
GLint backingWidth;
GLint backingHeight;
int gl_width;
int gl_height;
static bool bNeedInit;

static unsigned int align_on_power_of_2(unsigned int value) {
	int i;
	/* browse all power of 2 value, and find the one just >= value */
	for(i=0; i<32; i++) {
		unsigned int c = 1 << i;
		if (value <= c)
			return c;
	}
	return 0;
}

static void load_orthographic_matrix(float left, float right, float bottom, float top, float near, float far, float* mat)
{
	float r_l = right - left;
	float t_b = top - bottom;
	float f_n = far - near;
	float tx = - (right + left) / (right - left);
	float ty = - (top + bottom) / (top - bottom);
	float tz = - (far + near) / (far - near);

	mat[0] = (2.0f / r_l);
	mat[1] = mat[2] = mat[3] = 0.0f;

	mat[4] = 0.0f;
	mat[5] = (2.0f / t_b);
	mat[6] = mat[7] = 0.0f;

	mat[8] = mat[9] = 0.0f;
	mat[10] = -2.0f / f_n;
	mat[11] = 0.0f;

	mat[12] = tx;
	mat[13] = ty;
	mat[14] = tz;
	mat[15] = 1.0f;
}


static void printGLString(const char *name, GLenum s) {
	const char *v = (const char *) glGetString(s);
	LOGI("GL %s = %s\n", name, v);
}

static void checkGlError(const char* op) {
	for (GLint error = glGetError(); error; error
	= glGetError()) {
		LOGI("after %s() glError (0x%x)\n", op, error);
	}
}

static const char gVertexShader[] = 
		"attribute vec2 position;							\n"
		"attribute vec2 uv;									\n"
		"uniform mat4 proj_matrix;							\n"
		"uniform float rotation;							\n"
		"varying vec2 uvVarying;							\n"
		"void main()										\n"
		"{													\n"
		"	mat3 rot = mat3(vec3(cos(rotation), sin(rotation),0.0), vec3(-sin(rotation), cos(rotation), 0.0), vec3(0.0, 0.0, 1.0));	\n"
		"	gl_Position = proj_matrix * vec4(rot * vec3(position.xy, 0.0), 1.0);	\n"
		"	uvVarying = uv;									\n"
		"}													\n";

static const char gFragmentShader[] = 
		"precision mediump float;							\n"
		"uniform sampler2D t_texture_y;						\n"
		"uniform sampler2D t_texture_u;						\n"
		"uniform sampler2D t_texture_v;						\n"
		"varying vec2 uvVarying;							\n"
		"void main()										\n"
		"{													\n"
		"	float y,u,v,r,g,b, gradx, grady;				\n"
		"	y = texture2D(t_texture_y, uvVarying).r;		\n"
		"	u = texture2D(t_texture_u, uvVarying).r;		\n"
		"	v = texture2D(t_texture_v, uvVarying).r;		\n"
		"	y = 1.16438355 * (y - 0.0625);					\n"
		"	u = u - 0.5;									\n"
		"	v = v - 0.5;									\n"
		"	r = clamp(y + 1.596 * v, 0.0, 1.0);				\n"
		"	g = clamp(y - 0.391 * u - 0.813 * v, 0.0, 1.0);	\n"
		"	b = clamp(y + 2.018 * u, 0.0, 1.0);				\n"
		"	gl_FragColor = vec4(r,g,b,1.0);					\n"
		"}													\n"
		;

GLuint loadShader(GLenum shaderType, const char* pSource) {
	GLuint shader = glCreateShader(shaderType);
	if (shader) {
		glShaderSource(shader, 1, &pSource, NULL);
		glCompileShader(shader);
		GLint compiled = 0;
		glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
		if (!compiled) {
			GLint infoLen = 0;
			glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLen);
			if (infoLen) {
				char* buf = (char*) malloc(infoLen);
				if (buf) {
					glGetShaderInfoLog(shader, infoLen, NULL, buf);
					LOGE("Could not compile shader %d:\n%s\n",
							shaderType, buf);
					free(buf);
				}
				glDeleteShader(shader);
				shader = 0;
			}
		}
	}
	return shader;
}

GLuint createProgram(const char* pVertexSource, const char* pFragmentSource) {
	GLuint vertexShader = loadShader(GL_VERTEX_SHADER, pVertexSource);
	if (!vertexShader) {
		return 0;
	}

	GLuint pixelShader = loadShader(GL_FRAGMENT_SHADER, pFragmentSource);
	if (!pixelShader) {
		return 0;
	}

	GLuint program = glCreateProgram();
	if (program) {
		glAttachShader(program, vertexShader);
		checkGlError("glAttachShader");
		glAttachShader(program, pixelShader);
		checkGlError("glAttachShader");

		glBindAttribLocation(program, ATTRIB_VERTEX, "position");
		checkGlError("glBindAttribLocation");
		glBindAttribLocation(program, ATTRIB_UV, "uv");
		checkGlError("glBindAttribLocation");

		glLinkProgram(program);
		GLint linkStatus = GL_FALSE;
		glGetProgramiv(program, GL_LINK_STATUS, &linkStatus);
		if (linkStatus != GL_TRUE) {
			GLint bufLength = 0;
			glGetProgramiv(program, GL_INFO_LOG_LENGTH, &bufLength);
			if (bufLength) {
				char* buf = (char*) malloc(bufLength);
				if (buf) {
					glGetProgramInfoLog(program, bufLength, NULL, buf);
					LOGE("Could not link program:\n%s\n", buf);
					free(buf);
				}
			}
			glDeleteProgram(program);
			program = 0;
		}
	}
	return program;
}



bool setupGraphics(int w, int h) {
	printGLString("Version", GL_VERSION);
	printGLString("Vendor", GL_VENDOR);
	printGLString("Renderer", GL_RENDERER);
	printGLString("Extensions", GL_EXTENSIONS);

	LOGI("Init YUV buffers");
	s_pYBuffer = (char*)malloc(TEST_YUV_WIDTH*TEST_YUV_HEIGHT);
	s_pUBuffer = (char*)malloc(TEST_YUV_WIDTH*TEST_YUV_HEIGHT/4);
	s_pVBuffer = (char*)malloc(TEST_YUV_WIDTH*TEST_YUV_HEIGHT/4);
	if(s_pYBuffer == NULL || s_pUBuffer == NULL || s_pVBuffer == NULL)
	{
		LOGE("can't malloc memory for YUV buffer\n");
		goto out;
	}

	glGenTextures(3, spriteTexture);
	checkGlError("glGenTextures");

	LOGI("setupGraphics(%d, %d)", w, h);
	gProgram = createProgram(gVertexShader, gFragmentShader);
	if (!gProgram) {
		LOGE("Could not create program.");
		return false;
	}

	uniforms[UNIFORM_PROJ_MATRIX] = glGetUniformLocation(gProgram, "proj_matrix");
	checkGlError("glGetUniformLocation");
	uniforms[UNIFORM_ROTATION] = glGetUniformLocation(gProgram, "rotation");
	checkGlError("glGetUniformLocation");
	uniforms[UNIFORM_TEXTURE_Y] = glGetUniformLocation(gProgram, "t_texture_y");
	checkGlError("glGetUniformLocation");
	uniforms[UNIFORM_TEXTURE_U] = glGetUniformLocation(gProgram, "t_texture_u");
	checkGlError("glGetUniformLocation");
	uniforms[UNIFORM_TEXTURE_V] = glGetUniformLocation(gProgram, "t_texture_v");
	checkGlError("glGetUniformLocation");

	glViewport(0, 0, w, h);
	backingWidth = w;
	backingHeight = h;
	checkGlError("glViewport");
	bNeedInit = true;
	return true;
	out:
	if(s_pYBuffer)
	{
		free(s_pYBuffer);
		s_pYBuffer = NULL;
	}
	if(s_pUBuffer)
	{
		free(s_pUBuffer);
		s_pUBuffer = NULL;
	}
	if(s_pVBuffer)
	{
		free(s_pVBuffer);
		s_pVBuffer = NULL;
	}
	return false;
}

const GLfloat gTriangleVertices[] = { 0.0f, 0.5f, -0.5f, -0.5f,
		0.5f, -0.5f };

void renderFrame() {
	gl_width = align_on_power_of_2(TEST_YUV_WIDTH);
	gl_height = align_on_power_of_2(TEST_YUV_HEIGHT);
	LOGI("renderFrame (gl_width %d,gl_height %d)", gl_width, gl_height);
	if (bNeedInit)
	{
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, spriteTexture[Y]);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, gl_width, gl_height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, 0);

		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, spriteTexture[U]);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, gl_width >> 1, gl_height >> 1, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, 0);

		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, spriteTexture[V]);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, gl_width >> 1, gl_height >> 1, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, 0);

		bNeedInit = false;
	}
	float uLeft = 0.0f, vBottom = 0.0f;
	float uRight = TEST_YUV_WIDTH / (float)(gl_width + 1);
	float vTop = TEST_YUV_HEIGHT / (float)(gl_height + 1);

	int x,y,w,h;
	float xpos = 0.0f, ypos = 0.0f;

	GLfloat squareVertices[8];
	GLfloat squareUvs[] = {
			uLeft, vTop,
			uRight, vTop,
			uLeft, vBottom,
			uRight, vBottom
	};

	glUseProgram(gProgram);
	static unsigned char color = 0;

	memset(s_pYBuffer, color, TEST_YUV_WIDTH*TEST_YUV_HEIGHT);
	memset(s_pUBuffer, color, TEST_YUV_WIDTH*TEST_YUV_HEIGHT/4);
	memset(s_pVBuffer, color, TEST_YUV_WIDTH*TEST_YUV_HEIGHT/4);
	color++;
	/* upload Y plane */
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, spriteTexture[Y]);
	glTexSubImage2D(GL_TEXTURE_2D, 0,
			0, 0, TEST_YUV_WIDTH, TEST_YUV_HEIGHT,
			GL_LUMINANCE, GL_UNSIGNED_BYTE, s_pYBuffer);
	glUniform1i(uniforms[UNIFORM_TEXTURE_Y], 0);

	/* upload U plane */
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, spriteTexture[U]);
	glTexSubImage2D(GL_TEXTURE_2D, 0,
			0, 0, TEST_YUV_WIDTH/2, TEST_YUV_HEIGHT >> 1,
			GL_LUMINANCE, GL_UNSIGNED_BYTE, s_pUBuffer);
	glUniform1i(uniforms[UNIFORM_TEXTURE_U], 1);

	/* upload V plane */
	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, spriteTexture[V]);
	glTexSubImage2D(GL_TEXTURE_2D, 0,
			0, 0, TEST_YUV_WIDTH/2, TEST_YUV_HEIGHT >> 1,
			GL_LUMINANCE, GL_UNSIGNED_BYTE, s_pVBuffer);
	glUniform1i(uniforms[UNIFORM_TEXTURE_V], 2);


	glViewport(0, 0, backingWidth, backingHeight);
	glClearColor(0.0, 0.0, 0.0, 1);
	glClear(GL_COLOR_BUFFER_BIT);

	int screenW = backingWidth;
	int screenH = backingHeight;
	float aspectratio = TEST_YUV_WIDTH / TEST_YUV_HEIGHT;
	h = screenH;
	w = screenW;

	x = xpos * screenW;
	y = ypos * screenH;
	LOGI("x %d,y %d,w %d,h %d\n",x,y,w,h);
	squareVertices[0] = (x - w * 0.5) / screenW - 0.;
	squareVertices[1] = (y - h * 0.5) / screenH - 0.;
	squareVertices[2] = (x + w * 0.5) / screenW - 0.;
	squareVertices[3] = (y - h * 0.5) / screenH - 0.;
	squareVertices[4] = (x - w * 0.5) / screenW - 0.;
	squareVertices[5] = (y + h * 0.5) / screenH - 0.;
	squareVertices[6] = (x + w * 0.5) / screenW - 0.;
	squareVertices[7] = (y + h * 0.5) / screenH - 0.;

	GLfloat mat[16];
	float zoom_factor = 1.0f;
	float zoom_cx = 0.0f;
	float zoom_cy = 0.0f;
#define VP_SIZE 1.0f
	float scale_factor = 1.0 / zoom_factor;
	float vpDim = (VP_SIZE * scale_factor) / 2;

#define ENSURE_RANGE_A_INSIDE_RANGE_B(a, aSize, bMin, bMax) \
		if (2*aSize >= (bMax - bMin)) \
		a = 0; \
		else if ((a - aSize < bMin) || (a + aSize > bMax)) {  \
			float diff; \
			if (a - aSize < bMin) diff = bMin - (a - aSize); \
			else diff = bMax - (a + aSize); \
			a += diff; \
		}

	ENSURE_RANGE_A_INSIDE_RANGE_B(zoom_cx, vpDim, squareVertices[0], squareVertices[2])
	ENSURE_RANGE_A_INSIDE_RANGE_B(zoom_cy, vpDim, squareVertices[1], squareVertices[7])

	load_orthographic_matrix(
			zoom_cx - vpDim,
			zoom_cx + vpDim,
			zoom_cy - vpDim,
			zoom_cy + vpDim,
			0, 0.5, mat);

	glUniformMatrix4fv(uniforms[UNIFORM_PROJ_MATRIX], 1, GL_FALSE, mat);
#define degreesToRadians(d) (2.0 * 3.14157 * d / 360.0)
	float rad = degreesToRadians(0);

	glUniform1f(uniforms[UNIFORM_ROTATION], rad);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, spriteTexture[Y]);
	glUniform1i(uniforms[UNIFORM_TEXTURE_Y], 0);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, spriteTexture[U]);
	glUniform1i(uniforms[UNIFORM_TEXTURE_U], 1);
	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, spriteTexture[V]);
	glUniform1i(uniforms[UNIFORM_TEXTURE_V], 2);

	glVertexAttribPointer(ATTRIB_VERTEX, 2, GL_FLOAT, 0, 0, squareVertices);
	glEnableVertexAttribArray(ATTRIB_VERTEX);
	glVertexAttribPointer(ATTRIB_UV, 2, GL_FLOAT, 1, 0, squareUvs);
	glEnableVertexAttribArray(ATTRIB_UV);

	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

}

extern "C" {
JNIEXPORT void JNICALL Java_com_android_gl2jni_GL2JNILib_init(JNIEnv * env, jobject obj,  jint width, jint height);
JNIEXPORT void JNICALL Java_com_android_gl2jni_GL2JNILib_step(JNIEnv * env, jobject obj);
};

JNIEXPORT void JNICALL Java_com_android_gl2jni_GL2JNILib_init(JNIEnv * env, jobject obj,  jint width, jint height)
{
	setupGraphics(width, height);
}

JNIEXPORT void JNICALL Java_com_android_gl2jni_GL2JNILib_step(JNIEnv * env, jobject obj)
{
	renderFrame();
}
