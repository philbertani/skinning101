"use strict";

function main() {
  // Get A WebGL context
  /** @type {HTMLCanvasElement} */
  const canvas = document.querySelector("#canvas");
  const gl = canvas.getContext("webgl2");
  if (!gl) {
    return;
  }

  // Tell the twgl to match position with a_position, n
  // normal with a_normal etc..
  twgl.setAttributePrefix("a_");

  // -- vertex shader --
  const vs = `#version 300 es
  in vec4 a_position;
  in vec4 a_weight;
  in uvec4 a_boneNdx;

  uniform mat4 projection;
  uniform mat4 view;
  uniform sampler2D boneMatrixTexture;
  uniform float numBones;

  mat4 getBoneMatrix(uint boneNdx) {
    return mat4(
      texelFetch(boneMatrixTexture, ivec2(0, boneNdx), 0),
      texelFetch(boneMatrixTexture, ivec2(1, boneNdx), 0),
      texelFetch(boneMatrixTexture, ivec2(2, boneNdx), 0),
      texelFetch(boneMatrixTexture, ivec2(3, boneNdx), 0));
  }

  void main() {

    gl_Position = projection * view *
                  (getBoneMatrix(a_boneNdx[0]) * a_position * a_weight[0] +
                   getBoneMatrix(a_boneNdx[1]) * a_position * a_weight[1] +
                   getBoneMatrix(a_boneNdx[2]) * a_position * a_weight[2] +
                   getBoneMatrix(a_boneNdx[3]) * a_position * a_weight[3]);

  }
  `;
  const fs = `#version 300 es
  precision highp float;
  uniform vec4 color;
  out vec4 outColor;
  void main () {
    outColor = color;
  }
  `;
  const vs2 = `#version 300 es
  in vec4 a_position;

  uniform mat4 projection;
  uniform mat4 view;
  uniform mat4 model;

  void main() {
    gl_Position = projection * view * model * a_position;
  }
  `;

  // compiles and links the shaders, looks up attribute and uniform locations
  const programInfo = twgl.createProgramInfo(gl, [vs, fs]);

  const arrays = {
    position: {
      numComponents: 2,
      data: [
        0,  1,  // 0
        0, -1,  // 1
        2,  1,  // 2
        2, -1,  // 3
        4,  1,  // 4
        4, -1,  // 5
        6,  1,  // 6
        6, -1,  // 7
        8,  1,  // 8
        8, -1,  // 9
      ],
    },
    boneNdx: {
      numComponents: 4,
      data: new Uint8Array([
        0, 0, 0, 0,  // 0
        0, 0, 0, 0,  // 1
        0, 1, 0, 0,  // 2
        0, 1, 0, 0,  // 3
        1, 0, 0, 0,  // 4
        1, 0, 0, 0,  // 5
        1, 2, 0, 0,  // 6
        1, 2, 0, 0,  // 7
        2, 0, 0, 0,  // 8
        2, 0, 0, 0,  // 9
      ]),
    },
    weight: {
      numComponents: 4,
      data: [
         1,  0,  0,  0,  // 0
         1,  0,  0,  0,  // 1
        .5, .5,  0,  0,  // 2
        .5, .5,  0,  0,  // 3
         1,  0,  0,  0,  // 4
         1,  0,  0,  0,  // 5
        .5, .5,  0,  0,  // 6
        .5, .5,  0,  0,  // 7
         1,  0,  0,  0,  // 8
         1,  0,  0,  0,  // 9
      ],
    },

    indices: {
      numComponents: 2,
      data: [
        0, 1,
        0, 2,
        1, 3,
        2, 3, //
        2, 4,
        3, 5,
        4, 5,
        4, 6,
        5, 7, //
        6, 7,
        6, 8,
        7, 9,
        8, 9,
      ],
    },
  };
  // calls gl.createBuffer, gl.bindBuffer, gl.bufferData
  const bufferInfo = twgl.createBufferInfoFromArrays(gl, arrays);
  const skinVAO = twgl.createVAOFromBufferInfo(gl, programInfo, bufferInfo);

  // 4 matrices, one for each bone
  const numBones = 4;
  const boneArray = new Float32Array(numBones * 16);

  // prepare the texture for bone matrices
  var boneMatrixTexture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, boneMatrixTexture);
  // since we want to use the texture for pure data we turn
  // off filtering
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

  const uniforms = {
    projection: m4.orthographic(-20, 20, -10, 10, -1, 1),
    view: m4.translation(-6, 0, 0),
    boneMatrixTexture,
    color: [1, 0, 0, 1],
  };

  // make views for each bone. This lets all the bones
  // exist in 1 array for uploading but as separate
  // arrays for using with the math functions
  const boneMatrices = [];  // the uniform data
  const bones = [];         // the value before multiplying by inverse bind matrix
  const bindPose = [];      // the bind matrix
  for (let i = 0; i < numBones; ++i) {
    boneMatrices.push(new Float32Array(boneArray.buffer, i * 4 * 16, 16));
    bindPose.push(m4.identity());  // just allocate storage
    bones.push(m4.identity());     // just allocate storage
  }

  // rotate each bone by a and simulate a hierarchy
  function computeBoneMatrices(bones, angle) {
    const m = m4.identity();
    m4.zRotate(m, angle, bones[0]);
    m4.translate(bones[0], 4, 0, 0, m);
    m4.zRotate(m, angle, bones[1]);
    m4.translate(bones[1], 4, 0, 0, m);
    m4.zRotate(m, angle, bones[2]);
    // bones[3] is not used
  }

  // compute the initial positions of each matrix
  computeBoneMatrices(bindPose, 0);

  // compute their inverses
  const bindPoseInv = bindPose.map(function(m) {
    return m4.inverse(m);
  });

  function render(time) {
    twgl.resizeCanvasToDisplaySize(gl.canvas);
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    const aspect = gl.canvas.clientWidth / gl.canvas.clientHeight;
    m4.orthographic(-aspect * 10, aspect * 10, -10, 10, -1, 1, uniforms.projection);

    const t = time * 0.001;
    const angle = Math.sin(t) * 0.8;
    computeBoneMatrices(bones, angle);

    // multiply each by its bindPoseInverse
    bones.forEach(function(bone, ndx) {
      m4.multiply(bone, bindPoseInv[ndx], boneMatrices[ndx]);
    });

    gl.useProgram(programInfo.program);

    gl.bindVertexArray(skinVAO);

    // update the texture with the current matrices
    gl.bindTexture(gl.TEXTURE_2D, boneMatrixTexture);
    gl.texImage2D(
        gl.TEXTURE_2D,
        0,           // level
        gl.RGBA32F,  // internal format
        4,           // width 4 pixels, each pixel has RGBA so 4 pixels is 16 values
        numBones,    // one row per bone
        0,           // border
        gl.RGBA,     // format
        gl.FLOAT,    // type
        boneArray);

    // calls gl.uniformXXX, gl.activeTexture, gl.bindTexture
    twgl.setUniforms(programInfo, uniforms);

    // calls gl.drawArrays or gl.drawIndices
    twgl.drawBufferInfo(gl, bufferInfo, gl.LINES);

    drawAxis(uniforms.projection, uniforms.view, bones);

    requestAnimationFrame(render);
  }
  requestAnimationFrame(render);

  // --- ignore below this line - it's not relevant to the exmample and it's kind of a bad example ---

  let axisProgramInfo;
  let axisBufferInfo;
  let axisVAO;
  function drawAxis(projection, view, bones) {
    if (!axisProgramInfo) {
      axisProgramInfo = twgl.createProgramInfo(gl, [vs2, fs]);
      axisBufferInfo = twgl.createBufferInfoFromArrays(gl, {
        position: {
          numComponents: 2,
          data: [
            0, 0,
            1, 0,
          ],
        },
      });
      axisVAO = twgl.createVAOFromBufferInfo(gl, axisProgramInfo, axisBufferInfo);
    }

    const uniforms = {
      projection: projection,
      view: view,
    };

    gl.useProgram(axisProgramInfo.program);
    gl.bindVertexArray(axisVAO);

    for (let i = 0; i < 3; ++i) {
      drawLine(bones[i], 0, [0, 1, 0, 1]);
      drawLine(bones[i], Math.PI * 0.5, [0, 0, 1, 1]);
    }

    function drawLine(mat, angle, color) {
      uniforms.model = m4.zRotate(mat, angle);
      uniforms.color = color;
      twgl.setUniforms(axisProgramInfo, uniforms);
      twgl.drawBufferInfo(gl, axisBufferInfo, gl.LINES);
    }
  }
}

main();
