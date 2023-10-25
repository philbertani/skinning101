"use strict";

function main() {
  // Get A WebGL context
  /** @type {HTMLCanvasElement} */
  const canvas = document.querySelector("#canvas");
  const gl = canvas.getContext("webgl2",{antialias:true, alpha:false});
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
  in vec4 a_normal;

  out vec4 normalVec;
  out vec4 viewZ;
  out vec4 pos;
  out vec4 eyeCoords;  //position transoformed by the View-World Mat

  uniform mat4 projection;
  uniform mat4 view;
  uniform mat3 normal;
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

    mat4 b0 = getBoneMatrix(a_boneNdx[0]);
    mat4 b1 = getBoneMatrix(a_boneNdx[1]);
    mat4 b2 = getBoneMatrix(a_boneNdx[2]);
    mat4 b3 = getBoneMatrix(a_boneNdx[3]);

    vec4 p0 = b0 * a_position * a_weight[0];
    vec4 p1 = b1 * a_position * a_weight[1];
    vec4 p2 = b2 * a_position * a_weight[2];
    vec4 p3 = b3 * a_position * a_weight[3];

    vec4 newPos = p0 + p1 + p2 + p3;

    eyeCoords = view * newPos;
    gl_Position = projection * eyeCoords;

    //we really need the inverse transposes of the bonematrices
    //send another texture with those so we don't waste GPU
    //don't need transpose since we are multiplying vector on right side
    
    normalVec.xyz = normal * (
         inverse(transpose(b0))*a_normal*a_weight[0]
       + inverse(transpose(b1))*a_normal*a_weight[1] 
       + inverse(transpose(b2))*a_normal*a_weight[2] 
       + inverse(transpose(b3))*a_normal*a_weight[3]
    ).xyz;
    
    //normalVec = a_normal;

    viewZ = view * vec4(0,0,1,1); //view[3]; //view * vec4(0,0,2,1); //view[2].xyz;

    pos = gl_Position;

  }
  `;

  const fs = `#version 300 es
  precision highp float;
  uniform vec4 color;
  out vec4 outColor;
  in vec4 normalVec;
  in vec4 viewZ;
  in vec4 pos;
  in vec4 eyeCoords;

  void main () {

    //the most elegant solution to see all the detail pop out
    //flat shading by directly computing the face normal with hardware derivatives
    vec3 dx = dFdx( eyeCoords.xyz );
    vec3 dy = dFdy( eyeCoords.xyz );
    vec3 norm = normalVec.xyz; //normalize(cross(dx,dy)); //normalVec.xyz
    
    vec3 light = -normalize(viewZ.xyz+vec3(0,0,0));
    vec3 halfv = light; //normalize(light - viewZ.xyz);

    float sign = gl_FrontFacing ? 1. : -1.;  //set to 1 if using dx,dy
    float diffuse = max(0.,dot(light, sign*norm));
    float spec = pow( max(0.,dot(halfv, sign*norm)), 16.);

    outColor = vec4( vec3(0,.01,.2)+ diffuse + 8.*spec*vec3(0,1,0),1.);

    //change the color of the inside of the tube
    if (!gl_FrontFacing) outColor.xyz *= 3.*vec3(0,.5,1);
   
    //fade with distance
    outColor *= exp(-pos.w*pos.w/50.);

    outColor = 1.- exp(-outColor*outColor);

  }
  `;


  const fsy = `#version 300 es
  precision highp float;
  uniform vec4 color;
  out vec4 outColor;
  in vec4 normalVec;
  in vec4 viewZ;
  in vec4 pos;
  in vec4 eyeCoords;

  void main () {

  }
  `;

  const fsx = `#version 300 es
  precision highp float;
  uniform vec4 color;
  out vec4 outColor;
  in vec4 normalVec;
  in vec4 viewZ;
  in vec4 pos;

  void main () {
    vec3 light = -viewZ.xyz; //normalize(vec3(.5,1,-1));
    light.y -= 3.; light = normalize(light);
    float depth = pos.w;
    float sign = gl_FrontFacing ? 1. : -1.;

    vec3 halfV =  .5*(-viewZ.xyz + light);  //using the half vector instead of reflect()
    //float spec = max(0.,dot( halfV, -normalVec.xyz));
    float spec = dot( halfV, -sign*normalVec.xyz);
    spec = pow(spec,4.);

    //float diffuse = max(0.,dot(-normalVec.xyz,light ));
    float diffuse = dot( -sign*normalVec.xyz,light ); //allow it be negative

    diffuse *= diffuse; diffuse*=diffuse;
    diffuse = 1. - exp(-diffuse);
    vec3 total = vec3(0,.7,.5) + (diffuse + 2.*spec)  * vec3(1,.8,.4);

    total *=total; total*=total;
    total = 1. - exp(-total);
    total *= exp(-depth*depth/50.);
    outColor = vec4(total, .9);   //color;

  }
  `;


  const vs2 = `#version 300 es
  in vec4 a_position;

  uniform mat4 projection;
  uniform mat4 view;
  uniform mat4 model;

  void main() {
    //gl_PointSize = 5.;
    gl_Position = projection * view * model * a_position;
  }
  `;

  const fs2 = `#version 300 es
  precision highp float;
  uniform vec4 color;
  out vec4 outColor;
  void main () {
    outColor = color;
  }
  `;

  function computeNormal(a, b, vertexArray, num = 3) {
    const an = a * num;
    const v = [vertexArray[an], vertexArray[an + 1], vertexArray[an + 2]];
    const bn = b * num;
    const w = [vertexArray[bn], vertexArray[bn + 1], vertexArray[bn + 2]];

    //console.log(a, b, v, w, vertexArray, num, vertexArray[a*num]);

    const crx = [
      v[1] * w[2] - v[2] * w[1],
      v[2] * w[0] - v[0] * w[2],
      v[0] * w[1] - v[1] * w[0],
    ];

    return normalize(crx);
  }

  function vecSub(v,w) {
    const out=[];
    for (let i=0;i<v.length;i++) {
      out.push(v[i]-w[i]);
    }
    return out;
  }

  function vecAdd(v,w) {
    const out=[];
    for (let i=0;i<v.length;i++) {
      out.push(v[i]+w[i]);
    }
    return out;
  }

  function vecScalar(v,s) {
    const out=[];
    for (let i=0;i<v.length;i++) {
      out.push(v[i]*s);
    }
    return out;    
  }

  function normalize(v) {
    let length = 0;
    for (let i = 0; i < v.length; i++) {
      length += v[i] * v[i];
    }
    length = Math.sqrt(length);
    const out = [];
    for (let i = 0; i < v.length; i++) {
      out.push(v[i] / length);
    }
    return out;
  }

  // compiles and links the shaders, looks up attribute and uniform locations
  const programInfo = twgl.createProgramInfo(gl, [vs, fs]);

  const cos = Math.cos, sin = Math.sin;
  function createArrays() {
    //get roots of unity as base vertices
    const n = 5;
    const radius = .6;
    const height = 7;
    const numLevels = 10;
    const dx = height / numLevels;
    const numVertices = n * numLevels;

    const baseVertices = [];
    const cylinder = [];
    const normals = [];
    for (let i = 0; i < n; i++) {
      const angle = (i * 2 * Math.PI) / n;
      const ca = cos(angle),
        sa = sin(angle);
      baseVertices.push([0, radius*cos(angle), radius*sin(angle)]);
      cylinder.push(0, radius*ca, radius*sa); //make flattened array at same time
      normals.push( 0, ca, sa); //normals are in line with the vertex and perpendicular to axis of cylinder
    }

    for (let h = dx; h < height; h += dx) {
      const center = [h,0,0];  //center of the cylinder at this level
      const newRadius = 1; //radius; //- h/20.;
      for (let i = 0; i < n; i++) {
        const newVertex = [
          newRadius*baseVertices[i][0] + h,
          newRadius*baseVertices[i][1],
          newRadius*baseVertices[i][2]         
        ]
        cylinder.push( ...newVertex );  //flatten it for WebGL

        const norm = normalize(vecSub(newVertex,center)); //normals are in line with the vertex and perpendicular to axis of cylinder
        normals.push(...norm);
        
      }
    }

    const indices = [];
    const normMap = {};
    for (let h = 0; h < numLevels - 1; h += 1) {
      for (let i = 0; i < n; i++) {
        const j = h * n;
        const j1 = h * n;
        const j2 = (h + 1) * n;
        const ii = (i + 1) % n;
        const [p0, p1, p2] = [j1 + i, j1 + ii, j2 + ii];
        indices.push(p0, p1, p2);

        //we do not need to do this for face normals for a cylincer
        const normal = computeNormal(p0, p1, cylinder);
        addToMap(p0, cylinder, normMap);
        addToMap(p1, cylinder, normMap);
        addToMap(p2, cylinder, normMap);

        const [p3, p4, p5] = [j1 + i, j2 + ii, j2 + i];
        indices.push(p3, p4, p5);
        addToMap(p3, cylinder, normMap);
        addToMap(p4, cylinder, normMap);
        addToMap(p5, cylinder, normMap);
  
      }
    }

    console.log(normMap);
    const flatNorms = [];
    const keys = Object.keys(normMap).sort((a,b)=>a-b); //just in case not in order
    for (const key of keys) {
      const normalVec = normalize( normMap[key].vtx );  //dont need count - normalize will take care of everything
      flatNorms.push( ...normalVec );
    }
    console.log(flatNorms);

    function addToMap(pos, vertexArray, M) {
      const start=pos*3;
      if (!M[pos]) {
        M[pos] = {};
        M[pos].vtx = [0,0,0];
        M[pos].count = 0;
      }
      const copyVec = [...M[pos].vtx];
      M[pos].vtx = vecAdd( copyVec, [vertexArray[start],vertexArray[start+1],vertexArray[start+2]] );  //add together so we can average it
      M[pos].count ++;
    }

    //console.log(indices);

    const template = [
      [0, 0, 0, 0],
      [0, 1, 0, 0],
      [1, 0, 0, 0],
      [1, 2, 0, 0],
      [2, 1, 0, 0],
      [2, 0, 0, 0],
    ];

    const weightTemplate = [
      [1,0,0,0],
      [.5,.5,0,0],
      [1,0,0,0],
      [.5,.5,0,0],
      [1,0,0,0],
      [.5,.5,0,0]
    ]

    const divisor = Math.trunc(numVertices / template.length) ;
    const boneNdx = [];
    const weight = [];
    for (let i = 0; i < numVertices; i++) {
      let ii = Math.trunc(i / divisor);
      ii = Math.max(0,Math.min(ii,template.length-1));
      boneNdx.push(...template[ii]);
      weight.push(...weightTemplate[ii]);
    }

    // console.log(boneNdx);
    //console.log(weight);

    return {
      position: { numComponents: 3, data: new Float32Array(cylinder) },
      boneNdx: { numComponents: 4, data: new Uint8Array(boneNdx) },
      weight: { numComponents: 4, data: new Float32Array(weight) },
      indices: { numComponents: 2, data: new Uint16Array(indices) },
      normal: { numComponents: 3, data: new Float32Array(normals) },
    };
  }

  //see index-orig.mjs for the original "arrays" structure

  const a2 = createArrays();
  console.log(a2);

  // calls gl.createBuffer, gl.bindBuffer, gl.bufferData
  const bufferInfo = twgl.createBufferInfoFromArrays(gl, a2);
  const skinVAO = twgl.createVAOFromBufferInfo(gl, programInfo, bufferInfo);

  // 4 matrices, one for each bone
  const numBones = 4;
  const boneArray = new Float32Array(numBones * 16);

  console.log("bones 0",boneArray);

  // prepare the texture for bone matrices
  var boneMatrixTexture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, boneMatrixTexture);
  // since we want to use the texture for pure data we turn
  // off filtering
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

  const FOV = Math.PI / 3.;

  const uniforms = {
    projection: m4.perspective(FOV, 1, 0.1, 10, undefined), //m4.orthographic(-20, 20, -10, 10, -1, 1),
    view: m4.translation(0, 0, -2), //m4.translation(-6, 0, 0),
    boneMatrixTexture,
    color: [1, 0, 0, 1],
    normal: m4.mat3identity(),
  };

  console.log("bb0",boneArray);
  // make views for each bone. This lets all the bones
  // exist in 1 array for uploading but as separate
  // arrays for using with the math functions
  const boneMatrices = []; // the uniform data
  const bones = []; // the value before multiplying by inverse bind matrix
  const bindPose = []; // the bind matrix
  for (let i = 0; i < numBones; ++i) {
    //each different boneMatrix points to a different section of the same boneArray buffer
    boneMatrices.push(new Float32Array(boneArray.buffer, i * 4 * 16, 16));
    bindPose.push(m4.identity()); // just allocate storage
    bones.push(m4.identity()); // just allocate storage
    console.log("bones",i,boneMatrices);
  }
   
  console.log("bb",boneArray);
  console.log("bones",boneMatrices);

  // rotate each bone by a and simulate a hierarchy
  function computeBoneMatrices(bones, angle) {
    const m = m4.identity();
    m4.xRotate(m, angle, bones[0]);
    m4.translate(bones[0], 4, 0, 0, m);
    m4.xRotate(m, angle*1.2, bones[1]);
    m4.translate(bones[1], 4, 0, 0, m);
    m4.xRotate(m, angle*2, bones[2]);
    // bones[3] is not used
  }

  // compute the initial positions of each matrix
  computeBoneMatrices(bindPose, 0);

  console.log( "bones2", boneMatrices);

  // compute their inverses
  const bindPoseInv = bindPose.map(m=>m4.inverse(m));

  //const bindPoseInv = bindPose.map(function (m) {
  //  return m4.inverse(m);
  //});

  //gl.enable(gl.CULL_FACE);
   
  gl.enable(gl.DEPTH_TEST);
  //gl.enable(gl.BLEND);
  //gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);  
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  let frame = 0;
  function render(time) {

    if (frame <  2) console.log(boneArray);

    frame ++;

    twgl.resizeCanvasToDisplaySize(gl.canvas);
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    const aspect = gl.canvas.clientWidth / gl.canvas.clientHeight;
    //m4.orthographic(-aspect * 10, aspect * 10, -10, 10, -1, 1, uniforms.projection);
    m4.perspective(FOV, aspect, 0.01, 100, uniforms.projection);

    const t = time * 0.001;
    const angle = .5*Math.sin(t*3 );
    m4.multiply(uniforms.view, m4.yRotation(0.001), uniforms.view);
    uniforms.normal = m4.normalFromMat4(uniforms.view);

    computeBoneMatrices(bones, angle);

    // multiply each by its bindPoseInverse
    bones.forEach(function (bone, ndx) {
      m4.multiply(bone, bindPoseInv[ndx], boneMatrices[ndx]);
    });

    gl.useProgram(programInfo.program);

    gl.bindVertexArray(skinVAO);


    // update the texture with the current matrices
    gl.bindTexture(gl.TEXTURE_2D, boneMatrixTexture);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0, // level
      gl.RGBA32F, // internal format
      4, // width 4 pixels, each pixel has RGBA so 4 pixels is 16 values
      numBones, // one row per bone
      0, // border
      gl.RGBA, // format
      gl.FLOAT, // type
      boneArray
    );

    // calls gl.uniformXXX, gl.activeTexture, gl.bindTexture
    twgl.setUniforms(programInfo, uniforms);

    // calls gl.drawArrays or gl.drawIndices
    twgl.drawBufferInfo(gl, bufferInfo, gl.TRIANGLES);

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
      axisProgramInfo = twgl.createProgramInfo(gl, [vs2, fs2]);
      axisBufferInfo = twgl.createBufferInfoFromArrays(gl, {
        position: {
          numComponents: 2,
          data: [0, 0, 1, 0],
        },
      });
      axisVAO = twgl.createVAOFromBufferInfo(
        gl,
        axisProgramInfo,
        axisBufferInfo
      );
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
