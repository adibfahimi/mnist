const canvas = document.getElementById("arrayCanvas");
const ctx = canvas.getContext("2d");
const pixelSize = 10;
let arrayData = new Array(28 * 28).fill(0);

let isMouseDown = false;

function handleMouseDown(event) {
  isMouseDown = true;
  updateCanvas(event);
}

function handleMouseUp() {
  isMouseDown = false;
}

function handleMouseMove(event) {
  if (isMouseDown) {
    updateCanvas(event);
  }
}

function handleTouchStart(event) {
  isMouseDown = true;
  updateCanvas(event.touches[0]);
}

function handleTouchEnd() {
  isMouseDown = false;
}

function handleTouchMove(event) {
  if (isMouseDown) {
    event.preventDefault();
    updateCanvas(event.touches[0]);
  }
}

function updateCanvas(event) {
  const rect = canvas.getBoundingClientRect();
  const x = Math.floor((event.clientX - rect.left) / pixelSize);
  const y = Math.floor((event.clientY - rect.top) / pixelSize);

  if (x >= 0 && x < 28 && y >= 0 && y < 28) {
    const index = y * 28 + x;
    arrayData[index] = 1;
    drawPixel(x, y);
  }
}

function drawPixel(x, y) {
  ctx.fillStyle = "white";
  ctx.fillRect(x * pixelSize, y * pixelSize, pixelSize, pixelSize);
}

function initializeCanvas() {
  for (let i = 0; i < 28; i++) {
    for (let j = 0; j < 28; j++) {
      const value = arrayData[i * 28 + j];
      const grayscale = Math.floor(value * 255);
      ctx.fillStyle = `rgb(${grayscale},${grayscale},${grayscale})`;
      ctx.fillRect(j * pixelSize, i * pixelSize, pixelSize, pixelSize);
    }
  }
}

canvas.addEventListener("mousedown", handleMouseDown);
canvas.addEventListener("mouseup", handleMouseUp);
canvas.addEventListener("mousemove", handleMouseMove);

canvas.addEventListener("touchstart", handleTouchStart);
canvas.addEventListener("touchend", handleTouchEnd);
canvas.addEventListener("touchmove", handleTouchMove);

initializeCanvas();

function resetCanvas() {
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, 280, 280);
  arrayData.fill(0);

  document.getElementById("digit").innerHTML = "";
  document.getElementById("probability").innerHTML = "start drawing";
}

const model = new onnx.InferenceSession();
model
  .loadModel("./models/model.onnx")
  .then(() => {
    console.log("model loaded!");
  })
  .catch((err) => {
    console.log(err);
  });

function predict() {
  const inputTensor = new onnx.Tensor(
    new Float32Array(arrayData),
    "float32",
    [1, 1, 28, 28]
  );

  model
    .run([inputTensor])
    .then((output) => {
      const outputTensor = output.values().next().value;
      const predictions = outputTensor.data;
      const max = Math.max(...predictions);
      const index = predictions.indexOf(max);
      const digit = index.toString();
      const probability = ((max / predictions.length) * 100).toFixed(2);
      document.getElementById("digit").innerHTML = `Digit: ${digit}`;
      document.getElementById(
        "probability"
      ).innerHTML = `Probability: ${probability}%`;
      document.getElementById("digit").innerHTML = `Digit: ${digit}`;
      document.getElementById(
        "probability"
      ).innerHTML = `Probability: ${probability}%`;
    })
    .catch((err) => {
      console.log(err);
    });
}
