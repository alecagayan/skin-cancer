let net;


async function load_mobilenet() {
  console.log('Loading mobilenet...');

  // Load the model.
  net = await mobilenet.load();
  console.log('Successfully loaded model');

  const consoleEl = document.getElementById('console');
  consoleEl.style.display = "none";

  const fcEl = document.getElementById('filecontrol');
  fcEl.style.display = "block";
    
}

async function predict() {
    console.log('Predicting...');
  
    // Make a prediction through the model on our image.
    const imgEl = document.getElementById('img');
    const result = await net.classify(imgEl);
    console.log(result);

    const predictionEl = document.getElementById('prediction');
    predictionEl.innerHTML = result[0].className;

    const accuracyEl = document.getElementById('accuracy');
    accuracyEl.innerHTML = "Accuracy: " + result[0].probability;

  }
  
load_mobilenet();

