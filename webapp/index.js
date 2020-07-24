skin_classes = {
  0: 'akiec, Actinic Keratoses', // no bueno
  1: 'bcc, Basal Cell Carcinoma', // no bueno
  2: 'bkl, Benign Keratosis', //bueno
  3: 'df, Dermatofibroma', //bueno
  4: 'mel, Melanoma', // no bueno
  5: 'nv, Melanocytic Nevi', //bueno
  6: 'vasc, Vascular Skin Lesion' // why even bother go to the doctor
};

class_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'];

cancerous = {
  0: 'malignant',
  1: 'malignant',
  2: 'benign',
  3: 'benign',
  4: 'malignant',
  5: 'benign',
  6: 'benign'
};

// console.log("loading js");

let model;
let myChart;

async function load_model() {
  console.log('Loading model...');

  // Load the model.
  model = await tf.loadModel('model.json');
  console.log('Successfully loaded model');

  $("#console").html("");
  $("#lesion-banner").html("Select an image...");

  const fcEl = document.getElementById('filecontrol');
  fcEl.style.display = "block";
    
}

async function predict() {
    console.log('Predicting...');
    
    $("#lesion-results").empty();
    $("#lesion-banner").html("Processing...");

    // Make a prediction through the model on our image.
    const imgEl = document.getElementById('img');
    console.log(imgEl);

    let image = $('#img').get(0);

	  // Pre-process the image
	  let tensor = tf.fromPixels(image).resizeNearestNeighbor([224,224]).toFloat();	
	  let offset = tf.scalar(127.5);
	  tensor = tensor.sub(offset).div(offset).expandDims();

    console.log(tensor);
    //console.log(model);
  
    let predictions = await model.predict(tensor).data();
    console.log(predictions);
    

	  let top5 = Array.from(predictions).map(function (p, i) { 
            //console.log('array - ' + i);
			return {
				probability: p,
                className: skin_classes[i],
                badness: cancerous[i]
			};
				
			
		}).sort(function (a, b) {
			return b.probability - a.probability;
				
		}).slice(0, 7);

    $("#lesion-banner").html("Analysis Results:");

    top5.forEach(function (p) {
        //console.log('append - ' + p.className);
        if (p.probability >= 1e-3)
        {
          var color = 'green';
          if (p.badness == 'malignant') {
              color = 'red';
          }
          $("#lesion-results").append(`<li><span style='color:${color}'>${p.className}: ${p.probability.toFixed(3)}</span></li>`);
        }
    });

    update_chart(predictions);
  }
  
load_model();
load_chart();

function update_chart(skin_data) {
  myChart.config.data = {
    labels: class_labels,
    datasets: [{
      label: 'Probabilities',
      data: skin_data,
      backgroundColor: 'rgba(255, 99, 132, 1)',
    }]
  };
  myChart.update();
}

function load_chart() {
  var ctx = document.getElementById("myChart").getContext('2d');
  myChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: class_labels,
      datasets: [{
        label: 'Probabilities',
        data: [0,0,0,0,0,0,0],
        backgroundColor: 'rgba(255, 99, 132, 1)',
      }]
    },
    options: {
      responsive: false,
      maintainAspectRatio: false,    
      scales: {
        xAxes: [{
          display: false,
          barPercentage: 1.30,
        }, {
          display: true,
        }],
        yAxes: [{
          ticks: {
            beginAtZero:true
          }
        }]
      }
    }
  });
}