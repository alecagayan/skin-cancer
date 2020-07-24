skin_classes = {
    0: 'akiec, Actinic Keratoses', // no bueno
    1: 'bcc, Basal Cell Carcinoma', // no bueno
    2: 'bkl, Benign Keratosis', //bueno
    3: 'df, Dermatofibroma', //bueno
    4: 'mel, Melanoma', // no bueno
    5: 'nv, Melanocytic Nevi', //bueno
    6: 'vasc, Vascular Skin Lesion' // why even bother go to the doctor
  };

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
(async function () {
	$("#lesion-image").attr("src", "samplepic.jpg");

    console.log("loading model");
    model = await tf.loadModel('model.json');
	
    //setTimeout(predict(), 500);
    await predict();
	
})();

$("#lesion-file").change(async function () {
    console.log("change called");
 
    let reader = new FileReader();
	reader.onload = function () {
		let dataUrl = reader.result;
		$("#lesion-image").attr("src", dataUrl);
        console.log('set new image URL');
		//$("#lesion-results").empty();
	}
	
		
    let file = $("#lesion-file").prop('files')[0];

    console.log('file - ' + file);
    reader.readAsDataURL(file);
    
    
//    setTimeout(predict(), 500);
    await predict();
 
});


async function predict() {
    console.log("predict called");
    let image = $('#lesion-image').get(0);

    //console.log(image);
	
	// Pre-process the image
	let tensor = tf.fromPixels(image).resizeNearestNeighbor([224,224]).toFloat();	
	let offset = tf.scalar(127.5);
	tensor = tensor.sub(offset).div(offset).expandDims();

    console.log(tensor);
    //console.log(model);
  
    let predictions = await model.predict(tensor).data();
    console.log(predictions);
    

	let top5 = Array.from(predictions)
		.map(function (p, i) { 
            //console.log('array - ' + i);
			return {
				probability: p,
                className: skin_classes[i],
                badness: cancerous[i]
			};
				
			
		}).sort(function (a, b) {
			return b.probability - a.probability;
				
		}).slice(0, 7);
	
    //console.log('clear');
	
    $("#lesion-banner").html("Analysis Results:");
    $("#lesion-results").empty();
    
    top5.forEach(function (p) {
        //console.log('append - ' + p.className);
        var color = 'green';
        if (p.badness == 'malignant') {
            color = 'red';
        }
        console.log('color - ' + color);
        $("#lesion-results").append(`<li><span style='color:${color}'>${p.className}: ${p.probability.toFixed(3)}</span></li>`);
    });

};



