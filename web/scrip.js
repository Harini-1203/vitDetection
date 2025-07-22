function uploadDataset() {
    let files = document.getElementById("fileInput").files;
    let filePaths = Array.from(files).map(file => file.path);
    eel.uploadDataset(filePaths)(function(result) {
        alert(result);
    });
}


function dataPreprocessing() {
    eel.data_preprocessing()(function(result) {
        document.getElementById('output').innerText = result;
    });
}

function runSVM() {
    eel.run_svm()(function(result) {
        document.getElementById('output').innerText = result;
    });
}

function runCNN() {
    eel.run_cnn()(function(result) {
        document.getElementById('output').innerText = result;
    });
}

function predictDeficiency() {
    eel.predict()(function(result) {
        document.getElementById('output').innerText = result;
    });
}
