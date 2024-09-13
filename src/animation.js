// function to load txt data
async function load_data(path) {
    var response = await fetch(path);
    var data = await response.text();
    return data
        .trim()
        .replace(/(\r)/gm, '')
        .split('\n')
        .map(line => line.split(' ').map(x => parseFloat(x)));
}


// function to create line object
function createLine(p1, p2, color, alpha) {
    var geometry = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(p1[0], p1[1], 0),
        new THREE.Vector3(p2[0], p2[1], 0)
    ]);
    var material = new THREE.LineBasicMaterial({ 
        color: new THREE.Color(...color),
        transparent: true,
        opacity: alpha
    });
    return new THREE.Line(geometry, material);
}


// Function to draw lines with interval
function drawLines(lines, scene) {
    var interval = 1;
    var index = 0;
    var intervalId = setInterval(function() {
        if (index < lines.length) {
            var line = lines[index];
            scene.add(line);
            index++;
            document.getElementById('frame').innerHTML = index
        } else {
            clearInterval(intervalId);
        }
    }, interval);
}


// Animation loop
function animate(scene, camera, renderer) {
    requestAnimationFrame(() => animate(scene, camera, renderer));
    renderer.render(scene, camera);
}


// Function to setup scene
function sceneSetup() {
    var scene = new THREE.Scene();

    var aspectRatio = 1;
    var cameraWidth = 2;
    var cameraHeight = cameraWidth / aspectRatio;
    var camera = new THREE.OrthographicCamera(-cameraWidth / 2, cameraWidth / 2, cameraHeight / 2, -cameraHeight / 2, 0.1, 1000);
    camera.position.z = 5;

    var renderer = new THREE.WebGLRenderer();

    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);
    var dimension = Math.min(window.innerWidth, window.innerHeight)
    renderer.setSize(dimension, dimension);

    window.addEventListener('resize', function() {
        var dimension = Math.min(window.innerWidth, window.innerHeight)
        renderer.setSize(dimension, dimension);
        camera.aspect = 1;
        camera.updateProjectionMatrix();
    });

    return [scene, camera, renderer];

}

async function main() {

    // load data
    var path = await load_data('results/path.txt');
    var palette = await load_data('results/palette.txt');

    // calculate pin locations
    var num_points = 256;
    var theta = 2 * Math.PI / num_points;
    var points = [...Array(num_points)].map((_, i) => [Math.sin(theta * i), -Math.cos(theta * i)])

    // create lines
    var lines = [];
    var p1, p2, ic;
    for (let i = 0; i < path.length; i++) {
        [p1, p2, ic] = path[i];
        lines.push(createLine(points[p1], points[p2], palette[ic], 0.25));
    }

    // setup scene
    var scene, camera, renderer;
    [scene, camera, renderer] = sceneSetup();

    lines.map(line => scene.add(line))
    // drawLines(lines, scene);
    animate(scene, camera, renderer);
}
main();
